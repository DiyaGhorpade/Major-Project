"""Jetson orchestration against mock nodes and the production receiver."""
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs
from urllib.error import URLError

import pytest
import receiver
from scheduler import Scheduler, ProcessLock, load_settings


@pytest.fixture
def settings(tmp_path, output):
    path = Path(__file__).parents[1] / 'scheduler.example.json'
    settings = load_settings(path)
    settings.update(csv_file=str(output), journal_file=str(tmp_path / 'runs.jsonl'),
                    completion_timeout=0.15, poll_interval=0.005, command_timeout=0.1)
    for node in settings['nodes']:
        node['sampling_points'] = {k: v for k, v in node['sampling_points'].items() if k in ('1', '2')}
    return settings


def upload(session, node, point, plant=None):
    with receiver.app.test_client() as client:
        return client.post('/upload', data=f'100,{session},{node},{point},{plant or point},50,23,60,512').status_code


@contextmanager
def mock_node(handler):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/collect'
            assert self.headers['Content-Type'] == 'application/x-www-form-urlencoded'
            form = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode())
            status = handler(int(form['session_id'][0]))
            self.send_response(status)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_http_nodes_wait_for_records_not_acceptance(settings):
    accepted = threading.Event()
    finish = threading.Event()
    node2_called = threading.Event()

    def first(session):
        accepted.set()
        def produce():
            assert finish.wait(3)
            for point in (1, 1, 2):  # Duplicate does not stand in for point 2.
                assert upload(session, '1', point) == 200
        threading.Thread(target=produce, daemon=True).start()
        return 202

    def second(session):
        node2_called.set()
        for point in (1, 2):
            assert upload(session, '2', point) == 200
        return 202

    with mock_node(first) as first_url, mock_node(second) as second_url:
        settings['nodes'][0]['url'], settings['nodes'][1]['url'] = first_url, second_url
        settings['completion_timeout'] = 3
        scheduler = Scheduler(settings)
        results = []
        thread = threading.Thread(target=lambda: results.append(scheduler.run('scheduled-slot')))
        thread.start()
        try:
            assert accepted.wait(2)
            assert not node2_called.wait(0.05)
            assert scheduler.run()['status'] == 'skipped'
            finish.set()
            thread.join(4)
            assert not thread.is_alive()
            assert results[0]['status'] == 'complete'
            assert results[0]['nodes'] == {'1': 'complete', '2': 'complete'}
            journal = [json.loads(line) for line in Path(settings['journal_file']).read_text().splitlines()]
            assert [r['status'] for r in journal] == ['running', 'complete']
            assert journal[0]['nodes'] == {}
            assert Scheduler(settings).run('scheduled-slot')['status'] == 'skipped'
        finally:
            finish.set()
            thread.join(4)


@pytest.mark.parametrize('mode', ['rejected', 'unreachable', 'incomplete', 'duplicate', 'invalid', 'wrong-session', 'wrong-plant'])
def test_failures_never_trigger_node2(settings, mode):
    calls = []
    def command(node, session, timeout):
        calls.append(node['id'])
        assert timeout == settings['command_timeout']
        if mode == 'rejected':
            return 409
        if mode == 'unreachable':
            raise URLError('unreachable')
        if mode == 'duplicate':
            for _ in range(3):
                assert upload(session, '1', 1) == 200
        if mode == 'invalid':
            assert upload(session, '1', 13) == 400
        if mode == 'wrong-session':
            for p in (1, 2):
                assert upload(session - 1, '1', p) == 200
        if mode == 'wrong-plant':
            for p in (1, 2):
                assert upload(session, '1', p, plant=8) == 200
        return 202
    result = Scheduler(settings, command=command).run()
    assert result['status'] == 'failed'
    assert calls == ['1']
    if mode not in ('rejected', 'unreachable'):
        assert 'TimeoutError' in result['error']


def test_http_rejection_and_socket_timeout(settings):
    with mock_node(lambda session: 409) as url:
        settings['nodes'][0]['url'] = url
        assert Scheduler(settings).run()['status'] == 'failed'
    waiting = threading.Event()
    def stalled(session):
        waiting.wait(1)
        return 202
    with mock_node(stalled) as url:
        settings['nodes'][0]['url'] = url
        try:
            assert Scheduler(settings).run()['status'] == 'failed'
        finally:
            waiting.set()


def test_daily_slots_and_restart_do_not_replay(settings):
    calls = []
    def command(node, session, timeout):
        calls.append((node['id'], session))
        for p in (1, 2):
            upload(session, node['id'], p)
        return 202
    scheduler = Scheduler(settings, command=command)
    assert scheduler.tick(datetime(2026, 10, 19, 8, 59)) is None
    assert scheduler.tick(datetime(2026, 10, 19, 9, 0))['status'] == 'complete'
    assert scheduler.tick(datetime(2026, 10, 19, 9, 0, 30))['status'] == 'skipped'
    restarted = Scheduler(settings, command=command)
    assert restarted.tick(datetime(2026, 10, 19, 9, 0))['status'] == 'skipped'
    assert restarted.tick(datetime(2026, 10, 19, 15, 0))['status'] == 'complete'
    assert len(calls) == 4 and calls[2][1] > calls[0][1]


def test_node2_failure_records_node1_completion(settings):
    def command(node, session, timeout):
        if node['id'] == '2':
            return 409
        for point in (1, 2):
            upload(session, '1', point)
        return 202
    result = Scheduler(settings, command=command).run()
    assert result['status'] == 'failed'
    assert result['nodes'] == {'1': 'complete', '2': 'requesting'}


def test_partial_and_invalid_stored_rows_do_not_complete(settings):
    scheduler = Scheduler(settings)
    upload(7, '1', 1)
    with open(settings['csv_file'], 'a') as stream:
        stream.write('partial,row\n')
        stream.write('id,received,accepted,100,7,1,2,2,nan,23,60,512\n')
    assert scheduler.received_points(7, settings['nodes'][0]) == {'1'}
    with open(settings['csv_file'], 'w') as stream:
        stream.write('wrong,header\n')
    with pytest.raises(ValueError, match='header'):
        scheduler.received_points(7, settings['nodes'][0])


def test_process_lock_prevents_second_scheduler(tmp_path):
    path = str(tmp_path / 'scheduler.lock')
    with ProcessLock(path):
        with pytest.raises(RuntimeError, match='already running'):
            with ProcessLock(path):
                pytest.fail('second scheduler acquired lock')
    with ProcessLock(path):
        pass


def test_configuration_validation(tmp_path):
    source = json.loads((Path(__file__).parents[1] / 'scheduler.example.json').read_text())
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(source))
    assert load_settings(path)['csv_file'] == str(tmp_path / 'sensor_data.csv')
    source['completion_timeout'] = float('inf')
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='finite'):
        load_settings(path)
