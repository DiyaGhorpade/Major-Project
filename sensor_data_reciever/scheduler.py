"""Single Jetson schedule owner. Reads the existing receiver's durable CSV.

Run beside receiver.py; configure both processes to use the same CSV path.
"""
import argparse
import csv
from ipaddress import IPv4Address
import json
import math
import os
from pathlib import Path
import threading
import time
from datetime import datetime
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from zoneinfo import ZoneInfo

import config
import validator


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def send_command(node, session, timeout):
    request = Request(node['url'].rstrip('/') + '/collect',
                      data=urlencode({'session_id': session}).encode(),
                      headers={'Content-Type': 'application/x-www-form-urlencoded'})
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout) as response:
        return response.status  # Acceptance is never completion; no body needed.


def load_settings(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding='utf-8'))
    ZoneInfo(settings['timezone'])
    times = settings['times']
    if not times or len(set(times)) != len(times):
        raise ValueError('Configure unique daily HH:MM collection times')
    for value in times:
        if datetime.strptime(value, '%H:%M').strftime('%H:%M') != value:
            raise ValueError('Collection times must be HH:MM')
    if [str(n['id']) for n in settings['nodes']] != ['1', '2']:
        raise ValueError('Configure Node 1 followed by Node 2')
    for node in settings['nodes']:
        url = urlsplit(node['url'])
        if url.scheme != 'http' or not url.hostname or url.path not in ('', '/') or url.query or url.fragment or url.username:
            raise ValueError('Node url must be an HTTP origin')
        IPv4Address(url.hostname)  # Avoid DNS resolution outside socket timeouts.
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError('Invalid node port')
        points = node['sampling_points']
        if not points:
            raise ValueError('Each node needs expected sampling points')
        for point, mapping in points.items():
            if str(int(point)) != point or not 1 <= int(point) <= 12:
                raise ValueError('Sampling points must be 1..12')
            if not 1 <= mapping['record_plant_id'] <= 16 or not mapping['plant_id'] or not mapping['group']:
                raise ValueError('Each sampling point needs record_plant_id, plant_id and group')
    for key in ('command_timeout', 'completion_timeout', 'poll_interval'):
        value = settings[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('csv_file', 'journal_file'):
        settings[key] = str((path.parent / settings[key]).resolve())
    if settings['csv_file'] == settings['journal_file']:
        raise ValueError('CSV and journal must be different files')
    return settings


class Scheduler:
    def __init__(self, settings, command=send_command, monotonic=time.monotonic,
                 sleep=time.sleep):
        self.settings, self.command = settings, command
        self.monotonic, self.sleep = monotonic, sleep
        self.lock = threading.Lock()
        self.history = []
        journal = Path(settings['journal_file'])
        if journal.exists():
            # Fail closed on damaged history, so session IDs are never reused.
            self.history = [json.loads(line) for line in journal.read_text().splitlines()]

    def record(self, event):
        event = {**event, 'at': datetime.now().astimezone().isoformat()}
        with open(self.settings['journal_file'], 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.history.append(json.loads(json.dumps(event)))

    def received_points(self, session, node):
        points = set()
        try:
            with open(self.settings['csv_file'], newline='', encoding='utf-8') as stream:
                rows = csv.reader(stream, strict=True)
                if next(rows, None) != ['record_id', 'received_at', 'accepted_at', *config.CSV_HEADER]:
                    raise ValueError('Receiver CSV header does not match stored schema')
                for row in rows:
                    # Concurrent append may expose a partial last row: never count it.
                    if len(row) != 12 or not row[0] or not row[1] or not row[2]:
                        continue
                    parsed = validator.validate_record(row[3:])
                    if isinstance(parsed, list):
                        continue
                    point = str(parsed['sampling_point'])
                    mapping = node['sampling_points'].get(point)
                    if (parsed['session_id'] == session and parsed['node_id'] == str(node['id'])
                            and mapping and parsed['plant_id'] == mapping['record_plant_id']):
                        points.add(point)
        except FileNotFoundError:
            pass
        except csv.Error:
            pass  # A later poll will read the completed append.
        return points

    def run(self, slot=None):
        if not self.lock.acquire(blocking=False):
            return {'status': 'skipped', 'reason': 'run already active'}
        try:
            if slot is not None and any(e.get('slot') == slot for e in self.history):
                return {'status': 'skipped', 'reason': 'slot already started'}
            session = max(int(time.time()), max((e['session_id'] for e in self.history), default=0) + 1)
            if session > 2147483647:
                raise ValueError('Session ID exceeds ESP32 signed integer range')
            result = {'session_id': session, 'slot': slot, 'status': 'running', 'nodes': {}}
            self.record(result)  # Durable allocation before any network command.
            try:
                for node in self.settings['nodes']:
                    node_id = str(node['id'])
                    result['nodes'][node_id] = 'requesting'
                    if self.command(node, session, self.settings['command_timeout']) != 202:
                        raise RuntimeError(f'Node {node_id} rejected command')
                    result['nodes'][node_id] = 'accepted'
                    deadline = self.monotonic() + self.settings['completion_timeout']
                    while True:
                        points = self.received_points(session, node)
                        if points == set(node['sampling_points']):
                            break
                        remaining = deadline - self.monotonic()
                        if remaining <= 0:
                            missing = sorted(set(node['sampling_points']) - points)
                            raise TimeoutError(f'Node {node_id} incomplete; missing {missing}')
                        self.sleep(min(self.settings['poll_interval'], remaining))
                    result['nodes'][node_id] = 'complete'
                result['status'] = 'complete'
            except Exception as exc:
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self.record(result)
            return result
        finally:
            self.lock.release()

    def tick(self, now=None):
        now = now or datetime.now(ZoneInfo(self.settings['timezone']))
        if now.strftime('%H:%M') in self.settings['times']:
            return self.run(now.strftime('%Y-%m-%dT%H:%M'))


class ProcessLock:
    """OS lock released on process exit; works on Jetson and Windows tests."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.stream = open(self.path, 'a+b')
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                if not self.stream.read(1):
                    self.stream.write(b'0')
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('Another scheduler is already running') from None
        return self

    def __exit__(self, *args):
        self.stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true', help='Run one collection immediately')
    args = parser.parse_args()
    settings = load_settings(args.config)
    with ProcessLock(settings['journal_file'] + '.lock'):
        scheduler = Scheduler(settings)
        # An interrupted run is failed, never replayed after restart.
        finished = {e['session_id'] for e in scheduler.history if e['status'] != 'running'}
        for event in list(scheduler.history):
            if event['status'] == 'running' and event['session_id'] not in finished:
                scheduler.record({**event, 'status': 'failed', 'error': 'Scheduler interrupted'})
        if args.once:
            result = scheduler.run()
            print(json.dumps(result), flush=True)
            return 0 if result['status'] == 'complete' else 1
        while True:
            result = scheduler.tick()
            if result and result['status'] != 'skipped':
                print(json.dumps(result), flush=True)
            time.sleep(1)


if __name__ == '__main__':
    raise SystemExit(main())
