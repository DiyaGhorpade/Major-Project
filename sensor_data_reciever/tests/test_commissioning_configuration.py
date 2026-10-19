"""Offline commissioning guardrails; no physical configuration is inferred."""
import json
from pathlib import Path

import pytest

from scheduler import load_settings


ROOT = Path(__file__).resolve().parents[1]


def example():
    # Existing software example values are fixtures, never deployment evidence.
    return json.loads((ROOT / 'scheduler.example.json').read_text())


def write_config(tmp_path, settings):
    path = tmp_path / 'scheduler.local.json'
    path.write_text(json.dumps(settings), encoding='utf-8')
    return path


def test_commissioning_template_marks_unknowns_and_preserves_source_topology(tmp_path):
    skeleton = json.loads((ROOT / 'scheduler.commissioning.example.json').read_text())
    assert skeleton['timezone'] == '' and skeleton['times'] == []
    assert skeleton['csv_file'] == '' and skeleton['journal_file'] == ''
    assert [node['id'] for node in skeleton['nodes']] == ['1', '2']
    for node, count in zip(skeleton['nodes'], (12, 4)):
        assert node['url'] == ''
        assert set(node['sampling_points']) == {str(i) for i in range(1, count + 1)}
        for point, mapping in node['sampling_points'].items():
            assert mapping['record_plant_id'] == int(point)
            assert mapping['plant_id'] is None and mapping['group'] == ''
    with pytest.raises((ValueError, KeyError)):
        load_settings(write_config(tmp_path, skeleton))


def test_completed_template_uses_existing_parser_without_network(tmp_path):
    skeleton = json.loads((ROOT / 'scheduler.commissioning.example.json').read_text())
    fixture = example()
    for key in ('timezone', 'times', 'csv_file', 'journal_file'):
        skeleton[key] = fixture[key]
    for node, known in zip(skeleton['nodes'], fixture['nodes']):
        node['url'] = known['url']
        for point, mapping in node['sampling_points'].items():
            mapping.update(plant_id=known['sampling_points'][point]['plant_id'],
                           group=known['sampling_points'][point]['group'])
    parsed = load_settings(write_config(tmp_path, skeleton))
    assert parsed['csv_file'] == str(tmp_path / fixture['csv_file'])
    assert parsed['nodes'] == fixture['nodes']


@pytest.mark.parametrize('key', [
    'timezone', 'times', 'nodes', 'command_timeout', 'completion_timeout',
    'poll_interval', 'csv_file', 'journal_file',
])
def test_missing_top_level_settings_are_rejected(tmp_path, key):
    settings = example()
    del settings[key]
    with pytest.raises(KeyError):
        load_settings(write_config(tmp_path, settings))


@pytest.mark.parametrize('key', ['id', 'url', 'sampling_points'])
def test_missing_node_settings_are_rejected(tmp_path, key):
    settings = example()
    del settings['nodes'][0][key]
    with pytest.raises(KeyError):
        load_settings(write_config(tmp_path, settings))


@pytest.mark.parametrize('key', ['record_plant_id', 'plant_id', 'group'])
def test_missing_mapping_settings_are_rejected(tmp_path, key):
    settings = example()
    del settings['nodes'][0]['sampling_points']['1'][key]
    with pytest.raises(KeyError):
        load_settings(write_config(tmp_path, settings))


@pytest.mark.parametrize('key,value', [
    ('timezone', ''), ('times', []), ('times', ['9:00']),
    ('times', ['09:00', '09:00']), ('command_timeout', 0),
    ('command_timeout', True), ('completion_timeout', -1),
    ('completion_timeout', float('inf')), ('poll_interval', float('nan')),
    ('journal_file', 'sensor_data.csv'),
])
def test_invalid_schedule_limits_and_shared_paths_are_rejected(tmp_path, key, value):
    settings = example()
    settings[key] = value
    with pytest.raises(ValueError):
        load_settings(write_config(tmp_path, settings))


@pytest.mark.parametrize('url', ['', 'http://PENDING', 'http://localhost',
                                 'https://127.0.0.1', 'http://127.0.0.1/collect'])
def test_unresolved_or_invalid_node_origin_is_rejected(tmp_path, url):
    settings = example()
    settings['nodes'][0]['url'] = url
    with pytest.raises(ValueError):
        load_settings(write_config(tmp_path, settings))


@pytest.mark.parametrize('key,value', [
    ('record_plant_id', 0), ('record_plant_id', 17),
    ('plant_id', None), ('group', ''),
])
def test_unresolved_or_out_of_range_mapping_is_rejected(tmp_path, key, value):
    settings = example()
    settings['nodes'][0]['sampling_points']['1'][key] = value
    with pytest.raises(ValueError):
        load_settings(write_config(tmp_path, settings))


def test_changed_topology_is_rejected_where_parser_enforces_it(tmp_path):
    settings = example()
    settings['nodes'].reverse()
    with pytest.raises(ValueError, match='Node 1 followed by Node 2'):
        load_settings(write_config(tmp_path, settings))
    settings = example()
    settings['nodes'][0]['sampling_points'] = {}
    with pytest.raises(ValueError, match='expected sampling points'):
        load_settings(write_config(tmp_path, settings))
