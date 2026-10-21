"""Persistent audit records stay bounded, parseable and limited to device fields."""
import json
from types import SimpleNamespace
from demo.join_log import JoinLog, device_metadata


def test_device_metadata_is_bounded_and_allowlisted():
    socket = SimpleNamespace(headers={'user-agent': 'a' * 5000, 'sec-ch-ua': 'b' * 1000,
        'sec-ch-ua-platform': 'c' * 500, 'sec-ch-ua-mobile': '?0',
        'cookie': 'secret', 'authorization': 'secret', 'x-forwarded-for': 'untrusted'},
        client=SimpleNamespace(host='127.0.0.1'))
    metadata = device_metadata(socket)
    assert len(metadata['user_agent']) == 2048
    assert len(metadata['browser_brands']) == 512
    assert len(metadata['platform']) == 128
    assert metadata['mobile'] is False and metadata['connection_address'] == '127.0.0.1'
    assert 'secret' not in str(metadata) and 'untrusted' not in str(metadata)
    assert device_metadata(SimpleNamespace(headers={}, client=None))['connection_address'] is None


def test_join_log_rotates_complete_json_records_with_bounded_backups(tmp_path):
    audit = JoinLog(tmp_path / 'joins.jsonl', max_bytes=600, backup_count=2)
    room = SimpleNamespace(current_track_id='test-track', phase=SimpleNamespace(value='LOBBY'))
    try:
        for index in range(20):
            audit.record(SimpleNamespace(client_id=str(index), nickname=f'Visitor {index}'), room,
                         {'user_agent': 'Test browser'})
    finally:
        audit.close()
    files = list(tmp_path.glob('joins.jsonl*'))
    assert len(files) == 3
    records = [json.loads(line) for file in files for line in file.read_text().splitlines()]
    assert 3 <= len(records) < 20
    assert all(row['event'] == 'participant_joined' for row in records)
    assert json.loads((tmp_path / 'joins.jsonl').read_text().splitlines()[-1])['nickname'] == 'Visitor 19'


def test_unicode_and_line_separators_remain_one_complete_record(tmp_path):
    name = 'Alex\u2028平板\n🙂'
    audit = JoinLog(tmp_path / 'joins.jsonl')
    try:
        audit.record(SimpleNamespace(client_id='client', nickname=name),
                     SimpleNamespace(current_track_id='track', phase=SimpleNamespace(value='LOBBY')),
                     {'user_agent': 'Browser\nwith a newline'})
    finally:
        audit.close()
    lines = (tmp_path / 'joins.jsonl').read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])['nickname'] == name
