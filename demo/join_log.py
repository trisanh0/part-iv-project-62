"""Append-only, rotating join records for post-event review."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


class _AuditHandler(RotatingFileHandler):
    def handleError(self, record):
        # Disk failures must be visible without interrupting the booth or dumping
        # the participant's record into an exception traceback.
        logger.error('join_audit_write_failed; check log storage and available disk space')


class JoinLog:
    def __init__(self, path, max_bytes=5 * 1024 * 1024, backup_count=5):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = _AuditHandler(self.path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))

    def record(self, player, state, metadata):
        data = {
            'event': 'participant_joined',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'client_id': player.client_id,
            'nickname': player.nickname,
            'track_id': state.current_track_id,
            'room_phase': state.phase.value,
            **metadata,
        }
        record = logging.LogRecord('demo.joins', logging.INFO, __file__, 0,
                                   json.dumps(data), (), None)
        self.handler.handle(record)

    def close(self):
        self.handler.close()


def device_metadata(websocket):
    """Keep a small allowlist, never cookies, URLs, credentials or resume tokens."""
    def header(name, limit):
        value = websocket.headers.get(name)
        return value[:limit] if value else None

    mobile = header('sec-ch-ua-mobile', 8)
    return {
        'user_agent': header('user-agent', 2048),
        'browser_brands': header('sec-ch-ua', 512),
        'platform': header('sec-ch-ua-platform', 128),
        'mobile': True if mobile == '?1' else False if mobile == '?0' else None,
        'connection_address': websocket.client.host if websocket.client else None,
    }
