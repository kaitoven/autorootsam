import re
from datetime import datetime

_TS_RE = re.compile(r'(\d{4}\.\d{2}\.\d{2})_(\d{6})')


def parse_timestamp(record: dict):
    """Parse a PRMI record timestamp.

    Priority:
    1) image_name contains YYYY.MM.DD_HHMMSS
    2) date field contains YYYY.MM.DD

    Returns: datetime
    """
    name = str(record.get('image_name',''))
    m = _TS_RE.search(name)
    if m:
        d, t = m.group(1), m.group(2)
        return datetime.strptime(d + t, '%Y.%m.%d%H%M%S')
    d = str(record.get('date',''))
    try:
        return datetime.strptime(d, '%Y.%m.%d')
    except Exception:
        # fallback: try dash format
        try:
            return datetime.strptime(d, '%Y-%m-%d')
        except Exception:
            return datetime.fromtimestamp(0)


def delta_days(t0: datetime, t1: datetime) -> float:
    return max(0.0, (t1 - t0).total_seconds() / 86400.0)
