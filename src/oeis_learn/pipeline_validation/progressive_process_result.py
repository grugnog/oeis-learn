"""Read a canonical host result after its supervisor's console heartbeats.

Console progress is not a JSON document. Only the exact supervisor prefix may
precede the single terminal document; callers still verify process termination
and equality with the independently validated original result publication.
"""
import json
from pathlib import Path
import re


_HEARTBEAT = re.compile(r'  running [0-9]+\.[0-9]s; remaining -?[0-9]+\.[0-9]s; logs [^\r\n]+\r?\n')


def stdout_result(path, *, maximum=256 << 20):
    path = Path(path)
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('bounded canonical result stdout required: ' + str(path))

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON member')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)

    try:
        text = data.decode('utf-8')
        start = 0
        while match := _HEARTBEAT.match(text, start):
            start = match.end()
        result = json.loads(text[start:], object_pairs_hook=pairs, parse_constant=nonfinite)
        if not isinstance(result, dict):
            raise ValueError('result object required')
        return result
    except ValueError as exc:
        raise ValueError('invalid canonical result stdout at ' + str(path) + ': ' + str(exc)) from exc
