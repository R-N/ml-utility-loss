import json
from pathlib import Path

def load_json(path, **kwargs):
    return json.loads(Path(path).read_text(), **kwargs)

def dump_json(x, path, **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')
