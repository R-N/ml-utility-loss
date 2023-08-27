
from collections import namedtuple
def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest != "help" and hasattr(action, "default"):
            defaults[action.dest] = action.default
    ArgsDefaults = namedtuple("ArgsDefaults", defaults)
    return ArgsDefaults(**defaults)

def try_argparse(parser, *args, **kwargs):
    if __name__ == '__main__':
        return parser.parse_args(*args, **kwargs)
    return get_argparse_defaults(parser)
