"""Synthesizers module."""

from .ctgan import CTGAN
from .tvae import TVAE

__all__ = (
    'CTGAN',
    'TVAE'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
