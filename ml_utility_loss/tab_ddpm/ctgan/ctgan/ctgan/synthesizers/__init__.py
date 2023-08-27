"""Synthesizers module."""

from .base import BaseSynthesizer, random_state
from .ctgan import CTGANSynthesizer, Discriminator, Generator, Residual
from .tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
