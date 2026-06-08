from collections.abc import Mapping
from typing import TypeVar

K = TypeVar('K')
V = TypeVar('V')


def make_dict_optional_keys(kvpairs: Mapping[K, V]) -> dict[K, V]:
    """Create a dictionary from the given mapping with only keys whose corresponding value is not None"""
    return {k: v for k, v in kvpairs.items() if v is not None}
