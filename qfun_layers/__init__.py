"""QFun-KAN prototype layers and encoding helpers."""

from .amplitude_encoding import create_grid, encode_function, query_grid
from .signed_encoding import mode_a_signed_encode, mode_b_signed_decompose
from .qkan_block import QKANBlock

__all__ = [
    "create_grid",
    "encode_function",
    "query_grid",
    "mode_a_signed_encode",
    "mode_b_signed_decompose",
    "QKANBlock",
]
