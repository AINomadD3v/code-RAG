# my_ops.py
import os

import cocoindex

print(f"--- Loading module: {__file__} ---")  # Add for debugging imports


# Implementation function
def _extract_extension_impl(filename: str) -> str:
    """Extracts the file extension."""
    return os.path.splitext(filename)[1]


# Define and register the operation HERE using the decorator
# Add cache/version parameters if appropriate for your op
@cocoindex.op.function(cache=True, behavior_version=1)
def extract_extension_op_registered(filename: str) -> str:
    """CocoIndex operation to extract file extension."""
    return _extract_extension_impl(filename)


# You could define other custom ops here as well
