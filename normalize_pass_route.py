"""Normalize separator spacing in the raw validation data's pass_route field."""

import re


_DOT_SPACING = re.compile(r"\s*\.\s*")


def normalize_pass_route(value: str) -> str:
    """Keep mode names and their order; trim only outer and dot-adjacent spaces."""
    return _DOT_SPACING.sub(".", value.strip())
