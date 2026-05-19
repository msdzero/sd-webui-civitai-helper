""" -*- coding: UTF-8 -*-
Module-level storage for manually-added Civitai resources.
These are accumulated via the txt2img / img2img "Civitai Extra Resources"
accordion and merged into the Civitai resource metadata when an image is saved.
"""

_pending = []
_enabled = True


def get_enabled() -> bool:
    return _enabled


def set_enabled(value: bool):
    global _enabled
    _enabled = bool(value)


def get():
    """Return a copy of the current pending resource list."""
    return list(_pending)


def add(resource: dict):
    """Append one resource entry to the pending list."""
    _pending.append(resource)


def remove(idx: int):
    """Remove the resource at position *idx* (0-based)."""
    if 0 <= idx < len(_pending):
        _pending.pop(idx)


def clear():
    """Empty the pending list."""
    global _pending
    _pending = []


def set_weight(idx: int, weight: float):
    """Update the weight of the resource at position idx (lora/lycoris only)."""
    if 0 <= idx < len(_pending):
        _pending[idx]["weight"] = round(float(weight), 4)
