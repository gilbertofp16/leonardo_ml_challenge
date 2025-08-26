"""
Core domain models for the application.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    """
    Represents a single image-text pair to be scored.
    """

    url: str
    caption: str
