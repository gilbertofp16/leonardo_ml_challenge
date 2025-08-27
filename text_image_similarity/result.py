"""
Defines the result objects for the scoring process.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoreResult:
    """
    Represents the outcome of a similarity scoring operation for a single record.
    """

    url: str
    score: float
    error: Optional[str] = None
