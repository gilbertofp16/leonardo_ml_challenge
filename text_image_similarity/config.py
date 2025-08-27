# text_image_similarity/config.py
"""
Configuration for the scoring process.
"""

from dataclasses import dataclass
from typing import Literal, Optional

Device = Literal["auto", "cpu", "cuda", "mps"]
Precision = Literal["fp32", "fp16"]


@dataclass(frozen=True)
class Config:
    """
    Configuration for the scoring process.
    """

    # Model batching (compute)
    batch_size: int = 32
    device: Device = "auto"  # "auto" picks cuda/mps/cpu
    precision: Precision = "fp32"  # fp16 only used on CUDA

    # Networking / retries
    timeout_s: int = 10
    retries: int = 2

    # I/O & throughput
    chunksize: Optional[int] = None  # CSV read chunk; None ⇒ max(1024, batch_size*32)
    max_io_workers: int = 16  # concurrent image downloads
