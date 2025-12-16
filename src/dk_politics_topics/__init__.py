"""
Core package for Folketinget political discourse analysis.

The modules are structured to support offline batch processing now and an
API/web layer later. See `agents.md` for ownership and workflow notes.
"""

from .config import DEFAULT_CONFIG, PipelineConfig

__all__ = ["DEFAULT_CONFIG", "PipelineConfig"]
