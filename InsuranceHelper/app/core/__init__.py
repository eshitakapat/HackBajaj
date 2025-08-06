"""
Core package for Insurance Helper API.

This package contains the core functionality and configuration for the Insurance Helper API,
including application settings, database configuration, and shared utilities.

Modules:
    config: Application configuration and settings management
"""
from typing import List

from .config import Settings, settings

__all__: List[str] = ["settings", "Settings"]
