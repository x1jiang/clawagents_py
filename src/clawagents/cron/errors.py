"""Cron error types."""

from __future__ import annotations


class SchedulerError(Exception):
    """Base class for all cron / scheduler failures."""
