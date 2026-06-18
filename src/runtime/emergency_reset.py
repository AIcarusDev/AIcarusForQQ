"""Compatibility wrappers for the legacy emergency reset API."""

from __future__ import annotations

from .maintenance import (
    RESET_COGNITION,
    EmergencyResetResult,
    maintenance_service,
)


def expected_confirmation() -> str:
    """Return the exact string required for the WebUI dangerous action."""
    return maintenance_service.expected_confirmation(RESET_COGNITION)


def is_runtime_epoch_stale(epoch: int) -> bool:
    return maintenance_service.is_runtime_epoch_stale(epoch)


def make_runtime_epoch_checker(epoch: int):
    return maintenance_service.make_runtime_epoch_checker(epoch)


def mark_result_aborted_by_reset(result, epoch: int):
    return maintenance_service.mark_result_aborted_by_reset(result, epoch)


async def perform_emergency_reset() -> EmergencyResetResult:
    """Clear current runtime state and park the bot in no-focus waiting mode."""
    return await maintenance_service.perform_emergency_reset()


__all__ = [
    "EmergencyResetResult",
    "expected_confirmation",
    "is_runtime_epoch_stale",
    "make_runtime_epoch_checker",
    "mark_result_aborted_by_reset",
    "perform_emergency_reset",
]
