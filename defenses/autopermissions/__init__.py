"""AutoPermissions-adapted runtime permission defense."""

from .adapter import (
    AutoPermissionsRuntimePolicyPlugin,
    AutoPermissionsService,
    PermissionPrediction,
    PermissionRequest,
)

__all__ = [
    "AutoPermissionsRuntimePolicyPlugin",
    "AutoPermissionsService",
    "PermissionPrediction",
    "PermissionRequest",
]
