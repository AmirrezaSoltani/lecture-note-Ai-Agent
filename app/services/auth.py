from __future__ import annotations

import secrets
from fastapi import HTTPException, Request, status

from app.config import settings

_PUBLIC_PATH_PREFIXES = ("/healthz", "/readyz", "/metrics", "/static")


def is_public_path(path: str) -> bool:
    return path.startswith(_PUBLIC_PATH_PREFIXES)


def is_authorized(request: Request) -> bool:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False
    try:
        import base64
        decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return False
    return secrets.compare_digest(username, settings.basic_auth_username) and secrets.compare_digest(
        password, settings.basic_auth_password
    )


def enforce_basic_auth(request: Request) -> None:
    if not settings.basic_auth_enabled or is_public_path(request.url.path):
        return
    if is_authorized(request):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )
