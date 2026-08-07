"""Live-process, exact-type authority broker for Exp425 private writes.

This deliberately depends on no checkpoint, runtime, or predecessor module.
Capabilities are process-local: persistence of one would be an authorization bug.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


_SENTINEL = object()
_authorities: dict[int, tuple[object, str]] = {}
_capabilities: dict[int, tuple["_Capability", int, Mapping[str, object]]] = {}


class _Capability:
    """Uncopyable opaque token whose identity must remain in the live registry."""

    __slots__ = ("__identity",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _SENTINEL:
            raise TypeError("Exp425 capability construction is private")
        self.__identity = object()

    def __copy__(self) -> object:
        raise TypeError("Exp425 capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Exp425 capabilities cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Exp425 capabilities cannot be pickled")


def _register_authority(authority: object, kind: str) -> None:
    if type(kind) is not str or not kind:
        raise ValueError("Exp425 authority kind is invalid")
    identity = id(authority)
    if identity in _authorities or authority is None:
        raise ValueError("Exp425 authority is not registerable")
    _authorities[identity] = (authority, kind)


def _close_authority(authority: object) -> None:
    identity = id(authority)
    entry = _authorities.get(identity)
    if entry is None or entry[0] is not authority:
        return
    del _authorities[identity]
    for token_id, (_, authority_id, _) in tuple(_capabilities.items()):
        if authority_id == identity:
            del _capabilities[token_id]


def _mint(authority: object, /, *, kind: str, **claims: object) -> object:
    authority_id = id(authority)
    authority_entry = _authorities.get(authority_id)
    if authority_entry is None or authority_entry[0] is not authority:
        raise ValueError("Exp425 authority is not live")
    if type(kind) is not str or not kind:
        raise ValueError("Exp425 capability kind is invalid")
    if "kind" in claims or any(type(key) is not str for key in claims):
        raise ValueError("Exp425 capability claims are invalid")
    authority_kind = authority_entry[1]
    allowed = {
        "runtime": frozenset({"native"}),
        "exp425_native": frozenset({"native"}),
        "source": frozenset({"import_begin", "import_stage", "import_promote"}),
        "exp424_import": frozenset({"import_begin", "import_stage", "import_promote"}),
        # The sealed local authority exists solely for broker mechanics tests.
        "local": frozenset({"native", "import_begin", "import_stage", "import_promote"}),
    }.get(authority_kind, frozenset())
    if kind not in allowed:
        raise ValueError("Exp425 authority kind cannot mint this operation")
    token = _Capability(_SENTINEL)
    frozen = MappingProxyType(_freeze_claims({"kind": kind, **claims}))
    _capabilities[id(token)] = (token, authority_id, frozen)
    return token


def _consume(capability: object, /, *, kind: str, **claims: object) -> Mapping[str, object]:
    """Atomically validate and consume a private capability."""
    if type(capability) is not _Capability:
        raise ValueError("Exp425 capability has the wrong type")
    entry = _capabilities.get(id(capability))
    if entry is None or entry[0] is not capability:
        raise ValueError("Exp425 capability is unknown or already consumed")
    _, authority_id, frozen = entry
    if _authorities.get(authority_id) is None:
        del _capabilities[id(capability)]
        raise ValueError("Exp425 capability authority is no longer live")
    requested = _freeze_claims({"kind": kind, **claims})
    # Store primitives name every security-relevant claim they know at the
    # operation boundary.  Additional sealed source claims (attestation and
    # fresh process nonce) remain broker-owned and are returned only after
    # authentic token validation.
    if any(frozen.get(key) != value for key, value in requested.items()):
        raise ValueError("Exp425 capability claims do not match")
    # Deletion happens before returning authorization, preventing a second use
    # even if the caller's transaction later fails.
    del _capabilities[id(capability)]
    return frozen


def _freeze_claims(value: object) -> object:
    """Detach token claims from caller-owned mutable containers."""
    if isinstance(value, Mapping):
        return {str(key): _freeze_claims(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_freeze_claims(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_claims(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_claims(item) for item in value)
    return value
