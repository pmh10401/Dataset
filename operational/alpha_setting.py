"""Synchronize and load the operational Alpha confidence-label setting."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_ALPHA_VALUES = (0.005, 0.01)
DEFAULT_ALPHA_SOURCE_ID = "macbook-air"
DEFAULT_ALPHA_CONFIG_PATH = Path(
    "/Users/minho/Documents/Dataset/operational_alpha_setting.json"
)


def _normalized_alpha(value):
    alpha = float(value)
    if not any(abs(alpha - allowed) <= 1e-12 for allowed in ALLOWED_ALPHA_VALUES):
        raise ValueError(f"unsupported alpha: {value}")
    return min(ALLOWED_ALPHA_VALUES, key=lambda allowed: abs(alpha - allowed))


def load_effective_alpha(path=DEFAULT_ALPHA_CONFIG_PATH):
    """Return the locally acknowledged Alpha value for detector consumers."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("policy_scope") != "strong_confidence_label_only":
        raise ValueError("unexpected alpha policy scope")
    if payload.get("automatic_hard_alerts_unchanged") is not True:
        raise ValueError("alpha config must preserve automatic Hard alerts")
    return _normalized_alpha(payload["effective_alpha"])


def _write_applied_config(path, source_id, version, alpha):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "version": int(version),
        "effective_alpha": _normalized_alpha(alpha),
        "policy_scope": "strong_confidence_label_only",
        "automatic_hard_alerts_unchanged": True,
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return payload


class OperationalAlphaClient:
    """Minimal PostgREST client using an existing authenticated access token."""

    def __init__(self, supabase_url, publishable_key, timeout=15):
        self.rest_url = supabase_url.rstrip("/") + "/rest/v1"
        self.publishable_key = publishable_key
        self.timeout = timeout

    def _request(self, method, path, access_token, payload=None):
        body = None
        headers = {
            "apikey": self.publishable_key,
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.rest_url + path,
            data=body,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status == 204:
                return None
            return json.load(response)

    def fetch_setting(self, access_token, source_id):
        query = urllib.parse.urlencode({
            "source_id": f"eq.{source_id}",
            "select": (
                "source_id,desired_alpha,effective_alpha,status,version,"
                "reason,requested_at,applied_at,last_error"
            ),
        })
        rows = self._request("GET", f"/operational_alpha_settings?{query}", access_token)
        return rows[0] if rows else None

    def acknowledge(
        self,
        access_token,
        source_id,
        version,
        alpha,
        success,
        error_message=None,
    ):
        return self._request(
            "POST",
            "/rpc/acknowledge_operational_alpha_change",
            access_token,
            {
                "p_source_id": source_id,
                "p_version": int(version),
                "p_alpha": _normalized_alpha(alpha),
                "p_success": bool(success),
                "p_error_message": error_message,
            },
        )


def apply_pending_alpha_setting(
    client,
    access_token,
    source_id=DEFAULT_ALPHA_SOURCE_ID,
    config_path=DEFAULT_ALPHA_CONFIG_PATH,
):
    """Apply one pending request locally, then acknowledge the exact DB version."""
    setting = client.fetch_setting(access_token, source_id)
    if setting is None:
        return {"status": "no_setting", "source_id": source_id}
    if setting.get("status") == "applied":
        version = int(setting["version"])
        alpha = _normalized_alpha(setting["effective_alpha"])
        try:
            local_matches = (
                load_effective_alpha(config_path) == alpha
                and json.loads(Path(config_path).read_text(encoding="utf-8")).get("version") == version
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            local_matches = False
        if not local_matches:
            _write_applied_config(config_path, source_id, version, alpha)
            return {
                "status": "repaired",
                "source_id": source_id,
                "version": version,
                "effective_alpha": alpha,
                "config_path": str(config_path),
            }
        return {
            "status": "no_change",
            "source_id": source_id,
            "version": version,
            "effective_alpha": alpha,
        }
    if setting.get("status") != "pending":
        return {
            "status": "no_change",
            "source_id": source_id,
            "version": setting.get("version"),
            "effective_alpha": setting.get("effective_alpha"),
        }

    version = int(setting["version"])
    try:
        alpha = _normalized_alpha(setting["desired_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        return {
            "status": "failed",
            "source_id": source_id,
            "version": version,
            "error": str(error)[:500],
        }

    try:
        payload = _write_applied_config(config_path, source_id, version, alpha)
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        message = str(error)[:500]
        client.acknowledge(
            access_token,
            source_id,
            version,
            alpha,
            False,
            message,
        )
        return {
            "status": "failed",
            "source_id": source_id,
            "version": version,
            "error": message,
        }

    client.acknowledge(
        access_token,
        source_id,
        version,
        alpha,
        True,
    )
    return {
        "status": "applied",
        "source_id": source_id,
        "version": version,
        "effective_alpha": alpha,
        "config_path": str(config_path),
        "automatic_hard_alerts_unchanged": payload["automatic_hard_alerts_unchanged"],
    }
