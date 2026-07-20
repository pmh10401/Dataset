#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from experiments.exp265_shadow_validation.reporting import EXPECTED_POLICY_VERSION, write_report


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the authenticated Exp265 prospective shadow report.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dashboard-url", required=True, help="Dashboard origin, for example https://dashboard.example")
    parser.add_argument("--access-token-env", default="SUPABASE_ACCESS_TOKEN", help="Environment variable containing the Supabase user session token")
    parser.add_argument("--expected-policy-version", default=EXPECTED_POLICY_VERSION)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    access_token = os.environ.get(args.access_token_env)
    if not access_token:
        raise SystemExit(f"missing access token environment variable: {args.access_token_env}")
    api_url = f"{args.dashboard_url.rstrip('/')}/api/shadow/report"
    request = urllib.request.Request(api_url, headers={"X-Supabase-Access-Token": access_token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise SystemExit(f"report request failed with HTTP {status}") from exc
    paths = write_report(args.output_dir, payload, api_url=api_url, expected_policy_version=args.expected_policy_version)
    print(paths.json)
    print(paths.markdown)
    print(paths.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
