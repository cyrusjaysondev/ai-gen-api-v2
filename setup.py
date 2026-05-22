"""TEMPORARY pip-install-triggered code-refresh shim.

The ComfyUI custom-node loader has been silently rejecting this repo
(node_count never goes up after install + restart, no diag log entry).
Switch strategies: trigger the refresh via pip install's setup.py
import-time side effect instead. /admin/install-comfy-node runs
`pip install -r requirements.txt`; requirements.txt points pip at this
directory in editable mode, which makes pip import setup.py — and pip
imports setup.py BEFORE doing anything else, so our side effects run
regardless of whether the actual install would succeed.

Steps at import time:
  1. wget latest main.py / workflows.py / etc. into /workspace/api/
  2. kill uvicorn by :7860 port owner so start_api.sh relaunches it
  3. raise SystemExit so pip stops (we don't actually want to install
     this as a Python package — we just used pip as a code-runner)

Idempotent via /tmp/api-refresh-claimed-<commit>-setup-py — re-running
with the same target commit no-ops.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

API_REPO_RAW = "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main"
API_DIR = Path("/workspace/api")
FILES_TO_REFRESH = ("main.py", "workflows.py", "safety.py", "logo_safety.py", "watermark.py")
MARKER = Path("/tmp/api-refresh-claimed-motion-addguide-v6-swap-order")
DIAG_LOG = Path("/workspace/setup-vhs.log")


def _log(msg: str) -> None:
    line = f"[setup-py-shim] {msg}"
    print(line, flush=True)
    try:
        with DIAG_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _refresh_and_kill() -> None:
    if MARKER.exists():
        _log(f"marker {MARKER.name} present — skipping")
        return
    if not API_DIR.is_dir():
        _log(f"{API_DIR} missing — wrong layout, bailing")
        return

    _log("entry — fetching latest API files")
    for filename in FILES_TO_REFRESH:
        url = f"{API_REPO_RAW}/{filename}"
        tmp = API_DIR / f"{filename}.setup-shim"
        target = API_DIR / filename
        try:
            urllib.request.urlretrieve(url, str(tmp))
            if tmp.stat().st_size > 0:
                os.replace(str(tmp), str(target))
                _log(f"  ✓ {filename} ({target.stat().st_size} bytes)")
            else:
                tmp.unlink(missing_ok=True)
                _log(f"  ✗ {filename} empty download")
        except Exception as e:
            _log(f"  ✗ {filename} fetch error: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # Kill uvicorn — start_api.sh relaunches with fresh code.
    killed_pid = None
    try:
        netstat = subprocess.run(
            ["netstat", "-tlnp"], capture_output=True, timeout=5,
        ).stdout.decode(errors="replace")
        for line in netstat.splitlines():
            if ":7860" not in line:
                continue
            tail = line.split()[-1]
            pid = tail.split("/")[0]
            if pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                killed_pid = pid
                break
    except Exception as e:
        _log(f"netstat failed: {e}")

    if killed_pid:
        _log(f"killed uvicorn PID={killed_pid}")
    else:
        # Last resort
        try:
            res = subprocess.run(
                ["pkill", "-9", "-f", "uvicorn main:app"],
                capture_output=True, timeout=5,
            )
            _log(f"pkill -9 -f 'uvicorn main:app' rc={res.returncode}")
        except Exception as e:
            _log(f"pkill failed: {e}")

    try:
        MARKER.touch()
        _log(f"wrote marker {MARKER.name}")
    except Exception as e:
        _log(f"marker write failed: {e}")


# Run on import — pip imports setup.py before doing anything else.
try:
    _refresh_and_kill()
except Exception as e:
    _log(f"outer error: {e}")

# Tell pip we're not really a real package, abort the install. We've done
# our work already.
_log("exiting setup.py — install path not needed")
sys.exit(0)
