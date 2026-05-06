"""Save torch/pickle/json objects through /tmp staging + cp/mv for resilience on
network filesystems (e.g. BeeGFS, NFS) that intermittently return EREMOTEIO/EIO
on large open/write/close cycles. Writing to /tmp first (local disk) and then
using cp + mv avoids that: cp handles partial failures, mv is an atomic rename
at filesystem level.

Also provides `torch_load_with_retry` which copies the source file to /tmp
first and then calls `torch.load` on the local copy — bypasses read flakiness
during `torch.load`'s zip-record reads.

Usage:
    from beegfs_safe_save import save_with_retry, torch_load_with_retry
    save_with_retry(obj, "/path/to/my_cache.pt", saver="torch")
    obj = torch_load_with_retry("/path/to/my_cache.pt")

If your filesystem is local SSD/NVMe and you don't need this, replace calls
with plain `torch.save(obj, path)` / `torch.load(path)`.
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch


def _torch_save(obj: Any, path: str) -> None:
    torch.save(obj, path)


def _json_save(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


SAVERS = {"torch": _torch_save, "json": _json_save}


def save_with_retry(
    obj: Any,
    target_path,
    saver: str = "torch",
    max_attempts: int = 5,
    sleep_s: float = 5.0,
) -> str:
    """Serialize `obj` to `target_path` via /tmp staging with retry.

    Returns the final target path (as string). Raises RuntimeError if all
    attempts fail; the staged /tmp file is NOT deleted in that case so the
    user can recover manually.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if saver not in SAVERS:
        raise ValueError(f"Unknown saver {saver!r}; choose from {list(SAVERS)}")

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"beegfs_stage_{target.name}_",
        suffix=target.suffix,
        dir="/tmp",
    )
    os.close(tmp_fd)
    SAVERS[saver](obj, tmp_path)
    print(f"[beegfs_safe_save] staged {tmp_path}  ({os.path.getsize(tmp_path)} bytes)", flush=True)

    last_err: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            tmp_target = str(target) + ".tmp_replace"
            subprocess.run(["cp", tmp_path, tmp_target], check=True)
            subprocess.run(["mv", tmp_target, str(target)], check=True)
            print(f"[beegfs_safe_save] saved {target}", flush=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return str(target)
        except Exception as e:  # noqa: BLE001 - want to retry on any I/O failure
            last_err = e
            print(
                f"[beegfs_safe_save] attempt {attempt+1}/{max_attempts} failed: {e}",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to write {target} after {max_attempts} attempts. "
        f"Staged file retained at {tmp_path}. Last error: {last_err}"
    )


def torch_load_with_retry(
    src_path,
    max_attempts: int = 5,
    sleep_s: float = 5.0,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Any:
    """torch.load from beegfs via /tmp staging.

    beegfs `torch.load` often hits EREMOTEIO inside the zip-record read.
    cp to /tmp (local disk) first then torch.load there. Deletes tmp copy
    on success; retains on repeated failure for manual recovery.
    """
    src = Path(src_path)
    last_err: Optional[BaseException] = None
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"beegfs_load_{src.name}_", suffix=src.suffix or ".pt", dir="/tmp"
    )
    os.close(tmp_fd)

    for attempt in range(max_attempts):
        try:
            subprocess.run(["cp", str(src), tmp_path], check=True)
            obj = torch.load(tmp_path, map_location=map_location, weights_only=weights_only)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return obj
        except Exception as e:
            last_err = e
            print(
                f"[beegfs_safe_save] torch_load_with_retry {src.name} attempt "
                f"{attempt+1}/{max_attempts} failed: {e}",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to torch.load {src} after {max_attempts} attempts. "
        f"Last error: {last_err}. Partial copy left at {tmp_path}."
    )
