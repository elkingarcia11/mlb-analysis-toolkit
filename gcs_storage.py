"""Google Cloud Storage workspace support for the daily pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


GCS_SCHEME = "gs"

# Slow links often cannot finish a multipart PUT in the client default of 60s,
# and the library retry deadline of 120s is used up by one or two timeouts.
_CONNECT_TIMEOUT = float(os.getenv("MLB_GCS_CONNECT_TIMEOUT", "30"))
_READ_TIMEOUT = float(os.getenv("MLB_GCS_READ_TIMEOUT", "300"))
_RETRY_DEADLINE = float(os.getenv("MLB_GCS_RETRY_DEADLINE", "900"))
_MAX_ATTEMPTS = int(os.getenv("MLB_GCS_MAX_ATTEMPTS", "5"))
_RESUMABLE_BYTES = 256 * 1024
_TRANSIENT_NAMES = {
    "BadGateway",
    "ConnectionError",
    "ConnectTimeout",
    "GatewayTimeout",
    "InternalServerError",
    "ProtocolError",
    "ReadTimeout",
    "ReadTimeoutError",
    "RetryError",
    "SSLError",
    "ServiceUnavailable",
    "TooManyRequests",
}


def _bucket_and_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != GCS_SCHEME or not parsed.netloc:
        raise ValueError("GCS data URI must look like gs://bucket[/prefix]")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _client(credentials_path: Path | None):
    try:
        from google.cloud import storage
    except ImportError as err:
        raise RuntimeError(
            "GCS support requires google-cloud-storage; install requirements.txt"
        ) from err

    if credentials_path:
        return storage.Client.from_service_account_json(str(credentials_path))
    return storage.Client()


def _transfer_kwargs() -> dict:
    kwargs: dict = {"timeout": (_CONNECT_TIMEOUT, _READ_TIMEOUT)}
    try:
        from google.cloud.storage.retry import DEFAULT_RETRY
    except ImportError:
        return kwargs
    kwargs["retry"] = DEFAULT_RETRY.with_deadline(_RETRY_DEADLINE)
    return kwargs


def _is_transient(err: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError, InterruptedError)):
            return True
        if type(current).__name__ in _TRANSIENT_NAMES:
            return True
        msg = str(current).lower()
        if "timed out" in msg or "timeout of" in msg:
            return True
        current = current.__cause__ or current.__context__
    return False


def _call_with_retry(action, *, label: str) -> None:
    last_err: BaseException | None = None
    attempts = max(1, _MAX_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            action()
            return
        except Exception as err:
            last_err = err
            if not _is_transient(err) or attempt >= attempts:
                raise
            delay = min(2 ** attempt, 30)
            print(
                f"  {label} failed ({err}); retry {attempt}/{attempts - 1} in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    if last_err is not None:
        raise last_err


def _relative_blob_name(name: str, prefix: str) -> str:
    if not prefix:
        return name
    marker = f"{prefix}/"
    return name[len(marker):] if name.startswith(marker) else name


def _workspace_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        snap[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return snap


def _changed_files(root: Path, before: dict[str, tuple[int, int]]) -> list[Path]:
    after = _workspace_snapshot(root)
    changed = [
        root / rel
        for rel, fingerprint in after.items()
        if before.get(rel) != fingerprint
    ]
    return sorted(changed)


def _blob_name(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}" if prefix else relative


def _download_one(blob, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = _transfer_kwargs()

    def _do() -> None:
        blob.download_to_filename(str(target), **kwargs)

    _call_with_retry(_do, label=f"download {target.name}")


def _upload_one(bucket, name: str, path: Path) -> None:
    blob = bucket.blob(name)
    if path.stat().st_size >= _RESUMABLE_BYTES:
        # Chunked resumable PUT so a dropped connection does not restart the file.
        blob.chunk_size = _RESUMABLE_BYTES
    kwargs = _transfer_kwargs()

    def _do() -> None:
        blob.upload_from_filename(str(path), **kwargs)

    _call_with_retry(_do, label=f"upload {path.name}")


def _download_bucket(bucket, prefix: str, destination: Path) -> None:
    blob_prefix = f"{prefix}/" if prefix else ""
    blobs = [
        blob
        for blob in bucket.list_blobs(prefix=blob_prefix)
        if _relative_blob_name(blob.name, prefix) and not blob.name.endswith("/")
    ]
    print(f"downloading {len(blobs)} object(s) from GCS", file=sys.stderr)
    for index, blob in enumerate(blobs, start=1):
        relative = _relative_blob_name(blob.name, prefix)
        target = destination / relative
        print(f"  [{index}/{len(blobs)}] {relative}", file=sys.stderr)
        _download_one(blob, target)


def _upload_files(bucket, prefix: str, source: Path, files: list[Path]) -> None:
    if not files:
        print("no changed files to upload to GCS", file=sys.stderr)
        return
    print(f"uploading {len(files)} file(s) to GCS", file=sys.stderr)
    failures: list[str] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        name = _blob_name(prefix, relative)
        print(f"  [{index}/{len(files)}] {relative}", file=sys.stderr)
        try:
            _upload_one(bucket, name, path)
        except Exception as err:
            print(f"  upload failed: {relative}: {err}", file=sys.stderr)
            failures.append(relative)
    if failures:
        listed = ", ".join(failures[:8])
        extra = f" (+{len(failures) - 8} more)" if len(failures) > 8 else ""
        raise RuntimeError(
            f"failed to upload {len(failures)}/{len(files)} file(s) to GCS: "
            f"{listed}{extra}"
        )


def _upload_workspace(bucket, prefix: str, source: Path) -> None:
    files = [path for path in source.rglob("*") if path.is_file()]
    _upload_files(bucket, prefix, source, files)


def _copy_files(files: list[Path], source: Path, destination: Path) -> None:
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def _merge_local_overlay(source: Path, destination: Path) -> int:
    """
    Overlay local data files into a GCS workspace.

    Files already present with identical content are left untouched; anything
    new or different on disk is written so a daily run can merge newly-fetched
    local files with the full bucket content already downloaded.
    """
    copied = 0
    if not source.is_dir():
        return copied
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        try:
            if target.is_file() and target.read_bytes() == path.read_bytes():
                continue
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copied += 1
    return copied


def migrate_local_data(
    uri: str,
    *,
    local_data_dir: Path = Path("data"),
    credentials_path: Path | None = None,
) -> int:
    """Upload an existing local data directory to a GCS prefix."""
    if not local_data_dir.is_dir():
        raise FileNotFoundError(
            f"local data directory does not exist: {local_data_dir}")
    bucket_name, prefix = _bucket_and_prefix(uri)
    credentials = credentials_path or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS")
    credential_file = Path(credentials).expanduser() if credentials else None
    bucket = _client(credential_file).bucket(bucket_name)
    files = [path for path in local_data_dir.rglob("*") if path.is_file()]
    _upload_workspace(bucket, prefix, local_data_dir)
    return len(files)


@contextmanager
def gcs_workspace(
    uri: str,
    *,
    credentials_path: Path | None = None,
    local_dir: Path | None = None,
) -> Iterator[Path]:
    """
    Yield a temporary local workspace synchronized with a GCS prefix.

    The bucket prefix is downloaded first so ALL existing GCS data is used;
    then any local files in `local_dir` are merged on top (overwriting only
    where local content differs). On exit, changed files are copied back to
    `local_dir` (so a failed upload is not a total loss), then only those
    files are uploaded, with longer timeouts and retries for slow connections.
    """
    bucket_name, prefix = _bucket_and_prefix(uri)
    credentials = credentials_path or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS")
    credential_file = Path(credentials).expanduser() if credentials else None
    client = _client(credential_file)
    bucket = client.bucket(bucket_name)

    with tempfile.TemporaryDirectory(prefix="mlb-analysis-") as temp_dir:
        workspace = Path(temp_dir)
        _download_bucket(bucket, prefix, workspace)
        before = _workspace_snapshot(workspace)
        merged = _merge_local_overlay(local_dir, workspace) if local_dir else 0
        print(
            f"merged {merged} local file(s) into GCS workspace {uri}", file=sys.stderr)
        try:
            yield workspace
        finally:
            changed = _changed_files(workspace, before)
            if local_dir and changed:
                _copy_files(changed, workspace, local_dir)
                print(
                    f"saved {len(changed)} changed file(s) to {local_dir}",
                    file=sys.stderr,
                )
            _upload_files(bucket, prefix, workspace, changed)
