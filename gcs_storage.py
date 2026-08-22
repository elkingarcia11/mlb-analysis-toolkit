"""Google Cloud Storage workspace support for the daily pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


GCS_SCHEME = "gs"


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


def _relative_blob_name(name: str, prefix: str) -> str:
    if not prefix:
        return name
    marker = f"{prefix}/"
    return name[len(marker):] if name.startswith(marker) else name


def _download_bucket(bucket, prefix: str, destination: Path) -> None:
    blob_prefix = f"{prefix}/" if prefix else ""
    for blob in bucket.list_blobs(prefix=blob_prefix):
        relative = _relative_blob_name(blob.name, prefix)
        if not relative or relative.endswith("/"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))


def _upload_workspace(bucket, prefix: str, source: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        name = f"{prefix}/{relative}" if prefix else relative
        bucket.blob(name).upload_from_filename(str(path))


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
    where local content differs). On exit the entire workspace — existing
    GCS data plus newly fetched/backfilled/trained/predicted files — is
    uploaded back to the bucket.
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
        merged = _merge_local_overlay(local_dir, workspace) if local_dir else 0
        print(
            f"merged {merged} local file(s) into GCS workspace {uri}", file=sys.stderr)
        try:
            yield workspace
        finally:
            _upload_workspace(bucket, prefix, workspace)
