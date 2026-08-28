from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gcs_storage as gcs


class _FakeBlob:
    def __init__(self, name: str, content: bytes = b""):
        self.name = name
        self.content = content
        self.chunk_size = None
        self.downloads = 0
        self.uploads: list[str] = []
        self.upload_timeouts = None
        self.upload_retry = None
        self.fail_uploads = 0

    def download_to_filename(self, filename, **kwargs):
        self.downloads += 1
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.content)

    def upload_from_filename(self, filename, **kwargs):
        if self.fail_uploads > 0:
            self.fail_uploads -= 1
            raise TimeoutError("The read operation timed out")
        self.uploads.append(filename)
        self.upload_timeouts = kwargs.get("timeout")
        self.upload_retry = kwargs.get("retry")
        self.content = Path(filename).read_bytes()


class _FakeBucket:
    def __init__(self, blobs: list[_FakeBlob] | None = None):
        self._blobs = {blob.name: blob for blob in (blobs or [])}

    def list_blobs(self, prefix=""):
        return [blob for blob in self._blobs.values() if blob.name.startswith(prefix)]

    def blob(self, name: str) -> _FakeBlob:
        if name not in self._blobs:
            self._blobs[name] = _FakeBlob(name)
        return self._blobs[name]


class _FakeClient:
    def __init__(self, bucket: _FakeBucket):
        self._bucket = bucket

    def bucket(self, _name: str) -> _FakeBucket:
        return self._bucket


def test_workspace_snapshot_and_changed_files(tmp_path: Path):
    (tmp_path / "keep.csv").write_text("a\n", encoding="utf-8")
    nested = tmp_path / "raw"
    nested.mkdir()
    (nested / "teams.csv").write_text("old\n", encoding="utf-8")

    before = gcs._workspace_snapshot(tmp_path)
    (nested / "teams.csv").write_text("new\n", encoding="utf-8")
    (tmp_path / "models" / "team_win.pkl").parent.mkdir()
    (tmp_path / "models" / "team_win.pkl").write_bytes(b"model")

    changed = {path.relative_to(tmp_path).as_posix() for path in gcs._changed_files(tmp_path, before)}
    assert changed == {"raw/teams.csv", "models/team_win.pkl"}


def test_is_transient_walks_retry_error_cause():
    timeout = TimeoutError("The read operation timed out")
    wrapped = RuntimeError("Timeout of 120.0s exceeded")
    wrapped.__cause__ = timeout
    assert gcs._is_transient(wrapped) is True
    assert gcs._is_transient(ValueError("bad object name")) is False


def test_call_with_retry_retries_timeouts(monkeypatch):
    monkeypatch.setattr(gcs, "_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(gcs.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("The read operation timed out")

    gcs._call_with_retry(flaky, label="upload foo.csv")
    assert calls["n"] == 3


def test_call_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr(gcs, "_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(gcs.time, "sleep", lambda _seconds: None)

    def always_fail() -> None:
        raise TimeoutError("The read operation timed out")

    with pytest.raises(TimeoutError):
        gcs._call_with_retry(always_fail, label="upload foo.csv")


def test_gcs_workspace_uploads_only_changed_and_merged_files(tmp_path: Path, monkeypatch):
    remote = _FakeBlob("raw/teams.csv", b"gcs-teams\n")
    unchanged = _FakeBlob("raw/players.csv", b"players\n")
    bucket = _FakeBucket([remote, unchanged])
    monkeypatch.setattr(gcs, "_client", lambda _creds: _FakeClient(bucket))

    local = tmp_path / "local"
    local.mkdir()
    (local / "raw").mkdir()
    (local / "raw" / "odds.csv").write_text("odds\n", encoding="utf-8")

    with gcs.gcs_workspace("gs://mlb-analysis-toolkit", local_dir=local) as workspace:
        (workspace / "raw" / "teams.csv").write_text("updated-teams\n", encoding="utf-8")
        (workspace / "models").mkdir()
        (workspace / "models" / "team_win.pkl").write_bytes(b"model")

    uploaded = {
        name: blob.content
        for name, blob in bucket._blobs.items()
        if blob.uploads
    }
    assert uploaded == {
        "raw/teams.csv": b"updated-teams\n",
        "raw/odds.csv": b"odds\n",
        "models/team_win.pkl": b"model",
    }
    assert unchanged.uploads == []
    assert remote.upload_timeouts == (gcs._CONNECT_TIMEOUT, gcs._READ_TIMEOUT)
    assert (local / "raw" / "teams.csv").read_text(encoding="utf-8") == "updated-teams\n"
    assert (local / "models" / "team_win.pkl").read_bytes() == b"model"


def test_gcs_workspace_keeps_local_copy_if_upload_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gcs, "_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(gcs, "_transfer_kwargs", lambda: {})
    remote = _FakeBlob("raw/teams.csv", b"gcs-teams\n")
    bucket = _FakeBucket([remote])

    def always_fail_blob(name: str) -> _FakeBlob:
        blob = _FakeBucket.blob(bucket, name)
        blob.fail_uploads = 9
        return blob

    bucket.blob = always_fail_blob  # type: ignore[method-assign]
    monkeypatch.setattr(gcs, "_client", lambda _creds: _FakeClient(bucket))

    local = tmp_path / "local"
    local.mkdir()

    with pytest.raises(RuntimeError, match="failed to upload"):
        with gcs.gcs_workspace("gs://mlb-analysis-toolkit", local_dir=local) as workspace:
            (workspace / "models").mkdir()
            (workspace / "models" / "team_win.pkl").write_bytes(b"model")

    assert (local / "models" / "team_win.pkl").read_bytes() == b"model"


def test_upload_one_retries_then_succeeds(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gcs, "_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(gcs.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(gcs, "_transfer_kwargs", lambda: {"timeout": (30, 300)})

    path = tmp_path / "pred.csv"
    path.write_text("ok\n", encoding="utf-8")
    blob = _FakeBlob("pred.csv")
    blob.fail_uploads = 2
    bucket = MagicMock()
    bucket.blob.return_value = blob

    gcs._upload_one(bucket, "pred.csv", path)
    assert blob.uploads == [str(path)]
    assert blob.content == b"ok\n"


def test_upload_files_continues_then_reports_failures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gcs, "_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(gcs, "_transfer_kwargs", lambda: {})

    good = tmp_path / "ok.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("1\n", encoding="utf-8")
    bad.write_text("2\n", encoding="utf-8")

    ok_blob = _FakeBlob("ok.csv")
    bad_blob = _FakeBlob("bad.csv")
    bad_blob.fail_uploads = 9

    def blob_for(name: str) -> _FakeBlob:
        return ok_blob if name == "ok.csv" else bad_blob

    bucket = MagicMock()
    bucket.blob.side_effect = blob_for

    with pytest.raises(RuntimeError, match="failed to upload 1/2"):
        gcs._upload_files(bucket, "", tmp_path, [good, bad])
    assert ok_blob.uploads == [str(good)]
    assert bad_blob.uploads == []
