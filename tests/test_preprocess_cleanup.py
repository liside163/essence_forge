from __future__ import annotations

import shutil
from pathlib import Path

from essence_forge.preprocess import _remove_directory_tree


def test_remove_directory_tree_falls_back_to_rename_when_rmtree_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "precomputed"
    root.mkdir()
    (root / "sample.npy").write_bytes(b"data")

    original_rmtree = shutil.rmtree
    calls: list[str] = []

    def flaky_rmtree(path, *args, **kwargs):
        target = Path(path)
        calls.append(target.name)
        if target.name == "precomputed":
            raise OSError(39, "Directory not empty")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

    _remove_directory_tree(root)

    assert root.exists() is False
    assert any(name.startswith("precomputed.stale.") for name in calls)
    assert list(tmp_path.glob("precomputed.stale.*")) == []
