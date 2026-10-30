"""The ImageNet dataset revision is pinned, so re-preparing the eval subset
("first 2 images per class in streaming order") reads the same files in the
same order instead of whatever the repo's latest commit is."""

from __future__ import annotations

import re
import sys
import types

from data_prep import imagenet


def test_revision_is_a_full_commit_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", imagenet.IMAGENET_REVISION)


def test_every_split_is_loaded_at_the_pinned_revision(monkeypatch):
    calls = []
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: calls.append((args, kwargs)) or "ds"
    monkeypatch.setitem(sys.modules, "datasets", fake)

    for split in ("train", "validation"):
        assert imagenet._load_streaming_split(split) == "ds"

    assert [c[0] for c in calls] == [(imagenet.IMAGENET_REPO_ID,)] * 2
    assert all(c[1]["revision"] == imagenet.IMAGENET_REVISION and c[1]["streaming"] is True for c in calls)
    assert [c[1]["split"] for c in calls] == ["train", "validation"]
