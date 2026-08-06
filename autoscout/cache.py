"""On-disk cache for the two expensive decode passes.

Reading the scoreboard and tracking the field each require a full pass over the
video.  The results of both depend only on the video and on the detection
parameters, so caching them makes re-running the (cheap, frequently retuned)
shot logic near-instant.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Optional

CACHE_VERSION = 5


def _video_id(video_path: str) -> str:
    """Stable short id for one video file."""
    st = os.stat(video_path)
    raw = f"{os.path.abspath(video_path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _key(video_path: str, tag: str, salt: str = "") -> str:
    raw = f"{CACHE_VERSION}|{_video_id(video_path)}|{tag}|{salt}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class Cache:
    def __init__(self, directory: Optional[str]):
        self.directory = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def path(self, video_path: str, tag: str, salt: str = "") -> Optional[str]:
        if not self.directory:
            return None
        # The video id sits in the filename so superseded entries can be pruned
        # without touching another video's cache.
        return os.path.join(
            self.directory,
            f"{tag}_{_video_id(video_path)}_{_key(video_path, tag, salt)}.pkl",
        )

    def load(self, video_path: str, tag: str, salt: str = "") -> Optional[Any]:
        p = self.path(video_path, tag, salt)
        if not p or not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def store(self, video_path: str, tag: str, value: Any, salt: str = "") -> None:
        p = self.path(video_path, tag, salt)
        if not p:
            return
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        self._prune(tag, video_path, keep=p)

    def _prune(self, tag: str, video_path: str, keep: str) -> None:
        """Drop superseded entries for this tag and this video.

        Every threshold change mints a new cache key, and these files run to
        tens of megabytes; without this the directory grows without bound as
        settings are tuned.  Other videos' entries are left alone.
        """
        if not self.directory:
            return
        prefix = f"{tag}_{_video_id(video_path)}_"
        for name in os.listdir(self.directory):
            if not name.startswith(prefix) or not name.endswith(".pkl"):
                continue
            path = os.path.join(self.directory, name)
            if os.path.abspath(path) == os.path.abspath(keep):
                continue
            try:
                os.remove(path)
            except OSError:
                pass
