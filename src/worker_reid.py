"""Embedding-based worker identity: map new tracker IDs to prior logical workers."""

from __future__ import annotations

import cv2
import numpy as np
import torch

import config

from .similarity import DinoV2

# (video_stem, logical_id) already written as JPEG
_saved_logical_snapshots: set[tuple[str, int]] = set()


def _cosine(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """emb_* shape [1, d] or [d]."""
    a = emb_a.view(1, -1)
    b = emb_b.view(1, -1)
    return float(torch.cosine_similarity(a, b, dim=1)[0].item())


def _best_disappeared_match(
    emb_cpu: torch.Tensor,
    disappeared: dict[int, torch.Tensor],
    threshold: float,
) -> tuple[int | None, float]:
    """
    Compare against every disappeared embedding; take argmax cosine similarity.
    Re-identify only if that global best score is >= threshold.
    """
    if not disappeared:
        return None, -1.0
    best_lid: int | None = None
    best_sim = -1.0
    for lid, emb_g in disappeared.items():
        sim = _cosine(emb_cpu, emb_g)
        if sim > best_sim:
            best_sim = sim
            best_lid = lid
    if best_lid is None or best_sim < threshold:
        return None, best_sim
    return best_lid, best_sim


class WorkerReIDManager:
    """
    Tracker track IDs can change after occlusion. We keep a stable logical worker
    id: when a new track id appears, embed the bbox crop and match against
    embeddings of workers who are "disappeared" (were visible before, not now).
    Cosine similarity >= threshold reuses that logical id.
    """

    def __init__(self) -> None:
        self.threshold = float(config.WORKER_SIMILARITY_THRESHOLD)
        self._embedder: DinoV2 | None = None

        self._byte_to_logical: dict[int, int] = {}
        self._last_byte_ids: set[int] = set()
        self._logical_to_embedding: dict[int, torch.Tensor] = {}
        self._disappeared: dict[int, torch.Tensor] = {}
        self._next_logical_id = 1

        self._pending_snapshots: list[tuple[int, np.ndarray]] = []

    def _embed(self) -> DinoV2:
        if self._embedder is None:
            print("Loading DinoV2 for worker re-ID embeddings (first run may download weights)...")
            self._embedder = DinoV2()
        return self._embedder

    def process_frame(self, frame: np.ndarray, tracks) -> None:
        """Update mappings; call after tracker.update_tracks. Requires frame (BGR)."""
        h, w = frame.shape[:2]
        confirmed = [t for t in tracks if t.is_confirmed()]
        current_byte_ids = {int(t.track_id) for t in confirmed}

        for bid in self._last_byte_ids - current_byte_ids:
            if bid in self._byte_to_logical:
                lid = self._byte_to_logical[bid]
                if lid in self._logical_to_embedding:
                    self._disappeared[lid] = self._logical_to_embedding[lid].detach().cpu()
                del self._byte_to_logical[bid]

        for t in confirmed:
            bid = int(t.track_id)
            if bid in self._byte_to_logical:
                continue

            x1, y1, x2, y2 = map(int, t.to_ltrb())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                lid = self._next_logical_id
                self._next_logical_id += 1
                self._byte_to_logical[bid] = lid
                continue

            crop = frame[y1:y2, x1:x2].copy()
            emb = self._embed().embed_bgr(crop)
            emb_cpu = emb.detach().cpu()

            best_lid, best_sim = _best_disappeared_match(
                emb_cpu, self._disappeared, self.threshold
            )

            if best_lid is not None:
                logical_id = best_lid
                del self._disappeared[best_lid]
                self._logical_to_embedding[logical_id] = emb_cpu
            else:
                logical_id = self._next_logical_id
                self._next_logical_id += 1
                self._logical_to_embedding[logical_id] = emb_cpu
                self._pending_snapshots.append((logical_id, crop))

            self._byte_to_logical[bid] = logical_id

        self._last_byte_ids = current_byte_ids

    def logical_id_for_byte(self, byte_track_id: int) -> int:
        return self._byte_to_logical[int(byte_track_id)]

    def write_snapshot_images(self, video_stem: str) -> None:
        """Write pending new-logical crops to output/worker_ids/<stem>/worker_<id>.jpg."""
        if not video_stem:
            return
        if not self._pending_snapshots:
            return
        out_dir = config.path_worker_id_images_dir(video_stem)
        for logical_id, crop in self._pending_snapshots:
            key = (video_stem, logical_id)
            if key in _saved_logical_snapshots:
                continue
            path = out_dir / f"worker_{logical_id:04d}.jpg"
            cv2.imwrite(
                str(path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            _saved_logical_snapshots.add(key)
        self._pending_snapshots.clear()
