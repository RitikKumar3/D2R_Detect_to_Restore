# d2r_dataloader.py

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image

# Optional PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    Dataset = object
    DataLoader = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("d2r_dataloader")

# Defaults
DEFAULT_FRAME_EXTS = ('.jpg', '.jpeg', '.png')
DEFAULT_MASK_EXTS = ('.png',)
_digit_re = re.compile(r'(\d+)')
FRAME_NUMBER_EXTRACT_RE = re.compile(r'(\d{1,7})')

# ---------------- utilities ----------------
def _natural_key(s: str) -> List[Any]:
    parts = _digit_re.split(s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_ordered_files(folder: str, exts: Tuple[str, ...] = DEFAULT_FRAME_EXTS) -> List[str]:
    if not folder:
        return []
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    files = [str(x) for x in p.iterdir() if x.suffix.lower() in exts]
    files_sorted = sorted(files, key=lambda s: _natural_key(Path(s).name))
    return files_sorted


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def extract_frame_number_from_filename(fname: str) -> Optional[int]:
    stem = Path(fname).stem
    m = FRAME_NUMBER_EXTRACT_RE.search(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except:
        return None

# ---------------- normalize_annotation ----------------
def normalize_annotation(ann: Optional[Dict[str, Any]], n_frames_in_manifest: Optional[int] = None) -> Dict[str, Any]:
    """
    Normalize annotation dict. Converts normalized floats (0..1) -> absolute indices when n_frames provided.
    Returns dict with 'normalized_affected_frames' list of integer indices (0-based).
    """
    if not ann:
        return {"normalized_affected_frames": []}
    out = dict(ann)
    frames_set = set()
    masks_map = {}

    def try_int(v):
        try:
            return int(v)
        except:
            return None

    # handle normalized_affected_frames
    if "normalized_affected_frames" in ann and isinstance(ann["normalized_affected_frames"], list):
        for v in ann["normalized_affected_frames"]:
            if isinstance(v, (int, np.integer)):
                frames_set.add(int(v))
            elif isinstance(v, float):
                if n_frames_in_manifest:
                    idx = int(round(v * (n_frames_in_manifest - 1)))
                    frames_set.add(idx)
            else:
                iv = try_int(v)
                if iv is not None:
                    frames_set.add(iv)

    # common explicit lists
    for key in ["deleted_indices", "deleted_frames", "inserted_frames",
                "inserted_indices", "duplicated_frames", "duplicated_indices",
                "flip_frames", "zoom_frames", "affected_frames"]:
        if key in ann and isinstance(ann[key], list):
            for vv in ann[key]:
                if isinstance(vv, (int, np.integer)):
                    frames_set.add(int(vv))
                elif isinstance(vv, float):
                    if n_frames_in_manifest:
                        frames_set.add(int(round(vv * (n_frames_in_manifest - 1))))
                else:
                    iv = try_int(vv)
                    if iv is not None:
                        frames_set.add(iv)

    # masks_map if present
    if "masks_map" in ann and isinstance(ann["masks_map"], dict):
        for k, v in ann["masks_map"].items():
            ik = try_int(k)
            if ik is not None:
                masks_map[ik] = v
                frames_set.add(ik)
            else:
                # try parsing numeric substring from k
                try:
                    ik2 = int(re.sub(r'\D', '', str(k))) if re.sub(r'\D', '', str(k)) else None
                    if ik2 is not None:
                        masks_map[ik2] = v
                        frames_set.add(ik2)
                except:
                    pass

    out["normalized_affected_frames"] = sorted(frames_set)
    if masks_map:
        out["masks_map"] = masks_map
    return out

# ---------------- manifest reader ----------------
def read_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    entries = []
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning(f"Skipping malformed manifest line: {e}")
                continue
            # normalize some keys to platform-specific paths
            for k in ("frames_dir", "mask_dir", "video_path", "annotation_path"):
                if k in obj and obj[k]:
                    try:
                        obj[k] = os.path.normpath(obj[k])
                    except Exception:
                        pass
            entries.append(obj)
    return entries

# ---------------- video seeking (read only selected frames) ----------------
def read_selected_frames_from_video(video_path: str, indices: List[int]) -> Dict[int, np.ndarray]:
    """
    Efficiently read only frame indices from a video. Returns dict index -> RGB ndarray.
    Indices are zero-based.
    """
    if not indices:
        return {}
    import cv2
    indices_set = set(sorted(indices))
    max_idx = max(indices_set)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    collected = {}
    idx = 0
    ret, frame = cap.read()
    while ret and idx <= max_idx:
        if idx in indices_set:
            rgb = frame[:, :, ::-1].copy()  # BGR->RGB
            collected[idx] = rgb
            if len(collected) == len(indices_set):
                break
        idx += 1
        ret, frame = cap.read()
    cap.release()
    return collected

# ---------------- D2RFrameDataset ----------------
class D2RFrameDataset(Dataset):
    def __init__(self,
                 entries: List[Dict[str, Any]],
                 load_frames: bool = False,
                 clip_len: Optional[int] = None,
                 stride: int = 1,
                 frame_exts: Tuple[str, ...] = DEFAULT_FRAME_EXTS,
                 mask_exts: Tuple[str, ...] = DEFAULT_MASK_EXTS,
                 frame_transform=None,
                 mask_transform=None,
                 video_fallback: bool = False,
                 prefer_frames: bool = True):
        super().__init__()
        self.entries = entries
        self.load_frames = load_frames
        self.clip_len = clip_len
        self.stride = stride
        self.frame_exts = frame_exts
        self.mask_exts = mask_exts
        self.frame_transform = frame_transform
        self.mask_transform = mask_transform
        self.video_fallback = video_fallback
        self.prefer_frames = prefer_frames

    def __len__(self):
        return len(self.entries)

    def _list_frames_for_entry(self, entry: Dict[str, Any]) -> List[str]:
        fd = entry.get("frames_dir")
        if fd and os.path.isdir(fd):
            return list_ordered_files(fd, exts=self.frame_exts)
        return []

    def _list_masks_for_entry(self, entry: Dict[str, Any]) -> List[str]:
        md = entry.get("mask_dir")
        return list_ordered_files(md, exts=self.mask_exts)

    def _build_index_to_framefile_map(self, frames_paths: List[str]) -> Dict[int, str]:
        """
        Build mapping absolute_index (0-based) -> filepath.
        Uses numeric stems when possible (handles 1-based or 0-based numeric filenames).
        Falls back to positional mapping.
        """
        mapping = {}
        numeric_found = {}
        for pos, p in enumerate(frames_paths):
            num = extract_frame_number_from_filename(p)
            if num is not None:
                numeric_found[pos] = num
        if numeric_found:
            nums = list(numeric_found.values())
            min_num = min(nums)
            offset = 1 if min_num >= 1 else 0
            for pos, p in enumerate(frames_paths):
                num = extract_frame_number_from_filename(p)
                if num is not None:
                    idx = num - offset
                    mapping[idx] = p
                else:
                    if pos not in mapping:
                        mapping[pos] = p
        else:
            for pos, p in enumerate(frames_paths):
                mapping[pos] = p
        return mapping

    def _sample_indices(self, total_len: int, ann: Dict[str, Any]) -> List[int]:
        if total_len <= 0:
            return []
        if self.clip_len and self.clip_len < total_len:
            af = ann.get("normalized_affected_frames", []) if ann else []
            if af:
                # ensure the target is within bounds
                mids = [m for m in af if 0 <= m < total_len]
                if mids:
                    target = mids[len(mids)//2]
                else:
                    target = 0
                start = max(0, min(total_len - self.clip_len, int(target - self.clip_len//2)))
            else:
                import random
                start = random.randint(0, total_len - self.clip_len)
            indices = list(range(start, start + self.clip_len, self.stride))
            return indices
        else:
            return list(range(0, total_len, self.stride))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        vid = entry.get("video_id")
        frames_paths = self._list_frames_for_entry(entry) if self.prefer_frames else []
        masks_paths = self._list_masks_for_entry(entry)

        n_frames_manifest = int(entry.get("n_frames") or 0)
        total_frames = n_frames_manifest

        mapping = {}
        if frames_paths:
            mapping = self._build_index_to_framefile_map(frames_paths)
            if mapping:
                total_frames = max(mapping.keys()) + 1
            else:
                total_frames = len(frames_paths)

        # fallback: probe video for frame count if needed and allowed
        if total_frames == 0 and self.video_fallback and entry.get("video_path"):
            try:
                import cv2
                cap = cv2.VideoCapture(entry["video_path"])
                if cap.isOpened():
                    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if c > 0:
                        total_frames = c
                    cap.release()
            except Exception:
                pass

        # final fallback: if frames_dir exists but mapping failed, use file count
        if total_frames == 0 and frames_paths:
            total_frames = len(frames_paths)

        if total_frames == 0:
            raise FileNotFoundError(f"No frames available for sample {vid}")

        # load annotation and normalize (convert normalized floats -> indices)
        ann = None
        ann_path = entry.get("annotation_path")
        if ann_path and os.path.exists(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load annotation for {vid}: {e}")
                ann = None
        ann_norm = normalize_annotation(ann, n_frames_in_manifest=total_frames)

        selected_indices = self._sample_indices(total_frames, ann_norm)

        # update meta n_frames to be consistent with frames folder (useful if downsampled)
        meta = {"n_frames": total_frames, "fps": entry.get("fps"), "resolution": entry.get("resolution")}

        sample = {
            "video_id": vid,
            "category": entry.get("category"),
            "video_path": entry.get("video_path"),
            "frames_dir": entry.get("frames_dir"),
            "frames_paths": frames_paths,
            "mask_dir": entry.get("mask_dir"),
            "mask_paths": masks_paths,
            "annotation": ann_norm,
            "meta": meta,
            "selected_indices": selected_indices
        }

        # load frames
        if self.load_frames:
            frames_t = []
            if frames_paths and mapping:
                # use mapping; if an index missing, pick nearest available index
                available_idxs = sorted(mapping.keys())
                min_idx, max_idx = available_idxs[0], available_idxs[-1]
                for fi in selected_indices:
                    # clamp to available range
                    if fi in mapping:
                        fp = mapping[fi]
                    else:
                        # find nearest available index
                        if fi < min_idx:
                            fp = mapping[min_idx]
                        elif fi > max_idx:
                            fp = mapping[max_idx]
                        else:
                            # choose nearest by searching down then up
                            lo = fi
                            while lo >= min_idx and lo not in mapping:
                                lo -= 1
                            hi = fi
                            while hi <= max_idx and hi not in mapping:
                                hi += 1
                            if lo >= min_idx and lo in mapping:
                                fp = mapping[lo]
                            elif hi <= max_idx and hi in mapping:
                                fp = mapping[hi]
                            else:
                                fp = mapping[max_idx]
                    arr = load_image_as_array(fp)
                    if self.frame_transform:
                        arr = self.frame_transform(arr)
                    frames_t.append(arr)
            elif self.video_fallback and entry.get("video_path"):
                collected = read_selected_frames_from_video(entry["video_path"], selected_indices)
                for fi in selected_indices:
                    arr = collected.get(fi)
                    if arr is None:
                        # fallback: use nearest available
                        if collected:
                            nearest = max(collected.keys())
                            arr = collected[nearest]
                        else:
                            arr = np.zeros((224, 224, 3), dtype=np.uint8)
                    if self.frame_transform:
                        arr = self.frame_transform(arr)
                    frames_t.append(arr)
            else:
                raise FileNotFoundError(f"No frames available for sample {vid} and video_fallback disabled")

            # frames stacked as (T, H, W, C)
            sample["frames"] = np.stack(frames_t, axis=0)
        else:
            sample["frames"] = None

        # --- load masks aligned to selected_indices and resize to frame size before stacking
        if self.load_frames and masks_paths:
            loaded_masks = []
            # determine target size (H,W)
            if sample.get("frames") is not None:
                _, fh, fw, _ = sample["frames"].shape
                target_h, target_w = int(fh), int(fw)
            else:
                res = entry.get("resolution") or [0, 0]
                if res[0] > 0 and res[1] > 0:
                    target_w, target_h = int(res[0]), int(res[1])
                else:
                    target_h, target_w = 1, 1

            # build mapping for mask files
            mask_map = self._build_index_to_framefile_map(masks_paths) if masks_paths else {}

            # check annotation masks_map preference
            ann_masks_map = ann_norm.get("masks_map", {}) if ann_norm else {}

            for fi in selected_indices:
                explicit_mask_path = None
                if ann_masks_map and fi in ann_masks_map:
                    mf = ann_masks_map[fi]
                    if os.path.isabs(mf) and os.path.exists(mf):
                        explicit_mask_path = mf
                    else:
                        md = entry.get("mask_dir")
                        if md:
                            cand = os.path.join(md, mf)
                            if os.path.exists(cand):
                                explicit_mask_path = cand
                        if not explicit_mask_path:
                            for mp in masks_paths:
                                if os.path.basename(mp) == mf:
                                    explicit_mask_path = mp
                                    break

                if explicit_mask_path:
                    m = load_mask_as_array(explicit_mask_path)
                elif fi in mask_map:
                    m = load_mask_as_array(mask_map[fi])
                elif len(mask_map) == 1:
                    m = load_mask_as_array(next(iter(mask_map.values())))
                else:
                    m = np.zeros((target_h, target_w), dtype=np.uint8)

                # robustify mask shape
                if m is None:
                    m = np.zeros((target_h, target_w), dtype=np.uint8)
                else:
                    m = np.asarray(m)
                    if m.ndim == 3:
                        m = m[..., 0]
                    if m.ndim != 2:
                        m = m.reshape(m.shape[0], -1) if m.size else np.zeros((target_h, target_w), dtype=np.uint8)

                # resize if needed using nearest neighbor to preserve labels
                if (m.shape[0], m.shape[1]) != (target_h, target_w):
                    try:
                        import cv2
                        m = cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        new = np.zeros((target_h, target_w), dtype=np.uint8)
                        mh, mw = m.shape
                        new[:min(mh, target_h), :min(mw, target_w)] = m[:min(mh, target_h), :min(mw, target_w)]
                        m = new

                m = (m > 0).astype(np.uint8)

                if self.mask_transform:
                    m = self.mask_transform(m)

                loaded_masks.append(m)

            sample["masks"] = np.stack(loaded_masks, axis=0)  # (T, H, W)
        else:
            sample["masks"] = None

        return sample

# ---------------- convenience factory ----------------
def dataset_from_manifest(manifest_path: str = "manifest.jsonl", **kwargs) -> D2RFrameDataset:
    """
    Create D2RFrameDataset from a manifest file.
    Default manifest filename is 'manifest.jsonl' (in current working directory) - pass absolute path to use Drive manifest.
    """
    entries = read_manifest(manifest_path)
    return D2RFrameDataset(entries, **kwargs)

# ---------------- collate ----------------
def collate_video_batch(batch: List[Dict[str, Any]]):
    """
    Collate for DataLoader. Pads spatial and temporal dims.
    Expects sample["frames"] shape (T,H,W,C) when loaded.
    Returns dictionary: video_ids, frames (B,T,C,H,W), masks (B,T,1,H,W) or None, meta, raw
    """
    if not batch:
        return {}

    # if frames are not loaded return raw list
    if batch[0].get("frames") is None:
        return batch

    import torch
    import torch.nn.functional as F

    frames_tensors = []
    max_h = 0; max_w = 0; max_T = 0

    for sample in batch:
        fr = sample["frames"]
        if fr.ndim != 4:
            raise ValueError("Expected frames ndarray shape (T,H,W,C)")
        t = torch.from_numpy(fr).permute(0, 3, 1, 2)  # (T,C,H,W)
        frames_tensors.append(t)
        Tcur, Ccur, Hcur, Wcur = t.shape
        if Hcur > max_h: max_h = Hcur
        if Wcur > max_w: max_w = Wcur
        if Tcur > max_T: max_T = Tcur

    # Spatial targets
    target_h, target_w = max_h, max_w

    frames_padded = []
    for t in frames_tensors:
        Tcur, Ccur, Hcur, Wcur = t.shape
        if (Hcur, Wcur) != (target_h, target_w):
            t = F.interpolate(t.float(), size=(target_h, target_w), mode='bilinear', align_corners=False).to(t.dtype)
        if t.shape[0] < max_T:
            pad = torch.zeros((max_T - t.shape[0], t.shape[1], t.shape[2], t.shape[3]), dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        elif t.shape[0] > max_T:
            t = t[:max_T]
        frames_padded.append(t)

    frames_stacked = torch.stack(frames_padded, dim=0).float() / 255.0  # (B, T, C, H, W)

    # Masks
    masks_tensor = None
    if batch[0].get("masks") is not None:
        masks_list = []
        for sample in batch:
            m = sample["masks"]
            if m is None:
                masks_list.append(torch.zeros((max_T, 1, target_h, target_w), dtype=torch.float32))
                continue
            # m shape (T,H,W)
            if m.ndim == 3:
                mt = torch.from_numpy(m).unsqueeze(1)  # (T,1,H,W)
            elif m.ndim == 4:
                mt = torch.from_numpy(m)
            else:
                # unexpected
                mt = torch.from_numpy(m).unsqueeze(1)

            Tm, Cm, Hm, Wm = mt.shape
            if (Hm, Wm) != (target_h, target_w):
                mt = F.interpolate(mt.float(), size=(target_h, target_w), mode='nearest').to(mt.dtype)
            if Tm < max_T:
                padm = torch.zeros((max_T - Tm, mt.shape[1], mt.shape[2], mt.shape[3]), dtype=mt.dtype)
                mt = torch.cat([mt, padm], dim=0)
            elif Tm > max_T:
                mt = mt[:max_T]
            masks_list.append(mt.float())

        masks_tensor = torch.stack(masks_list, dim=0)  # (B, T, 1, H, W)

    metas = [sample["meta"] for sample in batch]
    vids = [sample["video_id"] for sample in batch]
    return {"video_ids": vids, "frames": frames_stacked, "masks": masks_tensor, "meta": metas, "raw": batch}

# ---------------- CLI test ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--load-frames", action="store_true")
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--video-fallback", action="store_true")
    parser.add_argument("--prefer-frames", action="store_true")
    args = parser.parse_args()

    ds = dataset_from_manifest(args.manifest, load_frames=args.load_frames, clip_len=args.clip_len, video_fallback=args.video_fallback, prefer_frames=args.prefer_frames)
    print("Dataset size:", len(ds))
    s = ds[args.sample_index]
    print("Sample:", s["video_id"], "n_frames:", s["meta"]["n_frames"], "selected:", s["selected_indices"])
