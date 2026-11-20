"""Audit image/mask datasets in a directory or ZIP without extracting them."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
from itertools import combinations
import json
import sys
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "validation", "valid": "validation", "validation": "validation", "test": "test"}


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _hash(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def audit(source: Path, duplicate_hashes: bool = False) -> dict:
    is_zip = source.is_file() and source.suffix.lower() == ".zip"
    archive = zipfile.ZipFile(source) if is_zip else None
    if is_zip:
        file_paths = [_norm(i.filename) for i in archive.infolist() if not i.is_dir()]
        opener = lambda p: archive.open(p)
    elif source.is_dir():
        file_paths = [_norm(str(p.relative_to(source))) for p in source.rglob("*") if p.is_file()]
        opener = lambda p: (source / Path(p)).open("rb")
    else:
        raise FileNotFoundError(f"Not a directory or ZIP file: {source}")

    by_parent_stem: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    split_paths: dict[str, str] = {}
    metadata_files = []
    for name in file_paths:
        p = Path(name)
        for part in p.parts[:-1]:
            if part.lower() in SPLIT_NAMES:
                split_paths[name] = SPLIT_NAMES[part.lower()]
                break
        if p.suffix.lower() in IMAGE_EXTS | MASK_EXTS:
            by_parent_stem[(str(p.parent).lower(), p.stem.lower())].append(name)
        if p.suffix.lower() in {".csv", ".json", ".txt", ".yaml", ".yml", ".xml"}:
            metadata_files.append(name)

    # Pair each image with one same-directory file of a mask format. Ambiguous
    # candidates are reported rather than silently selected.
    image_paths = [p for p in file_paths if Path(p).suffix.lower() in IMAGE_EXTS]
    mask_paths = [p for p in file_paths if Path(p).suffix.lower() in MASK_EXTS]
    paired, missing_masks, ambiguous = [], [], []
    used_masks = set()
    for image in image_paths:
        p = Path(image)
        candidates = [m for m in by_parent_stem[(str(p.parent).lower(), p.stem.lower())] if Path(m).suffix.lower() in MASK_EXTS]
        if len(candidates) == 1:
            paired.append((image, candidates[0]))
            used_masks.add(candidates[0])
        elif len(candidates) > 1:
            ambiguous.append({"image": image, "masks": sorted(candidates)})
        else:
            missing_masks.append(image)

    image_info, mask_info, corrupt = [], [], []
    mask_value_files = collections.Counter()
    mask_pixel_counts = collections.Counter()
    dimension_mismatches = []
    media_paths = image_paths + mask_paths
    total_media = len(media_paths)
    print(f"Decoding {total_media} image and mask files...", file=sys.stderr, flush=True)
    for role, paths, info_list in (("image", image_paths, image_info), ("mask", mask_paths, mask_info)):
        for index, name in enumerate(paths, 1):
            try:
                with opener(name) as f:
                    with Image.open(f) as im:
                        im.load()
                        rec = {"path": name, "format": im.format, "mode": im.mode, "width": im.width, "height": im.height}
                        if role == "mask":
                            pixels = np.asarray(im)
                            if pixels.ndim == 3:
                                unique, counts = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0, return_counts=True)
                                values = [tuple(int(c) for c in row) for row in unique]
                                pixel_counts = {str(v): int(c) for v, c in zip(values, counts)}
                            else:
                                unique, counts = np.unique(pixels, return_counts=True)
                                values = [int(v) for v in unique]
                                pixel_counts = {str(v): int(c) for v, c in zip(values, counts)}
                            rec["unique_values"] = values[:256]
                            rec["unique_value_count"] = len(values)
                            rec["pixel_counts"] = pixel_counts
                            mask_value_files.update(map(str, values))
                            mask_pixel_counts.update(pixel_counts)
                        info_list.append(rec)
            except Exception as exc:  # damaged member or unsupported image
                corrupt.append({"path": name, "role": role, "error": f"{type(exc).__name__}: {exc}"})
            if index % 100 == 0 or index == len(paths):
                completed = len(image_info) + len(mask_info) + len(corrupt)
                print(f"Decoded {completed}/{total_media} media files", file=sys.stderr, flush=True)
    ordered_mask_values = sorted(int(v) for v in mask_pixel_counts if str(v).isdigit())
    mask_value_clusters = []
    for value in ordered_mask_values:
        if not mask_value_clusters or value > mask_value_clusters[-1][-1] + 1:
            mask_value_clusters.append([value])
        else:
            mask_value_clusters[-1].append(value)
    im_dims = {x["path"]: (x["width"], x["height"]) for x in image_info}
    mk_dims = {x["path"]: (x["width"], x["height"]) for x in mask_info}
    for image, mask in paired:
        if image in im_dims and mask in mk_dims and im_dims[image] != mk_dims[mask]:
            dimension_mismatches.append({"image": image, "image_dimensions": im_dims[image], "mask": mask, "mask_dimensions": mk_dims[mask]})

    splits = collections.Counter(split_paths.get(p, "unspecified") for p in image_paths)
    unmatched_masks = sorted(set(mask_paths) - used_masks)
    duplicate_content = {}
    duplicate_candidate_file_count = 0
    if duplicate_hashes:
        if archive:
            size_by_path = {_norm(i.filename): i.file_size for i in archive.infolist() if not i.is_dir()}
        else:
            size_by_path = {p: (source / Path(p)).stat().st_size for p in image_paths + mask_paths}
        size_groups = collections.defaultdict(list)
        for name in image_paths + mask_paths:
            size_groups[size_by_path[name]].append(name)
        candidates = [name for group in size_groups.values() if len(group) > 1 for name in group]
        duplicate_candidate_file_count = len(candidates)
        hashes = collections.defaultdict(list)
        for name in candidates:
            with opener(name) as f:
                hashes[_hash(f)].append(name)
        duplicate_content = {digest: names for digest, names in hashes.items() if len(names) > 1}

    def distribution(records: list[dict]) -> dict:
        counts = collections.Counter(f"{r['width']}x{r['height']}" for r in records)
        return dict(sorted(counts.items()))

    adjacent_csvs = []
    adjacent_severity_labels = {}
    metadata_base = source.parent if is_zip else source
    for csv_path in sorted(metadata_base.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            columns = reader.fieldnames or []
            name_column = next((c for c in columns if c.lower() in {"image_name", "image", "filename", "file_name"}), None)
            label_column = next((c for c in columns if "severity" in c.lower()), None)
            source_image_stems = {Path(p).stem.lower() for p in image_paths}
            if name_column and label_column:
                adjacent_severity_labels.update({Path(row.get(name_column, "")).stem.lower(): row.get(label_column, "") for row in rows})
            adjacent_csvs.append({
                "path": str(csv_path), "row_count": len(rows), "columns": columns,
                "label_value_counts": dict(collections.Counter(row.get(label_column, "") for row in rows)) if label_column else {},
                "image_name_column": name_column,
                "matching_image_stems": sum(Path(row.get(name_column, "")).stem.lower() in source_image_stems for row in rows) if name_column else None,
            })
        except Exception as exc:
            adjacent_csvs.append({"path": str(csv_path), "error": f"{type(exc).__name__}: {exc}"})

    duplicate_image_pairs = []
    pair_to_mask = {image: mask for image, mask in paired}
    hash_by_path = {name: digest for digest, names in duplicate_content.items() for name in names}
    for names in duplicate_content.values():
        images = [name for name in names if Path(name).suffix.lower() in IMAGE_EXTS]
        for first, second in combinations(images, 2):
            stem_a, stem_b = Path(first).stem.lower(), Path(second).stem.lower()
            severity_a, severity_b = adjacent_severity_labels.get(stem_a), adjacent_severity_labels.get(stem_b)
            mask_a, mask_b = pair_to_mask.get(first), pair_to_mask.get(second)
            duplicate_image_pairs.append({
                "images": [first, second],
                "splits": [split_paths.get(first, "unspecified"), split_paths.get(second, "unspecified")],
                "crosses_split": split_paths.get(first) != split_paths.get(second),
                "masks": [mask_a, mask_b],
                "masks_identical": bool(mask_a and mask_b and mask_a in hash_by_path and mask_b in hash_by_path and hash_by_path[mask_a] == hash_by_path[mask_b]),
                "severity_labels": [severity_a, severity_b],
                "severity_conflict": bool(severity_a and severity_b and severity_a != severity_b),
            })

    report = {
        "source": str(source), "source_type": "zip" if is_zip else "directory",
        "archive_entries": len(file_paths), "image_count": len(image_paths), "mask_count": len(mask_paths),
        "paired_count": len(paired), "split_image_counts": dict(sorted(splits.items())),
        "image_formats": dict(collections.Counter(Path(p).suffix.lower() for p in image_paths)),
        "mask_formats": dict(collections.Counter(Path(p).suffix.lower() for p in mask_paths)),
        "image_dimensions": distribution(image_info), "mask_dimensions": distribution(mask_info),
        "mask_modes": dict(collections.Counter(x["mode"] for x in mask_info)),
        "mask_unique_pixel_values": sorted(mask_value_files, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)),
        "mask_value_clusters": [[cluster[0], cluster[-1]] for cluster in mask_value_clusters],
        "mask_binary_assessment": {
            "candidate_midpoint_threshold": (mask_value_clusters[0][-1] + mask_value_clusters[1][0]) // 2 if len(mask_value_clusters) == 2 else None,
            "observed_value_clusters": [[cluster[0], cluster[-1]] for cluster in mask_value_clusters],
            "status": "candidate only; validate semantic meaning visually and against authoritative annotation guidance",
        },
        "mask_value_file_counts": dict(mask_value_files),
        "mask_pixel_counts": dict(mask_pixel_counts),
        "mask_value_counts_by_file": {x["path"]: x.get("unique_values", []) for x in mask_info},
        "metadata_files": sorted(metadata_files), "adjacent_metadata_csvs": adjacent_csvs,
        "missing_masks_for_images": sorted(missing_masks), "unmatched_masks": unmatched_masks,
        "ambiguous_pairs": ambiguous, "dimension_mismatches": dimension_mismatches,
        "corrupt_files": corrupt, "duplicate_content": duplicate_content,
        "duplicate_candidate_file_count": duplicate_candidate_file_count,
        "duplicate_image_pairs": duplicate_image_pairs,
        "duplicate_image_pair_count": len(duplicate_image_pairs),
        "duplicate_image_pairs_crossing_splits": sum(pair["crosses_split"] for pair in duplicate_image_pairs),
        "duplicate_image_pairs_with_different_masks": sum(not pair["masks_identical"] for pair in duplicate_image_pairs),
        "duplicate_image_pairs_with_severity_conflict": sum(pair["severity_conflict"] for pair in duplicate_image_pairs),
        "duplicate_check_performed": duplicate_hashes,
        "note": "The binary threshold is an inference from observed pixel-value clusters; the archive does not document the label convention.",
    }
    if archive:
        archive.close()
    return report


def create_overlays(source: Path, output: Path, pairs: list[tuple[str, str]], count: int = 4) -> list[str]:
    """Write a few masked image composites for visual alignment review."""
    from PIL import ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    is_zip = source.is_file() and source.suffix.lower() == ".zip"
    archive = zipfile.ZipFile(source) if is_zip else None

    def open_image(name: str) -> Image.Image:
        if archive:
            return Image.open(archive.open(name))
        return Image.open(source / Path(name))

    created = []
    for image_name, mask_name in pairs[:count]:
        with open_image(image_name) as im, open_image(mask_name) as mask:
            base = im.convert("RGB")
            if mask.mode != "L":
                mask = mask.convert("L")
            m = mask
            if m.size != base.size:
                if m.size == (base.height, base.width) and base.width != base.height:
                    m = m.transpose(Image.Transpose.ROTATE_90)
                else:
                    continue
            overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
            # Midpoint threshold separates the observed 0-9 / 247-255 clusters.
            alpha = m.point(lambda x: 105 if x >= 128 else 0)
            overlay.putalpha(alpha)
            composite = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
            composite.thumbnail((1000, 1000))
            label = Path(image_name).stem
            ImageDraw.Draw(composite).text((8, 8), f"Mask overlay: {label}", fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
            target = output / f"overlay_{label}.png"
            composite.save(target)
            created.append(str(target))
    if archive:
        archive.close()
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Dataset directory or ZIP archive")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_audit/report.json"))
    parser.add_argument("--overlays", type=Path, help="Optional output directory for a few sample overlays")
    parser.add_argument("--overlay-count", type=int, default=4)
    parser.add_argument("--hash-duplicates", action="store_true", help="Hash same-size image/mask candidates to detect exact duplicates (can take time)")
    args = parser.parse_args()
    report = audit(args.source, duplicate_hashes=args.hash_duplicates)
    if args.overlays:
        pairs = []
        if args.source.is_file() and args.source.suffix.lower() == ".zip":
            with zipfile.ZipFile(args.source) as z:
                names = [_norm(i.filename) for i in z.infolist() if not i.is_dir()]
        else:
            names = [_norm(str(p.relative_to(args.source))) for p in args.source.rglob("*") if p.is_file()]
        for image in sorted(p for p in names if Path(p).suffix.lower() in IMAGE_EXTS):
            p = Path(image)
            mask = next((m for m in names if Path(m).parent == p.parent and Path(m).stem.lower() == p.stem.lower() and Path(m).suffix.lower() in MASK_EXTS), None)
            if mask and (image, mask):
                pairs.append((image, mask))
        report["overlay_files"] = create_overlays(args.source, args.overlays, pairs, args.overlay_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("image_count", "mask_count", "paired_count", "split_image_counts", "image_dimensions", "mask_dimensions", "mask_unique_pixel_values", "corrupt_files", "missing_masks_for_images", "unmatched_masks")}, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
