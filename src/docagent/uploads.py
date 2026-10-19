"""Submitted files: copied into a content-addressed upload store, zips expanded safely.

Why a store: retry / resume must not depend on where the file came from (Gradio's temp dir is cleaned up, users
move files). `<upload_dir>/<sha256><ext>` = one copy per distinct content, whatever its name or how often it is
uploaded; the original name travels with the Attachment for display, drawing ids and `--file` lookup.

Zips: members are streamed straight into the store by content - nothing is ever written under a name taken from
the archive, so path traversal cannot happen by construction. Unsafe names are still reported (they signal a
crafted archive), and limits bound member count and size (zip bombs).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .schemas import Attachment

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _place(tmp: Path, sha: str, ext: str, upload_dir: Path) -> Path:
    """Move a fully written temp file to its content address (atomic; an existing copy wins)."""
    dst = upload_dir / f"{sha}{ext.lower()}"
    if dst.exists():
        tmp.unlink()
    else:
        os.replace(tmp, dst)
    return dst


def store_file(path: str | Path, upload_dir: Path, name: str | None = None) -> Attachment:
    path = Path(path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    att = Attachment.from_path(path, name=name)            # hashes the original
    dst = upload_dir / f"{att.sha256}{path.suffix.lower()}"
    if path.resolve() != dst.resolve() and not dst.exists():
        fd, tmp = tempfile.mkstemp(dir=upload_dir, prefix=".up-")
        os.close(fd)
        shutil.copyfile(path, tmp)
        _place(Path(tmp), att.sha256, path.suffix, upload_dir)
    return att.model_copy(update={"path": str(dst)})


def _unsafe(member: str) -> str | None:
    p = PurePosixPath(member.replace("\\", "/"))
    if member.startswith(("/", "\\")) or (len(member) > 1 and member[1] == ":"):
        return "absolute path"
    if ".." in p.parts:
        return "path traversal ('..')"
    return None


def extract_zip(zip_path: str | Path, upload_dir: Path, max_members: int, max_total: int,
                max_member: int) -> tuple[list[Attachment], list[str]]:
    """-> (attachments, notes). Raises ValueError when the whole archive is refused."""
    zip_path = Path(zip_path)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"{zip_path.name}: not a valid zip ({e})")
    with zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise ValueError(f"{zip_path.name}: {len(infos)} members, limit {max_members} - refused")
        total = sum(i.file_size for i in infos)
        if total > max_total:
            raise ValueError(f"{zip_path.name}: {total / 1e9:.1f} GB uncompressed, limit {max_total / 1e9:.1f} GB - refused")
        upload_dir.mkdir(parents=True, exist_ok=True)
        atts: list[Attachment] = []
        notes: list[str] = []
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name.replace("\\", "/")).parts
            if info.is_dir():
                continue
            if why := _unsafe(name):
                notes.append(f"{zip_path.name}: skipped {name!r}: {why}")
                continue
            if parts and (parts[0] == "__MACOSX" or any(x.startswith(".") for x in parts)):
                continue                                          # macOS resource forks, hidden files
            ext = PurePosixPath(parts[-1]).suffix.lower()
            if ext not in SUPPORTED:
                notes.append(f"{zip_path.name}: skipped {name!r}: unsupported type")
                continue
            if info.file_size > max_member:
                notes.append(f"{zip_path.name}: skipped {name!r}: {info.file_size / 1e6:.0f} MB > limit")
                continue
            # stream into a temp file in the store, hashing on the way; cap actual bytes (header can lie)
            fd, tmp = tempfile.mkstemp(dir=upload_dir, prefix=".up-")
            h, n, too_big = hashlib.sha256(), 0, False
            with os.fdopen(fd, "wb") as out, zf.open(info) as src:
                while block := src.read(1 << 20):
                    n += len(block)
                    if n > max_member:
                        too_big = True
                        break
                    h.update(block)
                    out.write(block)
            if too_big:
                os.unlink(tmp)
                notes.append(f"{zip_path.name}: skipped {name!r}: larger than declared / limit")
                continue
            dst = _place(Path(tmp), h.hexdigest(), ext, upload_dir)
            atts.append(Attachment.from_path(dst, name=parts[-1]))
        if not atts:
            notes.append(f"{zip_path.name}: no supported files inside")
        return atts, notes


def expand(paths: list[str | Path], cfg) -> tuple[list[Attachment], list[str]]:
    """Files, folders and zips -> stored Attachments (+ notes on what was skipped). Folders: supported files
    and zips inside, recursively, sorted."""
    upload_dir = cfg.path(cfg.store.upload_dir)
    b = cfg.batch
    atts: list[Attachment] = []
    notes: list[str] = []
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED | {".zip"}
                            and not any(x.startswith(".") for x in f.relative_to(p).parts))
        else:
            files.append(p)
    for f in files:
        if f.suffix.lower() == ".zip":
            try:
                z, n = extract_zip(f, upload_dir, b.max_zip_members, b.max_zip_bytes, b.max_member_bytes)
            except ValueError as e:
                notes.append(str(e))
                continue
            atts += z
            notes += n
        else:
            atts.append(store_file(f, upload_dir))
    return atts, notes
