"""Pure Python Zero-Dependency Distribution Builder (PEP 427 Wheel + PEP 517 sdist).

Builds 100% compliant `.whl` and `.tar.gz` distributions without requiring
external build packages.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src" / "veriflux"
DIST_DIR = ROOT_DIR / "dist"

VERSION = "0.1.0"
PKG_NAME = "veriflux"
WHEEL_NAME = f"{PKG_NAME}-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"{PKG_NAME}-{VERSION}.tar.gz"


def _urlsafe_b64encode_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _file_hash_and_size(data: bytes) -> tuple[str, int]:
    digest = hashlib.sha256(data).digest()
    hash_str = "sha256=" + _urlsafe_b64encode_nopad(digest)
    return hash_str, len(data)


def build_wheel() -> Path:
    """Build pure Python universal wheel."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    wheel_path = DIST_DIR / WHEEL_NAME
    if wheel_path.exists():
        wheel_path.unlink()

    dist_info_dir = f"{PKG_NAME}-{VERSION}.dist-info"
    records: list[str] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add all package files from src/veriflux
        for root, _, files in os.walk(SRC_DIR):
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                file_path = Path(root) / file
                rel_to_src = file_path.relative_to(ROOT_DIR / "src")
                archive_name = str(rel_to_src).replace("\\", "/")

                with open(file_path, "rb") as f:
                    data = f.read()

                zf.writestr(archive_name, data)
                h, s = _file_hash_and_size(data)
                records.append(f"{archive_name},{h},{s}")

        # 2. Add WHEEL file
        wheel_content = (
            "Wheel-Version: 1.0\n"
            "Generator: veriflux_dist_builder (1.0)\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("utf-8")
        wheel_meta_name = f"{dist_info_dir}/WHEEL"
        zf.writestr(wheel_meta_name, wheel_content)
        h, s = _file_hash_and_size(wheel_content)
        records.append(f"{wheel_meta_name},{h},{s}")

        # 3. Add METADATA file
        readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        metadata_lines = [
            "Metadata-Version: 2.1",
            f"Name: {PKG_NAME}",
            f"Version: {VERSION}",
            "Summary: The definitive AI/ML testing, evaluation, guardrail, and reliability library. Built from scratch.",
            "Author: Veriflux Contributors",
            "License: Apache-2.0",
            "Requires-Python: >=3.10",
            "Classifier: Development Status :: 4 - Beta",
            "Classifier: Intended Audience :: Developers",
            "Classifier: Intended Audience :: Science/Research",
            "Classifier: License :: OSI Approved :: Apache Software License",
            "Classifier: Operating System :: OS Independent",
            "Classifier: Programming Language :: Python :: 3",
            "Classifier: Programming Language :: Python :: 3.10",
            "Classifier: Programming Language :: Python :: 3.11",
            "Classifier: Programming Language :: Python :: 3.12",
            "Classifier: Programming Language :: Python :: 3.13",
            "Classifier: Programming Language :: Python :: 3.14",
            "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Classifier: Topic :: Software Development :: Testing",
            "Classifier: Framework :: Pytest",
            "Description-Content-Type: text/markdown",
            "",
            readme_text,
        ]
        metadata_content = "\n".join(metadata_lines).encode("utf-8")
        meta_name = f"{dist_info_dir}/METADATA"
        zf.writestr(meta_name, metadata_content)
        h, s = _file_hash_and_size(metadata_content)
        records.append(f"{meta_name},{h},{s}")

        # 4. Add entry_points.txt
        entry_points_content = (
            "[console_scripts]\n"
            "veriflux = veriflux.cli.main:cli_entry\n\n"
            "[pytest11]\n"
            "veriflux = veriflux.pytest_plugin.plugin\n"
        ).encode("utf-8")
        ep_name = f"{dist_info_dir}/entry_points.txt"
        zf.writestr(ep_name, entry_points_content)
        h, s = _file_hash_and_size(entry_points_content)
        records.append(f"{ep_name},{h},{s}")

        # 5. Add RECORD
        records.append(f"{dist_info_dir}/RECORD,,")
        record_content = "\n".join(records) + "\n"
        zf.writestr(f"{dist_info_dir}/RECORD", record_content.encode("utf-8"))

    print(f"[Build] Built universal wheel: {wheel_path.name} ({wheel_path.stat().st_size:,} bytes)")
    return wheel_path


def build_sdist() -> Path:
    """Build standard sdist tar.gz."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    sdist_path = DIST_DIR / SDIST_NAME
    if sdist_path.exists():
        sdist_path.unlink()

    base_dir = f"{PKG_NAME}-{VERSION}"
    files_to_include = [
        ("pyproject.toml", ROOT_DIR / "pyproject.toml"),
        ("README.md", ROOT_DIR / "README.md"),
        ("LICENSE", ROOT_DIR / "LICENSE"),
    ]

    with tarfile.open(sdist_path, "w:gz") as tar:
        for archive_rel, file_path in files_to_include:
            if file_path.exists():
                tar.add(str(file_path), arcname=f"{base_dir}/{archive_rel}")

        # Add src/
        for root, _, files in os.walk(ROOT_DIR / "src"):
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                p = Path(root) / file
                rel = p.relative_to(ROOT_DIR)
                tar.add(str(p), arcname=f"{base_dir}/{rel.as_posix()}")

    print(f"[Build] Built source distribution: {sdist_path.name} ({sdist_path.stat().st_size:,} bytes)")
    return sdist_path


if __name__ == "__main__":
    print(f"Building Veriflux v{VERSION} distribution artifacts...")
    whl = build_wheel()
    sdist = build_sdist()
    print("Distribution builds complete. Ready for PyPI and GitHub Releases.")
