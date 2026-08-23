#!/usr/bin/env python3
"""Capture a non-secret environment manifest beside each AnySearch run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "anysearch",
    "slime",
    "torch",
    "transformers",
    "ray",
    "sglang",
    "sglang-router",
    "numpy",
    "datasets",
    "faiss-cpu",
    "faiss-gpu",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--slime-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "seed": args.seed,
        "command": sys.argv,
    }
    if args.config:
        manifest["config"] = {"path": str(args.config.resolve()), "sha256": sha256(args.config)}
    if args.slime_dir:
        manifest["slime"] = {
            "path": str(args.slime_dir.resolve()),
            "commit": command_output(["git", "rev-parse", "HEAD"], cwd=args.slime_dir),
            "dirty": bool(command_output(["git", "status", "--porcelain"], cwd=args.slime_dir)),
        }
    if args.model_dir:
        config_path = args.model_dir / "config.json"
        manifest["model"] = {
            "path": str(args.model_dir.resolve()),
            "config_sha256": sha256(config_path) if config_path.exists() else None,
        }
    manifest["nvidia_smi"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
