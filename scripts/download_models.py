#!/usr/bin/env python3
"""
Automate downloading:
  1) Hugging Face model: BAAI/bge-m3  -> ./models/bge-m3
  2) Docling Tools models (all)       -> ./models/docling

Usage:
  python download_models.py
  python download_models.py --models-dir ./models --hf-revision main --retries 3

Exit codes:
  0 = success
  1 = partial failure
  2 = complete failure
"""

import os
import sys
import shutil
import subprocess
import time
import argparse
from typing import List, Tuple


def log(msg: str):
    print(f"[download] {msg}", flush=True)


def run_cmd(cmd: List[str], cwd: str = None, env: dict = None) -> Tuple[int, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out, _ = proc.communicate()
        return proc.returncode, out
    except Exception as e:
        return 1, f"ERROR executing {' '.join(cmd)}: {e}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def have_executable(name: str) -> bool:
    return shutil.which(name) is not None


def pip_install(packages: List[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    code, out = run_cmd(cmd)
    log(out)
    return code == 0


def download_hf_model(target_dir: str, retries: int, revision: str = None) -> bool:
    """
    Try CLI first: `huggingface-cli download`
    Fallback to Python API: huggingface_hub.snapshot_download
    """
    ensure_dir(target_dir)
    model_id = "BAAI/bge-m3"
    log(f"Downloading HF model: {model_id} -> {target_dir}")

    # Prefer CLI if available
    if have_executable("huggingface-cli"):
        base_cmd = [
            "huggingface-cli",
            "download",
            model_id,
            "--local-dir",
            target_dir,
            "--local-dir-use-symlinks",
            "False",
        ]
        if revision:
            base_cmd += ["--revision", revision]

        for attempt in range(1, retries + 1):
            log(f"(HF CLI) Attempt {attempt}/{retries}: {' '.join(base_cmd)}")
            code, out = run_cmd(base_cmd)
            log(out)
            if code == 0:
                log("HF CLI download succeeded.")
                return True
            time.sleep(2 * attempt)

        log("HF CLI download failed after retries; trying Python API fallback...")

    # Python API fallback
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("huggingface_hub not found; installing...")
        if not pip_install(["huggingface_hub>=0.23.0"]):
            log("Failed to install huggingface_hub.")
            return False
        from huggingface_hub import snapshot_download  # re-import after install

    for attempt in range(1, retries + 1):
        try:
            log(f"(HF API) Attempt {attempt}/{retries} via snapshot_download")
            snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                revision=revision,
                ignore_patterns=[
                    "*.msgpack",
                    "*.parquet",
                ],  # optional: skip HF cache artifacts
            )
            log("HF API download succeeded.")
            return True
        except Exception as e:
            log(f"HF API error: {e}")
            time.sleep(2 * attempt)

    return False


def download_docling_models(target_dir: str, retries: int) -> bool:
    """
    Prefer `docling-tools` CLI if present.
    Fallback to installing it.
    If CLI ultimately unavailable, attempt Docling Python API if exposed (best effort).
    """
    ensure_dir(target_dir)
    log(f"Downloading Docling Tools models -> {target_dir}")

    def try_cli():
        base_cmd = ["docling-tools", "models", "download", "--all", "-o", target_dir]
        for attempt in range(1, retries + 1):
            log(f"(Docling CLI) Attempt {attempt}/{retries}: {' '.join(base_cmd)}")
            code, out = run_cmd(base_cmd)
            log(out)
            if code == 0:
                log("Docling CLI download succeeded.")
                return True
            time.sleep(2 * attempt)
        return False

    # Try CLI if exists
    if have_executable("docling-tools"):
        if try_cli():
            return True
        else:
            log(
                "Docling CLI present but download failed; will try reinstall then retry."
            )

    # Try installing docling-tools
    log("Installing docling-tools...")
    # Try a few likely package names
    candidate_pkgs = [
        "docling-tools",  # if published under this name
        "docling[tools]",  # extras-based install
        "docling",  # minimal, may provide the CLI via extras
    ]
    installed = False
    for pkg in candidate_pkgs:
        if pip_install([pkg]):
            installed = True
            if have_executable("docling-tools"):
                break

    if have_executable("docling-tools"):
        if try_cli():
            return True
        else:
            log("Docling CLI still failing after install.")

    # Python API fallback (best effort). Not all versions expose a direct API for models.
    try:
        import docling  # noqa: F401

        # If future versions expose programmatic download, add logic here.
        log(
            "Docling Python package found, but no stable programmatic download API is known; CLI is preferred."
        )
    except ImportError:
        log("Docling Python package not importable; unable to use API fallback.")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download HF and Docling models for deployment."
    )
    parser.add_argument(
        "--models-dir", default="./models", help="Root models directory."
    )
    parser.add_argument("--hf-subdir", default="bge-m3", help="Subdir for BAAI/bge-m3.")
    parser.add_argument(
        "--docling-subdir", default="docling", help="Subdir for Docling Tools models."
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional HF revision (e.g., 'main' or a commit hash).",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retries per download."
    )
    args = parser.parse_args()

    # removed erroneous assignments that referenced undefined `subdir`;
    # the underscored argparse attributes (hf_subdir/docling_subdir) are used below

    # Fix underscored attribute names (argparse auto converts dashes to underscores)
    hf_dir = os.path.join(args.models_dir, getattr(args, "hf_subdir", "bge-m3"))
    docling_dir = os.path.join(
        args.models_dir, getattr(args, "docling_subdir", "docling")
    )

    ensure_dir(args.models_dir)

    log(f"Target root: {args.models_dir}")
    log(f"HF model dir: {hf_dir}")
    log(f"Docling dir  : {docling_dir}")

    ok_hf = download_hf_model(
        target_dir=hf_dir, retries=args.retries, revision=args.hf_revision
    )
    ok_doc = download_docling_models(target_dir=docling_dir, retries=args.retries)

    if ok_hf and ok_doc:
        log("✅ All downloads completed successfully.")
        sys.exit(0)
    elif ok_hf or ok_doc:
        log("⚠️ Partial success. One of the downloads failed.")
        sys.exit(1)
    else:
        log("❌ Both downloads failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
