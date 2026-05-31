"""
Run HM-Conformer test/inference on the SAME data distribution used during training:

Training distribution (see run_train_selected_fake_models.py):
- Keep ALL real samples
- Keep ONLY fake samples where labels.json entry["model_or_speaker"] == trained_on_fake_model

This script runs HM-Conformer in TEST mode and supports:
- One run with ALL languages together (default)
- Or a per-language loop (for later summarization)

Relies on env overrides in `hm_conformer/arguments.py`:
  HM_SELECTED_LANGUAGE, HM_SELECTED_FAKE_MODEL, HM_LOAD_EPOCH, HM_PATH_PARAMS,
  HM_LABELS_PATH, HM_DATASET_ROOT, HM_USABLE_GPU, HM_TEST, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _load_languages(labels_path: Path) -> List[str]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    langs = sorted({str(x.get("language")).strip() for x in data if x.get("language")})
    return [lang for lang in langs if lang]


def _run_and_tee(cmd: List[str], env: dict, *, live: bool, log_path: Optional[Path]) -> int:
    """
    Run a subprocess.
    - If log_path is provided, write all output to it (and stream to stdout if live=True).
    - If log_path is None, stream to stdout only.
    """
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("w", encoding="utf-8")
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write(f"START: {_timestamp()}\n\n")
        f.flush()
    else:
        f = None

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            if f is not None:
                f.write(line)
            if live:
                sys.stdout.write(line)
                sys.stdout.flush()
        exit_code = p.wait()

        if f is not None:
            f.write("\n")
            f.write(f"END: {_timestamp()}\n")
            f.write(f"EXIT_CODE: {exit_code}\n")
            f.flush()

        return exit_code
    finally:
        if f is not None:
            f.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test a trained HM-Conformer checkpoint on the same (in-domain) dataset distribution: all real + selected fake model."
    )
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels.json")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root folder (optional; passed to HM_DATASET_ROOT)")
    parser.add_argument("--path_params", type=str, default=None, help="Checkpoint models folder (passed to HM_PATH_PARAMS)")
    parser.add_argument("--load_epoch", type=str, default=None, help='Epoch to load (e.g. "60") or "none"/"latest" for auto')
    parser.add_argument("--usable_gpu", type=str, default=None, help='CUDA_VISIBLE_DEVICES override (passed to HM_USABLE_GPU), e.g. "0" or "0,1"')
    parser.add_argument(
        "--trained_on_fake_model",
        type=str,
        required=True,
        help='Fake model name used during training (labels.json entry["model_or_speaker"]). Only this fake model will be used in test (reals always included).',
    )
    parser.add_argument(
        "--all_languages_together",
        action="store_true",
        help="Run a single test job using all languages together (no per-language filtering/loop).",
    )
    parser.add_argument("--out_dir", type=str, default="experiments/in_domain_eval/outputs", help="Output folder for per-language .txt logs (set to empty to disable)")
    parser.add_argument("--include", type=str, default=None, help='Comma-separated language codes to include (e.g. "en,es,it")')
    parser.add_argument("--exclude", type=str, default=None, help='Comma-separated language codes to exclude')
    parser.add_argument("--continue_on_error", action="store_true", help="Continue to next language even if a run fails (default: stop)")
    parser.add_argument("--no_live", action="store_true", help="Do not stream logs to console (still writes .txt files if out_dir is set)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # .../deepfake-speech-detection
    hm_main = repo_root / "HM-Conformer" / "hm_conformer" / "main.py"
    if not hm_main.exists():
        raise FileNotFoundError(f"Could not find HM-Conformer entrypoint: {hm_main}")

    labels_path = Path(args.labels_path).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    path_params = Path(args.path_params).resolve() if args.path_params else None

    out_dir: Optional[Path]
    if args.out_dir is None or str(args.out_dir).strip() == "":
        out_dir = None
    else:
        out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    trained_on = str(args.trained_on_fake_model).strip()
    if not trained_on:
        print("--trained_on_fake_model cannot be empty.", file=sys.stderr)
        return 2

    languages = _load_languages(labels_path)
    include = [x.strip() for x in (args.include.split(",") if args.include else []) if x.strip()]
    exclude = {x.strip() for x in (args.exclude.split(",") if args.exclude else []) if x.strip()}
    if include:
        include_set = set(include)
        languages = [lang for lang in languages if lang in include_set]
    if exclude:
        languages = [lang for lang in languages if lang not in exclude]
    if not languages and not args.all_languages_together:
        print("No languages to run (after include/exclude).", file=sys.stderr)
        return 2

    def _make_env(selected_language: str) -> dict:
        env = os.environ.copy()
        env["HM_TEST"] = "1"
        env["HM_LABELS_PATH"] = str(labels_path)

        # Filter to match training distribution:
        # keep ALL real, keep ONLY fake samples where model_or_speaker == trained_on
        env["HM_SELECTED_FAKE_MODEL"] = trained_on

        # Ensure exclusion is disabled (if user has it set in their shell)
        env.pop("HM_EXCLUDE_FAKE_MODELS", None)

        env["HM_SELECTED_LANGUAGE"] = selected_language

        if dataset_root is not None:
            env["HM_DATASET_ROOT"] = str(dataset_root)
        if path_params is not None:
            env["HM_PATH_PARAMS"] = str(path_params)
        if args.usable_gpu is not None:
            env["HM_USABLE_GPU"] = args.usable_gpu
        if args.load_epoch is not None:
            env["HM_LOAD_EPOCH"] = args.load_epoch

        env.setdefault("HM_PROJECT", "In-Domain-Testing")
        return env

    cmd = [sys.executable, "-u", str(hm_main)]

    if args.all_languages_together:
        env = _make_env("all")
        env["HM_NAME"] = f"HM-Conformer_in_domain_{trained_on}_ALL_LANGS"

        print("\n" + "=" * 90)
        print("Single run: ALL languages together (in-domain)")
        print(f"Selected fake model: {trained_on}")
        if path_params is not None:
            print(f"path_params: {path_params}")
        if args.load_epoch is not None:
            print(f"load_epoch: {args.load_epoch}")
        print("=" * 90)

        log_path = (out_dir / f"output_in_domain_{trained_on}_ALL_LANGS.txt") if out_dir is not None else None
        t0 = time.time()
        exit_code = _run_and_tee(cmd, env, live=(not args.no_live), log_path=log_path)
        dt = time.time() - t0

        print(f"\nDone ALL_LANGS exit_code={exit_code} elapsed={dt:.1f}s")
        if log_path is not None:
            print(f"Log file: {log_path}")
        return exit_code

    total_langs = len(languages)
    for idx, lang in enumerate(languages, start=1):
        env = _make_env(lang)
        env["HM_NAME"] = f"HM-Conformer_in_domain_{trained_on}_{lang}"

        print("\n" + "=" * 90)
        print(f"[{idx}/{total_langs}] language={lang!r} (in-domain)")
        print(f"Selected fake model: {trained_on}")
        if path_params is not None:
            print(f"path_params: {path_params}")
        if args.load_epoch is not None:
            print(f"load_epoch: {args.load_epoch}")
        print("=" * 90)

        log_path = (out_dir / f"output_in_domain_{trained_on}_{lang}.txt") if out_dir is not None else None
        t0 = time.time()
        exit_code = _run_and_tee(cmd, env, live=(not args.no_live), log_path=log_path)
        dt = time.time() - t0

        print(f"\n[{idx}/{total_langs}] Done language={lang!r} exit_code={exit_code} elapsed={dt:.1f}s")
        if log_path is not None:
            print(f"Log file: {log_path}")

        if exit_code != 0 and not args.continue_on_error:
            print(f"Run failed for language={lang} (exit_code={exit_code}).", file=sys.stderr)
            return exit_code

    print("\nAll tests finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

