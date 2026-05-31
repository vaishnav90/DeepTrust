"""
Run HM-Conformer training in a loop, one fake TTS model at a time.

Each run trains on:
- ALL real samples
- ONLY fake samples where labels.json entry["model_or_speaker"] == selected fake model

This script configures HM-Conformer via environment-variable overrides supported by:
`hm_conformer/arguments.py`
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


FAKE_MODELS = [
    "OuteTTS",
    "griffin_lim",
    "tts_models_multilingual_multi-dataset_bark",
    "tts_models_multilingual_multi-dataset_xtts_v1.1",
    "tts_models_multilingual_multi-dataset_xtts_v2",
    "Llasa-1B-Multilingual",
    "Chatterbox Multilingual",
    "Resemble.ai (April 12th, 2025)",
]


def _sanitize_for_name(s: str) -> str:
    # Safe-ish experiment name; keep it readable.
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def main() -> int:
    here = Path(__file__).resolve().parent  # .../HM-Conformer
    repo_root = here.parent  # .../deepfake-speech-detection
    default_dataset_root = repo_root / "dataset_audios"
    default_labels_path = default_dataset_root / "labels.json"
    default_log_root = here / "results"

    parser = argparse.ArgumentParser(description="Train HM-Conformer per fake model (all real + selected fake).")
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root),
        help="Dataset root folder (must contain audio/...). Default: <repo>/dataset_audios",
    )
    parser.add_argument(
        "--labels-path",
        default=str(default_labels_path),
        help="Path to labels.json. Default: <repo>/dataset_audios/labels.json",
    )
    parser.add_argument(
        "--log-root",
        default=str(default_log_root),
        help="Where to write logs/models via the local logger. Default: HM-Conformer/results",
    )
    parser.add_argument(
        "--project",
        default="Multilingual-Domain-Training",
        help="Experiment project name (wandb/neptune/local grouping).",
    )
    parser.add_argument(
        "--name-prefix",
        default="HM-Conformer_",
        help="Prefix for args['name'] per run.",
    )
    parser.add_argument(
        "--usable-gpu",
        default=None,
        help="Optional override for arguments.py usable_gpu (e.g. '0' or '0,1').",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable class-balanced sampling (oversample minority during training; does NOT discard data).",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable balancing (even if --balance is set elsewhere).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run only one model (exact name match from the list).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    labels_path = Path(args.labels_path).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()

    if not labels_path.exists():
        raise SystemExit(f"labels.json not found at: {labels_path}")
    if not dataset_root.exists():
        raise SystemExit(f"dataset_root not found at: {dataset_root}")

    main_py = here / "hm_conformer" / "main.py"
    if not main_py.exists():
        raise SystemExit(f"HM-Conformer main.py not found at: {main_py}")

    models = list(FAKE_MODELS)
    if args.only is not None:
        if args.only not in models:
            raise SystemExit(f"--only must be one of: {models!r}")
        models = [args.only]

    # Balancing toggle
    balance_enabled = bool(args.balance) and (not bool(args.no_balance))

    for i, model in enumerate(models, start=1):
        exp_name = f"{args.name_prefix}{_sanitize_for_name(model)}"
        print("\n" + "=" * 80)
        print(f"[{i}/{len(models)}] Training fake model: {model!r}")
        print(f"Experiment name: {exp_name}")
        print(f"Balance train:   {balance_enabled}")
        print("=" * 80)

        env = os.environ.copy()
        # Required filters for this sweep
        env["HM_SELECTED_LANGUAGE"] = "all"  # == None => all languages
        env["HM_SELECTED_FAKE_MODEL"] = model
        env["HM_PROJECT"] = str(args.project)
        env["HM_NAME"] = exp_name

        # Make local paths work out-of-the-box (arguments.py defaults are Colab/Docker-ish)
        env["HM_DATASET_ROOT"] = str(dataset_root)
        env["HM_LABELS_PATH"] = str(labels_path)
        env["HM_PATH_LOG"] = str(log_root)

        # Ensure we are training (not test-only)
        env["HM_TEST"] = "0"

        if args.usable_gpu is not None:
            env["HM_USABLE_GPU"] = str(args.usable_gpu)

        # Optional balancing (implemented in hm_conformer/main.py)
        env["HM_BALANCE_TRAIN"] = "1" if balance_enabled else "0"

        # Run in the hm_conformer/ folder so local imports like `import arguments` work.
        cmd = [sys.executable, "-u", str(main_py)]
        subprocess.run(cmd, env=env, cwd=str(main_py.parent), check=True)

    print("\nAll trainings finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

