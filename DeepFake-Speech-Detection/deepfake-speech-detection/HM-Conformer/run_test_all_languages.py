"""
Run HM-Conformer test/inference for every language in labels.json.

What it does:
- Reads languages from labels.json (unique `language` values)
- Runs `HM-Conformer/hm_conformer/main.py` once per language (TEST mode)
- Captures full stdout/stderr into per-language .txt logs (like example/output_example.txt)
- Parses key metrics from logs and writes a markdown summary report for comparisons

This script relies on env overrides implemented in `hm_conformer/arguments.py`:
  HM_SELECTED_LANGUAGE, HM_LOAD_EPOCH, HM_PATH_PARAMS, HM_LABELS_PATH, HM_DATASET_ROOT, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ParsedResult:
    language: str
    exit_code: int
    test_total: Optional[int] = None
    test_real: Optional[int] = None
    test_fake: Optional[int] = None
    eer: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_auc: Optional[float] = None
    threshold: Optional[float] = None


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _load_languages(labels_path: Path) -> List[str]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    langs = sorted({str(x.get("language")).strip() for x in data if x.get("language")})
    return [lang for lang in langs if lang]


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s.replace(",", "").strip())
    except Exception:
        return None


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except Exception:
        return None


def parse_metrics_from_log(text: str, language: str, exit_code: int) -> ParsedResult:
    r = ParsedResult(language=language, exit_code=exit_code)

    # Dataset info line e.g.: TEST:  Real=18,498, Fake=12,000, Total=30,498
    m = re.search(r"^TEST:\s+Real=([\d,]+),\s+Fake=([\d,]+),\s+Total=([\d,]+)\s*$", text, re.MULTILINE)
    if m:
        r.test_real = _parse_int(m.group(1))
        r.test_fake = _parse_int(m.group(2))
        r.test_total = _parse_int(m.group(3))

    # Metrics block
    m = re.search(r"^EER:\s+([\d.]+)%\s*$", text, re.MULTILINE)
    if m:
        r.eer = _parse_float(m.group(1))
    m = re.search(r"^Accuracy:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.accuracy = _parse_float(m.group(1))
    m = re.search(r"^F1 Score:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.f1 = _parse_float(m.group(1))
    m = re.search(r"^Precision:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.precision = _parse_float(m.group(1))
    m = re.search(r"^Recall:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.recall = _parse_float(m.group(1))
    m = re.search(r"^ROC AUC:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.roc_auc = _parse_float(m.group(1))
    m = re.search(r"^Threshold:\s+([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        r.threshold = _parse_float(m.group(1))

    return r


def _run_and_tee(cmd: List[str], env: dict, log_path: Path, *, live: bool) -> int:
    """
    Run a subprocess and write all output to log_path.
    If live=True, also stream output to this process stdout.
    """
    # Force unbuffered python output (helps see progress immediately)
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write(f"START: {_timestamp()}\n")
        f.write("\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            if live:
                sys.stdout.write(line)
                sys.stdout.flush()
        exit_code = p.wait()

        f.write("\n")
        f.write(f"END: {_timestamp()}\n")
        f.write(f"EXIT_CODE: {exit_code}\n")
        f.flush()

    return exit_code


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _fmt_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "-"
    return f"{x:,}"


def write_summary_report(
    report_path: Path,
    results: List[ParsedResult],
    *,
    title: str,
    labels_path: Path,
    dataset_root: Optional[Path],
    path_params: Optional[Path],
    load_epoch: Optional[int],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort by EER (ascending), unknowns last
    results_sorted = sorted(results, key=lambda r: (r.eer is None, r.eer if r.eer is not None else 1e9, r.language))

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- labels.json: `{labels_path}`")
    if dataset_root is not None:
        lines.append(f"- dataset_root: `{dataset_root}`")
    if path_params is not None:
        lines.append(f"- path_params: `{path_params}`")
    lines.append(f"- load_epoch: `{load_epoch}`" if load_epoch is not None else "- load_epoch: `auto/latest`")
    lines.append("")
    lines.append("## Overall Results Summary")
    lines.append("")
    lines.append("| Language | Test Samples | EER (%) | Accuracy | F1 | Precision | Recall | ROC AUC | Exit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results_sorted:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"**{r.language}**",
                    _fmt_int(r.test_total),
                    _fmt_pct(r.eer, 2),
                    _fmt_float(r.accuracy, 4),
                    _fmt_float(r.f1, 4),
                    _fmt_float(r.precision, 4),
                    _fmt_float(r.recall, 4),
                    _fmt_float(r.roc_auc, 4),
                    str(r.exit_code),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Per-language Logs")
    lines.append("")
    lines.append("Each run produces a full console log (`.txt`) per language in the output folder.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HM-Conformer tests for all languages and save per-language logs + summary report.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels.json")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root folder (optional; passed to HM_DATASET_ROOT)")
    parser.add_argument("--path_params", type=str, default=None, help="Checkpoint models folder (passed to HM_PATH_PARAMS)")
    parser.add_argument("--load_epoch", type=str, default=None, help='Epoch to load (e.g. "60") or "none"/"latest" for auto')
    parser.add_argument("--usable_gpu", type=str, default=None, help='CUDA_VISIBLE_DEVICES override (passed to HM_USABLE_GPU), e.g. "0" or "0,1"')
    parser.add_argument("--out_dir", type=str, default="experiments/multilingual_eval/outputs", help="Output folder for per-language .txt logs")
    parser.add_argument("--report_path", type=str, default="experiments/multilingual_eval/summary_report.md", help="Path to write the summary report")
    parser.add_argument("--title", type=str, default="Multilingual HM-Conformer Evaluation Summary", help="Title for the markdown report")
    parser.add_argument("--include", type=str, default=None, help='Comma-separated language codes to include (e.g. "en,es,it")')
    parser.add_argument("--exclude", type=str, default=None, help='Comma-separated language codes to exclude')
    parser.add_argument("--continue_on_error", action="store_true", help="Continue to next language even if a run fails (default: stop)")
    parser.add_argument("--no_live", action="store_true", help="Do not stream logs to console (still writes .txt files)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # .../deepfake-speech-detection
    hm_main = repo_root / "HM-Conformer" / "hm_conformer" / "main.py"
    if not hm_main.exists():
        raise FileNotFoundError(f"Could not find HM-Conformer entrypoint: {hm_main}")

    labels_path = Path(args.labels_path).resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    report_path = (repo_root / args.report_path).resolve() if not Path(args.report_path).is_absolute() else Path(args.report_path).resolve()

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    path_params = Path(args.path_params).resolve() if args.path_params else None

    languages = _load_languages(labels_path)
    include = [x.strip() for x in (args.include.split(",") if args.include else []) if x.strip()]
    exclude = {x.strip() for x in (args.exclude.split(",") if args.exclude else []) if x.strip()}
    if include:
        include_set = set(include)
        languages = [lang for lang in languages if lang in include_set]
    if exclude:
        languages = [lang for lang in languages if lang not in exclude]
    if not languages:
        print("No languages to run (after include/exclude).", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[ParsedResult] = []
    total_langs = len(languages)
    for idx, lang in enumerate(languages, start=1):
        log_path = out_dir / f"output_{lang}.txt"

        env = os.environ.copy()
        env["HM_TEST"] = "1"
        env["HM_SELECTED_LANGUAGE"] = lang
        env["HM_LABELS_PATH"] = str(labels_path)
        if dataset_root is not None:
            env["HM_DATASET_ROOT"] = str(dataset_root)
        if path_params is not None:
            env["HM_PATH_PARAMS"] = str(path_params)
        if args.usable_gpu is not None:
            env["HM_USABLE_GPU"] = args.usable_gpu
        if args.load_epoch is not None:
            env["HM_LOAD_EPOCH"] = args.load_epoch

        # Make experiment naming deterministic per language (useful in logs/results)
        env.setdefault("HM_PROJECT", "Multilingual-Testing")
        env["HM_NAME"] = f"HM-Conformer_{lang}"
        env.setdefault("PYTHONUNBUFFERED", "1")

        # -u makes python unbuffered (important for real-time streaming)
        cmd = [sys.executable, "-u", str(hm_main)]

        # High-level progress prints
        print(f"\n[{idx}/{total_langs}] Running language={lang!r}")
        print(f"Log file: {log_path}")
        t0 = time.time()

        # Put key env settings at the top of each log
        preface = (
            f"HM_SELECTED_LANGUAGE={lang}\n"
            f"HM_LABELS_PATH={labels_path}\n"
            + (f"HM_DATASET_ROOT={dataset_root}\n" if dataset_root is not None else "")
            + (f"HM_PATH_PARAMS={path_params}\n" if path_params is not None else "")
            + (f"HM_LOAD_EPOCH={args.load_epoch}\n" if args.load_epoch is not None else "")
            + (f"HM_USABLE_GPU={args.usable_gpu}\n" if args.usable_gpu is not None else "")
            + "\n"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(preface, encoding="utf-8")

        # Run and tee output (append to the preface)
        # NOTE: We open the log in _run_and_tee, so we append the preface ourselves above.
        # To avoid overwriting, run _run_and_tee on a temp path then append. Simpler: rerun with append mode.
        # Here we re-implement minimal append behavior by calling _run_and_tee with a temporary path.
        tmp_log = log_path.with_suffix(".tmp")
        exit_code = _run_and_tee(cmd, env, tmp_log, live=(not args.no_live))
        with log_path.open("a", encoding="utf-8") as f:
            f.write(tmp_log.read_text(encoding="utf-8", errors="replace"))
        try:
            tmp_log.unlink()
        except OSError:
            pass

        dt = time.time() - t0
        print(f"[{idx}/{total_langs}] Done language={lang!r} exit_code={exit_code} elapsed={dt:.1f}s")

        text = log_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_metrics_from_log(text, language=lang, exit_code=exit_code)
        results.append(parsed)

        if exit_code != 0 and not args.continue_on_error:
            print(f"Run failed for language={lang} (exit_code={exit_code}). See log: {log_path}", file=sys.stderr)
            break

    # Write summary report
    load_epoch_val: Optional[int] = None
    if args.load_epoch is not None and str(args.load_epoch).strip().lower() not in ("none", "null", "auto", "latest"):
        try:
            load_epoch_val = int(str(args.load_epoch).strip())
        except ValueError:
            load_epoch_val = None

    write_summary_report(
        report_path,
        results,
        title=args.title,
        labels_path=labels_path,
        dataset_root=dataset_root,
        path_params=path_params,
        load_epoch=load_epoch_val,
    )

    print(f"Wrote per-language logs to: {out_dir}")
    print(f"Wrote summary report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

