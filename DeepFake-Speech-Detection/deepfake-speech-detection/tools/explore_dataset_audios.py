import os
import argparse
from collections import defaultdict, Counter

def summarize_dataset(root_path: str):
    """
    Print a summary of the dataset structure, including found languages, metadata files,
    file extensions, and their counts, plus total files per language.
    """
    summary = {
        "languages": set(),
        "metadata_files": defaultdict(list),  # {ext: [paths]}
        "file_extensions": Counter(),
        "total_wavs": 0,
        "wavs_per_dir": dict(),
        "files_per_language": defaultdict(int),  # lang: count of files
        "all_files_scanned": 0,
    }

    # Recognized language directory patterns (could be expanded for your dataset)
    LANGUAGE_INDICATORS = {"en", "de", "fr", "it", "es", "pt", "pl", "cs", "ru", "zh", "ja", "ko", "tr", "ar"}

    def infer_language_from_path(dirpath):
        """
        Tries to infer the language code from the directory path.
        Returns the last matching language code found (or None).
        """
        parts = os.path.normpath(dirpath).split(os.sep)
        langs = []
        for p in parts:
            if p in LANGUAGE_INDICATORS:
                langs.append(p)
            # e.g. "en_US" or "en_GB" or "pt_BR"
            elif len(p) == 5 and p[2] == '_' and p[:2] in LANGUAGE_INDICATORS:
                langs.append(p)
        return langs[-1] if langs else None

    def visit(path: str):
        for dirpath, dirnames, filenames in os.walk(path):
            lang = infer_language_from_path(dirpath)
            if lang:
                summary["languages"].add(lang)
                summary["files_per_language"][lang] += len(filenames)
            summary["all_files_scanned"] += len(filenames)
            wav_count = 0
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                summary["file_extensions"][ext] += 1

                if ext == ".wav":
                    wav_count += 1
                    summary["total_wavs"] += 1

                if ext in {".csv", ".json", ".txt"}:
                    fpath = os.path.join(dirpath, fname)
                    summary["metadata_files"][ext].append(fpath)

            if wav_count:
                summary["wavs_per_dir"][dirpath] = wav_count

    root_path = os.path.abspath(root_path)
    if not os.path.isdir(root_path):
        print(f"Provided path does not exist or is not a directory: {root_path}")
        return

    visit(root_path)

    print("=== DATASET SUMMARY ===")
    print(f"Root: {root_path}")
    print()

    # Languages
    languages = sorted(summary["languages"])
    print(f"Languages found ({len(languages)}):")
    if languages:
        for lang in languages:
            nfiles = summary["files_per_language"].get(lang, 0)
            print(f"  - {lang}: {nfiles} file(s)")
    else:
        print("  [Could not infer any language directories]")
    print()

    # Quantity of total files per language
    if languages:
        print("Total files per language:")
        for lang in languages:
            print(f"  {lang}: {summary['files_per_language'][lang]}")
    print()

    # Metadata files
    print(f"Metadata files found:")
    for ext, paths in summary["metadata_files"].items():
        print(f"  {ext}: {len(paths)} file(s)")
        for p in sorted(paths)[:5]:
            print(f"    - {p}")
        if len(paths) > 5:
            print("    ...")
    if not summary["metadata_files"]:
        print("  [No metadata (.csv/.json/.txt) files found]")
    print()

    # File extension distribution
    print("File extension counts:")
    for ext, cnt in summary["file_extensions"].most_common():
        label = ext if ext else "[no extension]"
        print(f"  {label}: {cnt}")
    print()

    print(f"Total .wav files: {summary['total_wavs']}")
    print(f"Total files scanned: {summary['all_files_scanned']}")

def main():
    parser = argparse.ArgumentParser(description="Summarize dataset: languages, metadata files, file extensions, and .wav stats.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/raw_data",
        help="Root folder to summarize (default: dataset_audios in repo)",
    )
    args = parser.parse_args()
    summarize_dataset(args.path)

if __name__ == "__main__":
    main()
