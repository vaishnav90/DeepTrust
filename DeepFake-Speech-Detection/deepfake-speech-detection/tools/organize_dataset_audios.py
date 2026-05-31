import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_label_and_language(path_parts: Tuple[str, ...]) -> Tuple[str, str]:
    # Assumes structure: dataset_audios/raw_data/{real|fake}/<lang or lang_locale>/...
    if 'raw_data' not in path_parts:
        raise ValueError('Path does not include raw_data segment')
    idx = path_parts.index('raw_data')
    if idx + 1 >= len(path_parts):
        raise ValueError('Invalid raw_data path layout')
    label = path_parts[idx + 1]
    if label not in ('real', 'fake'):
        raise ValueError(f'Unknown label under raw_data: {label}')
    if idx + 2 >= len(path_parts):
        raise ValueError('Missing language segment after label')
    language_segment = path_parts[idx + 2]
    # Normalize language: take prefix before '_' if present (e.g., en_US -> en)
    language = language_segment.split('_')[0].lower()
    return label, language


def detect_model_or_speaker(label: str, wav_path: Path) -> str:
    parts = wav_path.parts
    # fake: dataset_audios/raw_data/fake/<lang>/<model>/.../file.wav
    if label == 'fake':
        try:
            lang_idx = parts.index('fake') + 1
            # model folder is right after language folder
            return parts[lang_idx + 1]
        except Exception:
            return 'unknown_model'
    # real: dataset_audios/raw_data/real/<lang>/by_book/<gender>/<speaker>/<book>/wavs/file.wav
    try:
        by_book_idx = parts.index('by_book')
        speaker = parts[by_book_idx + 2]
        return speaker
    except Exception:
        return 'unknown_speaker'


def collect_metadata_files_for_wav(wav_path: Path, label: str) -> List[str]:
    # For fake: metadata typically lives in the same directory as wavs (model folder)
    # For real: metadata lives one level above the wavs directory (book folder)
    if label == 'fake':
        search_dir = wav_path.parent
    else:
        search_dir = wav_path.parent.parent if wav_path.parent.name.lower() == 'wavs' else wav_path.parent

    if not search_dir.exists():
        return []

    meta_files: List[str] = []
    try:
        for entry in os.scandir(search_dir):
            if entry.is_file():
                name = entry.name.lower()
                if (name.endswith('.csv') or name.endswith('.json') or name.endswith('.txt')) and ('meta' in name or 'info' in name):
                    meta_files.append(str(Path(entry.path).resolve()))
    except Exception:
        pass

    return sorted(meta_files)


def next_id_str(next_id: int, width: int = 10) -> str:
    return str(next_id).zfill(width)


def organize_dataset(
    root: Path,
    start_id: int = 1,
    dry_run: bool = False,
) -> Dict[str, object]:
    raw_root = root / 'raw_data'
    out_real = root / 'audio' / 'real'
    out_fake = root / 'audio' / 'fake'
    ensure_dir(out_real)
    ensure_dir(out_fake)

    labels: List[Dict[str, object]] = []
    current_id = start_id

    for dirpath, _, filenames in os.walk(raw_root):
        for filename in filenames:
            if not filename.lower().endswith('.wav'):
                continue

            wav_path = Path(dirpath) / filename
            try:
                label, language = detect_label_and_language(wav_path.parts)
            except Exception:
                # Skip files outside expected structure
                continue

            model_or_speaker = detect_model_or_speaker(label, wav_path)
            meta_files = collect_metadata_files_for_wav(wav_path, label)

            id_str = next_id_str(current_id)
            out_dir = out_real if label == 'real' else out_fake
            out_path = out_dir / f'{id_str}.wav'

            if not dry_run:
                ensure_dir(out_dir)
                shutil.copy2(wav_path, out_path)

            labels.append({
                'id': id_str,
                'filename': f'{id_str}.wav',
                'original_path': str(wav_path.resolve()),
                'label': label,
                'language': language,
                'model_or_speaker': model_or_speaker,
                'metadata_files': meta_files,
            })

            current_id += 1

    summary: Dict[str, object] = {
        'count': len(labels),
        'start_id': start_id,
        'end_id': current_id - 1,
    }

    if not dry_run:
        labels_path = root / 'labels.json'
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Organize dataset_audios per DATASET_ORGANIZATION.md spec.')
    parser.add_argument(
        '--root',
        default='/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
        help='Root of dataset_audios directory',
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=1,
        help='Starting numeric id (default: 1)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and report without copying or writing labels.json',
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary = organize_dataset(root=root, start_id=args.start_id, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


