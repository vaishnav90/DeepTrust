"""
MultilingualDataset - Loads the new organized multilingual deepfake speech detection dataset.

This dataset loads from labels.json which contains metadata about audio files.
Audio files are expected to be in:
- Flat structure (preferred): dataset_audios/audio/{filename}.wav
- Organized structure (fallback): dataset_audios/audio/real/ and dataset_audios/audio/fake/
"""

import os
import json
from pathlib import Path
from typing import List, Optional

from ._dataclass import DF_Item


class MultilingualDataset:
    """
    Dataset loader for the multilingual deepfake speech detection dataset.
    
    This dataset loads audio files from:
    - Preferred: Flat structure at dataset_audios/audio/{filename}.wav
    - Fallback: Organized structure at dataset_audios/audio/real/ and dataset_audios/audio/fake/
    - labels.json contains metadata for all audio files
    
    Note: The dataset will NOT use original_path from the JSON file to avoid
    accessing files from their original location.
    """
    
    def __init__(
        self,
        labels_path: str,
        dataset_root: Optional[str] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        selected_language: Optional[str] = None,
        selected_fake_model: Optional[str] = None,
        exclude_fake_models: Optional[List[str]] = None,
        print_info: bool = False
    ):
        """
        Initialize the MultilingualDataset.
        
        Args:
            labels_path: Path to labels.json file
            dataset_root: Root directory of the dataset (default: parent of labels.json)
            train_split: Fraction of data for training (default: 0.8)
            val_split: Fraction of data for validation (default: 0.1)
            test_split: Fraction of data for testing (default: 0.1)
            random_seed: Random seed for splitting (default: 42)
            selected_language: Filter by language code (e.g., 'en', 'it'). If None, uses all languages (default: None)
            selected_fake_model: If set, keep ALL real samples, and keep ONLY fake samples where
                entry["model_or_speaker"] == selected_fake_model. Can be combined with selected_language. (default: None)
            exclude_fake_models: If set, keep ALL real samples, and EXCLUDE fake samples where
                entry["model_or_speaker"] is in exclude_fake_models. Mutually exclusive with selected_fake_model. (default: None)
            print_info: Whether to print dataset information (default: False)
        """
        import random
        import numpy as np
        from math import isclose

        # ---------------------------
        # Sanity checks (fail fast)
        # ---------------------------
        for name, v in (("train_split", train_split), ("val_split", val_split), ("test_split", test_split)):
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float, got {type(v).__name__}")
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0.0, got {v}")
            if v > 1.0:
                raise ValueError(f"{name} must be <= 1.0, got {v}")

        split_sum = float(train_split) + float(val_split) + float(test_split)
        # Use a slightly forgiving tolerance for float config values.
        if not isclose(split_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "train/val/test splits must sum to 1.0, "
                f"got train_split={train_split}, val_split={val_split}, test_split={test_split} (sum={split_sum})"
            )
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load labels.json
        labels_path = Path(labels_path).resolve()
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            all_labels = json.load(f)
        
        # Filter by language if selected_language is specified
        if selected_language is not None:
            self.labels = [entry for entry in all_labels if entry.get('language') == selected_language]
            if len(self.labels) == 0:
                raise ValueError(
                    f"No entries found in labels.json for selected_language={selected_language!r}. "
                    "Either set selected_language=None or verify your labels.json language codes."
                )
            if print_info:
                print(f"Filtered dataset by language: '{selected_language}'")
                print(f"Total entries before filtering: {len(all_labels)}")
                print(f"Total entries after filtering: {len(self.labels)}")
        else:
            self.labels = all_labels

        # Normalize exclude_fake_models (strip, drop empties)
        exclude_set = None
        if exclude_fake_models is not None:
            if isinstance(exclude_fake_models, str):
                exclude_fake_models = [exclude_fake_models]
            exclude_norm = []
            for x in list(exclude_fake_models):
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    exclude_norm.append(s)
            exclude_set = set(exclude_norm) if exclude_norm else None

        if selected_fake_model is not None and exclude_set is not None:
            raise ValueError("selected_fake_model and exclude_fake_models are mutually exclusive; set only one.")

        # Filter by fake model if selected_fake_model is specified:
        # - Keep ALL real entries
        # - Keep ONLY fake entries matching model_or_speaker
        if selected_fake_model is not None:
            before_n = len(self.labels)
            selected_fake_model_norm = str(selected_fake_model).strip()
            if selected_fake_model_norm == "":
                selected_fake_model_norm = None
            if selected_fake_model_norm is not None:
                filtered = []
                for entry in self.labels:
                    label_str = str(entry.get("label", "real")).strip().lower()
                    if label_str == "fake":
                        if str(entry.get("model_or_speaker", "")).strip() == selected_fake_model_norm:
                            filtered.append(entry)
                    else:
                        # For reals, do not filter by model_or_speaker (it's a speaker there)
                        filtered.append(entry)
                self.labels = filtered
                if print_info:
                    print(f"Filtered fake samples by model_or_speaker: '{selected_fake_model_norm}' (kept all real samples)")
                    print(f"Total entries before fake-model filtering: {before_n}")
                    print(f"Total entries after fake-model filtering:  {len(self.labels)}")

        # Exclude fake models if exclude_fake_models is specified:
        # - Keep ALL real entries
        # - Keep ONLY fake entries where model_or_speaker NOT IN exclude_set
        if exclude_set is not None:
            before_n = len(self.labels)
            filtered = []
            for entry in self.labels:
                label_str = str(entry.get("label", "real")).strip().lower()
                if label_str == "fake":
                    mos = str(entry.get("model_or_speaker", "")).strip()
                    if mos in exclude_set:
                        continue
                    filtered.append(entry)
                else:
                    filtered.append(entry)
            self.labels = filtered
            if print_info:
                excl_disp = ", ".join(sorted(exclude_set))
                print(f"Excluded fake samples by model_or_speaker in: [{excl_disp}] (kept all real samples)")
                print(f"Total entries before fake-model exclusion: {before_n}")
                print(f"Total entries after fake-model exclusion:  {len(self.labels)}")
        
        # Determine dataset root
        if dataset_root is None:
            dataset_root = labels_path.parent
        else:
            dataset_root = Path(dataset_root).resolve()
        
        self.dataset_root = dataset_root
        
        # Initialize sets
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.class_weight = []
        
        # Convert labels to DF_Item objects
        items = []
        
        for entry in self.labels:
            # Get audio file path
            # Priority order:
            # 1. Flat structure: audio/{filename}.wav (preferred)
            # 2. Organized structure: audio/real/{filename}.wav or audio/fake/{filename}.wav
            # 3. Skip if neither exists (don't use original_path)
            label_str = entry.get('label', 'real')
            filename = entry.get('filename', '')
            
            if not filename:
                # Skip entries without filename
                continue
            
            # Try flat structure first: audio/{filename}.wav
            audio_path = self.dataset_root / 'audio' / filename
            
            # If flat structure doesn't exist, try organized structure
            if not audio_path.exists():
                if label_str == 'real':
                    audio_path = self.dataset_root / 'audio' / 'real' / filename
                else:
                    audio_path = self.dataset_root / 'audio' / 'fake' / filename
            
            # Skip if file doesn't exist (don't fallback to original_path)
            if not audio_path.exists():
                continue
            
            # Convert label: 'real' -> 0, 'fake' -> 1
            label = 0 if label_str == 'real' else 1
            
            # Get model/speaker as attack_type
            attack_type = entry.get('model_or_speaker', 'unknown')
            if label == 0:
                attack_type = 'real'  # For real samples, use 'real' as attack_type
            
            # Create DF_Item
            item = DF_Item(
                path=str(audio_path.resolve()),
                label=label,
                attack_type=attack_type,
                is_fake=(label == 1)
            )
            items.append(item)
        
        if len(items) == 0:
            lang_msg = f" for selected_language={selected_language!r}" if selected_language is not None else ""
            raise ValueError(
                f"After resolving audio paths, no usable audio files were found{lang_msg}. "
                "Check that `dataset_root` is correct and audio files exist under "
                "`<dataset_root>/audio/` (flat) or `<dataset_root>/audio/{real,fake}/` (organized)."
            )

        # Shuffle items for train/val/test split
        random.shuffle(items)
        
        # Split dataset
        total = len(items)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        self.train_set = items[:train_end]
        self.val_set = items[train_end:val_end]
        self.test_set = items[val_end:]

        # Explicit split non-emptiness checks.
        # NOTE: Because we use integer truncation, small totals can yield empty val/test even if split > 0.
        # We prefer failing fast with a clear message instead of silently training/evaluating on empty sets.
        # Allow empty splits if and only if the corresponding split fraction is 0.
        # This enables "test-only" runs such as train_split=0, val_split=0, test_split=1.
        if train_split > 0.0 and len(self.train_set) == 0:
            raise ValueError(
                f"Train split is empty (total_items={total}, train_split={train_split}). "
                "Increase data size, adjust splits, or set selected_language=None."
            )
        if val_split > 0.0 and len(self.val_set) == 0:
            raise ValueError(
                f"Validation split is empty (total_items={total}, val_split={val_split}). "
                "This often happens when the selected language is sparse. "
                "Increase data size, adjust splits, or set selected_language=None."
            )
        if test_split > 0.0 and len(self.test_set) == 0:
            raise ValueError(
                f"Test split is empty (total_items={total}, test_split={test_split}). "
                "This often happens when the selected language is sparse. "
                "Increase data size, adjust splits, or set selected_language=None."
            )
        
        # Calculate class weights for balancing
        train_num_neg = sum(1 for item in self.train_set if item.label == 0)
        train_num_pos = sum(1 for item in self.train_set if item.label == 1)
        if train_num_neg > 0 and train_num_pos > 0:
            total_count = train_num_neg + train_num_pos
            self.class_weight.append(total_count / train_num_neg)  # weight for real (label=0)
            self.class_weight.append(total_count / train_num_pos)  # weight for fake (label=1)
        else:
            self.class_weight = [1.0, 1.0]
        
        # Print info if requested
        if print_info:
            train_real = sum(1 for item in self.train_set if item.label == 0)
            train_fake = sum(1 for item in self.train_set if item.label == 1)
            val_real = sum(1 for item in self.val_set if item.label == 0)
            val_fake = sum(1 for item in self.val_set if item.label == 1)
            test_real = sum(1 for item in self.test_set if item.label == 0)
            test_fake = sum(1 for item in self.test_set if item.label == 1)
            
            language_info = f"Language: {selected_language}" if selected_language else "Language: All languages"
            model_info = f"Fake model filter: {selected_fake_model}" if selected_fake_model else "Fake model filter: (none)"
            if exclude_set is not None:
                model_info += f" | Exclude fake models: {', '.join(sorted(exclude_set))}"
            info = (
                f'====================\n'
                + f'  MultilingualDataset\n'
                + f'{language_info}\n'
                + f'{model_info}\n'
                + f'====================\n'
                + f'TRAIN: Real={train_real:,}, Fake={train_fake:,}, Total={len(self.train_set):,}\n'
                + f'VAL:   Real={val_real:,}, Fake={val_fake:,}, Total={len(self.val_set):,}\n'
                + f'TEST:  Real={test_real:,}, Fake={test_fake:,}, Total={len(self.test_set):,}\n'
                + f'Class weights: Real={self.class_weight[0]:.4f}, Fake={self.class_weight[1]:.4f}\n'
                + f'====================\n'
            )
            print(info)

