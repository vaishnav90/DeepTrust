## 🧩 **Dataset Organization Script — Overview**

This script automates the process of restructuring the `dataset_audios` directory to create a unified and consistent dataset layout.

---

### ⚙️ **Objective**

Implement a Python script that:

1. **Scans recursively** through all subdirectories inside
   `dataset_audios/raw_data/`.

2. **Copy** every `.wav` audio file into one of next two output folders:

   ```
   dataset_audios/audio/
     ├── real/
     └── fake/
   ```

3. **Renames** each audio sequentially using zero-padded IDs (`00001.wav`, `00002.wav`, …).

   * The index increases **continuously across both folders**.
   * Example: if the last real audio is `423042.wav`, the first fake audio becomes `423043.wav`.

4. **Keeps all non-audio files** (metadata, transcripts, `.csv`, `.json`, `.txt`, etc.) **in their original locations** within `raw_data/`.
   These files are not moved or modified.

5. **Generates a single JSON file** at:

   ```
   dataset_audios/labels.json
   ```

   that contains detailed metadata for every audio file.

---

### 🗂️ **Output Directory Structure**

```
dataset_audios/
├── audio/
│   ├── real/
│   │   ├── 00001.wav
│   │   ├── 00002.wav
│   │   └── ...
│   └── fake/
│       ├── 423043.wav
│       ├── 423044.wav
│       └── ...
├── labels.json
└── raw_data/
    ├── real/
    ├── fake/
    └── ...
```

---

### 🧠 **Metadata Structure (`labels.json`)**

Each audio file is represented by an entry like this:

```json
{
  "id": "00001",
  "filename": "00001.wav",
  "original_path": "/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/raw_data/real/en_US/by_book/female/judy_bieber/ozma_of_oz/wavs/file001.wav",
  "label": "real",
  "language": "en",
  "model_or_speaker": "judy_bieber",
  "metadata_files": [
    "/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/raw_data/real/en_US/by_book/female/judy_bieber/ozma_of_oz/metadata.csv",
    "/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/raw_data/real/en_US/by_book/female/judy_bieber/ozma_of_oz/metadata_mls.json"
  ]
}
```

---

### 🏷️ **Field Descriptions**

| Field              | Description                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | Sequential numeric identifier (zero-padded).                                                                                                   |
| `filename`         | New name of the `.wav` file in `/audio/real/` or `/audio/fake/`.                                                                               |
| `original_path`    | Full original path of the `.wav` file before being moved.                                                                                      |
| `label`            | `"real"` or `"fake"`, determined automatically by parent folder name.                                                                          |
| `language`         | **Normalized language code** (e.g., `en_US`, `en_UK`, `en_EN` → `en`; `de_DE` → `de`; `fr_FR` → `fr`).                                         |
| `model_or_speaker` | For fake samples: the TTS model name (e.g., `tts_models_en_ljspeech_vits`); for real samples: the speaker/narrator name (e.g., `judy_bieber`). |
| `metadata_files`   | List of associated metadata files (e.g., `meta.csv`, `metadata_mls.json`, etc.) found in the same directory as the `.wav`.                     |

---

### 🧩 **Key Script Behaviors**

* **Label detection:** Based on parent folder (`fake` or `real`).
* **Language normalization:**
  The script extracts language codes from the directory path and converts all variants to a simplified, unified code.
  Example mapping:

  ```
  en_US, en_UK         →  en
  de_DE                →  de
  es_ES                →  es
  fr_FR                →  fr
  it_IT                →  it
  pl_PL                →  pl
  ru_RU                →  ru
  uk_UK                →  uk
  ```
* **Model/Speaker detection:**

  * For fake audios → identifies the TTS model name from origin folder (contains "tts", "voice", "bark", "resemble", etc.).
  * For real audios → extracts the speaker name from the directory (typically after `/female/`, `/male/`, or `/mix/`).
* **Audio preservation:** Metadata and audio files remain in their original locations.
* **Copy:** By default, the script copy `.wav` files.