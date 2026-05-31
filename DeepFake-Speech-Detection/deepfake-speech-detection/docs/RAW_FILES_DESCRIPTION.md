## 📂 Dataset Structure Overview

**Root Path:**
`/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios`

### 📁 Main Folders

| Folder      | Description                                                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |                                |
| `raw_data/` | Original dataset containing all unprocessed recordings, separated into **fake** (synthetic) and **real** (authentic) speech samples.   |

---

## 🧠 `raw_data/fake/` — Synthetic (Generated) Speech

This folder contains **AI-generated or TTS (Text-To-Speech)** audios organized by language.
Each language folder (e.g. `en`, `de`, `es`, `fr`, `it`, `pl`, `ru`, `uk`) includes **multiple subfolders**, one for each **TTS model or generator**.

### 🔤 Languages

Each of the following folders represents a language:

```
de, en, es, fr, it, pl, ru, uk
```

Inside each language directory:

* Every TTS model (like `tts_models_en_ljspeech_glow-tts`, `FishTTS`, `Resemble.ai`, etc.) has:

  * One `meta.csv` file describing the audio clips (transcripts, durations, etc.)
  * About **1000 `.wav` files**
* Some models also include backup metadata files like `meta.csv.bak` or hidden files like `.meta.csv.swp`.

#### Example structure:

```
raw_data/fake/en/
 ├── FishTTS/
 │   ├── meta.csv
 │   └── 1000 .wav files
 ├── Kyutai-TTS/
 │   ├── meta.csv
 │   └── 1000 .wav files
 ├── tts_models_en_ljspeech_glow-tts/
 │   ├── meta.csv
 │   └── 1000 .wav files
 └── Resemble.ai (April 12th, 2025)/
     ├── meta.csv
     └── 1000 .wav files
```

> ✅ In total, each language folder contains many different TTS models, each producing about 1000 audio clips — resulting in **hundreds of thousands of synthetic samples** across all languages.

---

## 🎙️ `raw_data/real/` — Authentic Human Speech

This directory contains **real recorded voices**, mostly from audiobook or speech-reading datasets.
It’s organized **by language variant**, then **by speaker gender and name**, and then by **book or project title**.

### 🌍 Language folders

Languages use extended locale codes:

```
de_DE, en_UK, en_US, es_ES, fr_FR, it_IT, pl_PL, ru_RU, uk_UK
```

Each language folder contains a `by_book/` directory, inside which the data is divided by:

* **Gender**: `female/`, `male/`, and sometimes `mix/`
* **Speaker name**: the narrator (e.g., `judy_bieber`, `ramona_deininger`, `elizabeth_klett`)
* **Book title**: each speaker has multiple books or stories as subfolders

Each **book folder** contains:

* Metadata files (`metadata.csv`, `metadata_mls.json`, and sometimes custom text metadata files)
* A `wavs/` directory with all the actual audio samples (`.wav`)
* Occasionally `info.txt` files describing the recording

#### Example structure:

```
raw_data/real/en_US/
 └── by_book/
     ├── female/
     │   └── judy_bieber/
     │       ├── ozma_of_oz/
     │       │   ├── metadata.csv
     │       │   ├── metadata_mls.json
     │       │   └── wavs/  (1863 audio files)
     │       └── sky_island/
     │           ├── metadata.csv
     │           ├── metadata_mls.json
     │           └── wavs/  (2335 audio files)
     └── male/
         └── elliot_miller/
             ├── pink_fairy_book/
             │   ├── metadata.csv
             │   ├── metadata_mls.json
             │   └── wavs/  (4238 audio files)
             └── poisoned_pen/
                 ├── metadata.csv
                 ├── metadata_mls.json
                 └── wavs/  (4432 audio files)
```

> ✅ Each book is a complete set of real recordings, with tens of hundreds of `.wav`s, plus metadata describing chapters, segments, and transcripts.

---

## 🔡 Summary of Key Characteristics

| Aspect               | Fake Data                            | Real Data                                                       |
| -------------------- | ------------------------------------ | --------------------------------------------------------------- |
| **Source**           | AI-generated / TTS                   | Human voice recordings                                          |
| **Languages**        | `en, de, es, fr, it, pl, ru, uk`     | `en_US, en_UK, de_DE, es_ES, fr_FR, it_IT, pl_PL, ru_RU, uk_UK` |
| **Organization**     | Per language → per model             | Per language → by_book → gender → speaker → book                |
| **Files per folder** | ~1000 `.wav` per TTS model           | Hundreds to tens of thousands per book                          |
| **Metadata**         | `meta.csv` per TTS model             | `metadata.csv`, `metadata_mls.json`, `info.txt` per book        |
| **Use case**         | Deepfake / synthetic voice detection | Real speech reference and training                              |

---

### 🧩 Notes

* Some folders include backups or temp files (`meta.csv.bak`, `._metadata_mls.json`).
* Languages are later **normalized** (e.g. `en_US`, `en_UK`, `en_EN` → `en`) during processing.
* All `.wav` files will be moved to `audio/real/` or `audio/fake/` while metadata stays in place.