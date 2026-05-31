# Deepfake Speech Detection

This repository contains experiments for deepfake voice detection. Due to the recent increase in generative models capable of producing very realistic synthetic audio, we seek to develop a model capable of detecting this new type of synthetic audio.

## Dataset Overview

The project uses a curated dataset composed of:
- **Synthetic Speech (Fake)**: AI-generated voice samples from 163 TTS models across 8 languages
- **Natural Speech (Real)**: Human voice samples from various speech corpora across 9 languages

### Source Datasets

The curated dataset was built from:
- **MLAAD Dataset** ([Hugging Face](https://huggingface.co/datasets/mueller91/MLAAD)): Source of synthetic voice samples for training deepfake detection models
- **M-AILABS Dataset** ([GitHub](https://github.com/imdatceleste/m-ailabs-dataset)): Source of real voice samples to complement the synthetic data

### Dataset Structure

```
dataset/
├── fake/          # AI-generated/synthetic speech (46.49 GB, 140,000 files)
│   ├── de/        # German - 17 TTS models
│   ├── en/        # English - 68 TTS models
│   ├── es/        # Spanish - 12 TTS models
│   ├── fr/        # French - 19 TTS models
│   ├── it/        # Italian - 15 TTS models
│   ├── pl/        # Polish - 8 TTS models
│   ├── ru/        # Russian - 7 TTS models
│   └── uk/        # Ukrainian - 6 TTS models
└── real/          # Human/natural speech (108.16 GB, 505,658 files)
    ├── de_DE/     # German real audio
    ├── en_UK/     # UK English real audio
    ├── en_US/     # US English real audio
    ├── es_ES/     # Spanish real audio
    ├── fr_FR/     # French real audio
    ├── it_IT/     # Italian real audio
    ├── pl_PL/     # Polish real audio
    ├── ru_RU/     # Russian real audio
    └── uk_UK/     # Ukrainian real audio
```

**Total Dataset:** 154.65 GB with 645,658 WAV files

See [DATASET_DESCRIPTION.md](DATASET_DESCRIPTION.md) for detailed statistics and TTS model information.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For MLAAD dataset access (if re-downloading), you may need to authenticate with Hugging Face:
```bash
huggingface-cli login
```

## Project Structure

- `dataset/` - Curated dataset with fake and real speech samples
- `explore-folders.py` - Script to analyze and explore the dataset structure
- `requirements.txt` - Python dependencies
- `DATASET_DESCRIPTION.md` - Detailed dataset documentation

## Usage

### Explore the Dataset

Run the exploration script to analyze the dataset structure:

```bash
python explore-folders.py
```

Or specify a custom path:

```bash
python explore-folders.py path/to/dataset
```

This will display:
- Dataset size and statistics
- Language breakdown
- TTS model counts
- File type distribution
