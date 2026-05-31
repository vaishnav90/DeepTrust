# Deepfake Speech Detection Dataset Description

## Overview

This dataset contains both **fake** (synthetic/AI-generated) and **real** (human) speech samples for deepfake speech detection research. The dataset is organized by language and TTS models.

**Total Dataset Size:** 154.65 GB  
**Total WAV Files:** 645,658  
**Total CSV Files:** 274  
**Total TTS Model Folders:** 163  
**Total Languages:** 17  
**Categories:** 2 (fake, real)

---

## Dataset Structure

```
dataset/
├── fake/          # AI-generated/synthetic speech
│   ├── de/        # German fake audio
│   ├── en/        # English fake audio
│   ├── es/        # Spanish fake audio
│   ├── fr/        # French fake audio
│   ├── it/        # Italian fake audio
│   ├── pl/        # Polish fake audio
│   ├── ru/        # Russian fake audio
│   └── uk/        # Ukrainian fake audio
│
└── real/          # Human/natural speech
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

---

## Category Breakdown

### Fake Speech (AI-Generated)

**Total Size:** 46.49 GB  
**Total WAV Files:** 140,000  
**Total CSV Files:** 151  
**Total TTS Models:** 151

| Language | Size    | Total Files | WAV Files | CSV Files | TTS Models |
|----------|---------|-------------|-----------|-----------|------------|
| **de** (German) | 5.26 GB | 17,017 | 17,000 | 17 | 17 |
| **en** (English) | 20.70 GB | 68,069 | 68,000 | 68 | 68 |
| **es** (Spanish) | 3.21 GB | 12,012 | 12,000 | 12 | 12 |
| **fr** (French) | 5.85 GB | 19,020 | 19,000 | 19 | 19 |
| **it** (Italian) | 3.94 GB | 15,016 | 15,000 | 15 | 15 |
| **pl** (Polish) | 2.71 GB | 8,008 | 8,000 | 8 | 8 |
| **ru** (Russian) | 2.45 GB | 7,007 | 7,000 | 7 | 7 |
| **uk** (Ukrainian) | 2.36 GB | 6,006 | 6,000 | 6 | 6 |

#### Sample TTS Models (English - de)

- Chatterbox Multilingual
- FishTTS
- Higgs-Audio-V2
- Kartoffelbox
- Llasa-1B-Multilingual
- NVidia Magpie-TTS Multilingual
- OuteTTS
- Resemble.ai
- And more...

#### Complete TTS Models List (English - en)

The English fake dataset includes the following TTS models (68 total):

- **Chatterbox** - Chatterbox TTS system
- **e2-tts** - End-to-end TTS
- **f5-tts** - F5 TTS system
- **facebook_mms-tts-eng** - Meta's MMS TTS (English)
- **FireRedTTS-2.0** - FireRed TTS v2.0
- **FishTTS** - FishTTS system
- **griffin_lim** - Griffin-Lim vocoder
- **Higgs-Audio-V2** - Higgs Audio v2
- **Index-TTS-1.5** - Index TTS v1.5
- **Index-TTS-2.0** - Index TTS v2.0
- **Indri-TTS-0.1** - Indri TTS v0.1
- **Kitten-TTS-Nano-0.1** - Kitten TTS Nano v0.1
- **Kitten-TTS-Nano-0.2** - Kitten TTS Nano v0.2
- **kokoro** - Kokoro TTS
- **Kyutai-TTS** - Kyutai TTS
- **Llasa-1B** - Llasa 1B model
- **Llasa-1B-Multilingual** - Llasa 1B Multilingual
- **Llasa-3B** - Llasa 3B model
- **Llasa-8B** - Llasa 8B model
- **Mars5** - Mars5 TTS
- **MatchaTTS** - MatchaTTS
- **MegaTTS3** - MegaTTS v3
- **MeloTTS** - MeloTTS
- **Metavoice-1B** - Metavoice 1B
- **Microsoft VibeVoice 1.5B** - Microsoft VibeVoice 1.5B
- **Microsoft VibeVoice Large** - Microsoft VibeVoice Large
- **microsoft_speecht5_tts** - Microsoft SpeechT5
- **MiniCPM-o-2.6** - MiniCPM o-2.6
- **Nari Dia-1.6B** - Nari Dia 1.6B
- **Openaudio-S1-Mini** - Openaudio S1 Mini
- **OpenVoiceV2** - OpenVoice v2
- **optispeech** - Optispeech
- **orpheus-tts-0.1-finetune** - Orpheus TTS v0.1
- **OuteTTS** - OuteTTS
- **parler_tts_large_v1** - Parler TTS Large v1
- **parler_tts_mini_v0.1** - Parler TTS Mini v0.1
- **parler_tts_mini_v1** - Parler TTS Mini v1
- **Qwen2.5-Omni** - Qwen2.5 Omni
- **Resemble.ai** - Resemble.ai
- **sesame_csm** - Sesame CSM
- **Spark-TTS-0.5B** - Spark TTS 0.5B
- **suno_bark** - Suno Bark
- **suno_bark-small** - Suno Bark Small
- **tts_models_en_ljspeech_vits** - Coqui TTS LJSpeech VITS
- **tts_models_en_ljspeech_tacotron2-DCA** - Coqui TTS Tacotron2 DCA
- **tts_models_en_ljspeech_glow-tts** - Coqui TTS Glow TTS
- **Veena** - Veena TTS
- **vixTTS** - vixTTS
- **VoxCPM-0.5B** - VoxCPM 0.5B
- **Voxtream** - Voxtream
- **WhisperSpeech** - WhisperSpeech
- **ZipVoice** - ZipVoice
- **zonosTTS-v0.1** - ZonosTTS v0.1
- And more...

### Real Speech (Human/Natural)

**Total Size:** 108.16 GB  
**Total WAV Files:** 505,658  
**Total CSV Files:** 123  
**Total TTS Model Folders:** 12

| Language | Size    | Total Files | WAV Files | CSV Files | TTS Models |
|----------|---------|-------------|-----------|-----------|------------|
| **de_DE** (German) | 26.64 GB | 118,563 | 118,528 | 35 | 1 |
| **en_UK** (UK English) | 4.90 GB | 23,561 | 23,559 | 2 | 1 |
| **en_US** (US English) | 10.55 GB | 46,308 | 46,294 | 14 | 1 |
| **es_ES** (Spanish) | 11.68 GB | 59,309 | 59,297 | 12 | 1 |
| **fr_FR** (French) | 20.50 GB | 90,363 | 90,348 | 15 | 3 |
| **it_IT** (Italian) | 13.77 GB | 73,425 | 73,405 | 20 | 1 |
| **pl_PL** (Polish) | 5.79 GB | 26,324 | 26,322 | 2 | 1 |
| **ru_RU** (Russian) | 5.05 GB | 20,503 | 20,495 | 8 | 1 |
| **uk_UK** (Ukrainian) | 9.27 GB | 35,424 | 35,410 | 14 | 1 |

#### Sample Dataset Sources (Real Speech)

Real speech datasets typically come from:
- **by_book** - Audio book recordings
- **CommonVoice** - Mozilla CommonVoice dataset
- **LibriSpeech** - Read speech corpus
- **Mozilla TTS** - Mozilla Text-to-Speech dataset
- And other natural speech corpora

---

## File Organization

Each TTS model folder typically contains:
- Multiple `.wav` audio files (usually 1000+ files per model)
- One `.csv` metadata file with transcriptions and metadata
- Some folders may contain `.bak` backup files

### Example Structure

```
dataset/fake/en/Llasa-8B/
├── audio_001.wav
├── audio_002.wav
├── ...
├── audio_1000.wav
└── metadata.csv

dataset/real/fr_FR/mix/
├── common_voice_fr_XXX/
├── sous_les_mers/
├── info.txt
└── metadata_mls.json
```

---

## Statistics Summary

### By Language Pair (Fake + Real)

| Language | Fake Files | Real Files | Total Files | Ratio (Real:Fake) |
|----------|------------|------------|-------------|-------------------|
| **German** | 17,000 | 118,528 | 135,528 | 7.0:1 |
| **English** | 68,000 | 69,853 | 137,853 | 1.0:1 |
| **Spanish** | 12,000 | 59,297 | 71,297 | 4.9:1 |
| **French** | 19,000 | 90,348 | 109,348 | 4.8:1 |
| **Italian** | 15,000 | 73,405 | 88,405 | 4.9:1 |
| **Polish** | 8,000 | 26,322 | 34,322 | 3.3:1 |
| **Russian** | 7,000 | 20,495 | 27,495 | 2.9:1 |
| **Ukrainian** | 6,000 | 35,410 | 41,410 | 5.9:1 |

### Overall Statistics

- **Total Languages:** 17 (8 for fake, 9 for real)
- **Most Common Language:** English (English + UK English + US English)
- **Largest Category:** Real speech (108.16 GB vs 46.49 GB fake)
- **Most Diverse Fake Dataset:** English (68 TTS models)
- **Most Balanced Dataset:** French (relatively balanced fake/real ratio)

---

## Usage Notes

1. **File Format:** All audio files are in WAV format
2. **Metadata:** CSV files contain transcriptions and metadata for each audio file
3. **Balance:** The dataset has more real speech samples than fake ones (3.6:1 ratio overall)
4. **Quality:** Dataset includes a diverse range of TTS models from various providers and research projects

