# Vietnamese speech data analysis and preprocessing

This repository independently analyzes and preprocesses the Vietnamese
configuration of Google FLEURS (`vi_vn`). It contains its own dataset loader,
audio and transcript preprocessing helpers, and visualization workflow.

## Setup

Python 3.11 is pinned in `.python-version`.

```bash
uv sync
```

Hugging Face downloads are cached in `~/data/fleurs/` by default. Set
`SPEECH_CACHE_DIR` to use another cache root. The model repository has its own
copies of the small loader and preprocessing helpers, so this project has no
runtime dependency on the model repository.

## Analyze FLEURS

```bash
uv run speech-analyze
```

The default analyzes the first 300 rows of each split and saves plots plus a
JSON summary under `outputs/`. To analyze every row:

```bash
uv run speech-analyze --max-per-split 0
```

The current overview figures and summary are included in `outputs/`.

## Preprocessing

`preprocessing.py` provides reusable helpers to convert audio to finite mono
16 kHz float samples and normalize transcripts to Unicode NFC with collapsed
whitespace. Vietnamese diacritics and punctuation are kept in labels. Metric
normalization is a separate helper that lowercases and strips punctuation.

The dataset is [Google FLEURS](https://huggingface.co/datasets/google/fleurs),
configuration `vi_vn`, licensed CC-BY 4.0. Retain attribution when sharing
derived work.
