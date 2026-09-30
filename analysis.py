"""Exploratory analysis and plots for the Vietnamese FLEURS dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cache import configure_cache
from data import SAMPLE_RATE, SPLITS, load_fleurs
from preprocessing import audio_to_mono_16k, normalize_transcript

configure_cache()


def analyze_dataset(
    *, max_per_split: int = 300, output_dir: Path, cache_dir: str | None = None
) -> dict[str, Any]:
    dataset = load_fleurs(streaming=False, cache_dir=cache_dir)
    report: dict[str, Any] = {
        "dataset": "google/fleurs",
        "config": "vi_vn",
        "sample_rate_hz": SAMPLE_RATE,
        "max_per_split": max_per_split,
        "duration_note": "Summaries use sampled rows when max_per_split is nonzero.",
        "splits": {},
    }
    durations: dict[str, list[float]] = {}
    transcript_lengths: dict[str, list[int]] = {}
    token_counts: Counter[str] = Counter()
    example_audio: np.ndarray | None = None

    for split_name in SPLITS:
        split_durations: list[float] = []
        split_text_lengths: list[int] = []
        for row_count, row in enumerate(dataset[split_name]):
            if max_per_split > 0 and row_count >= max_per_split:
                break
            text = normalize_transcript(row.get("transcription", ""))
            waveform = audio_to_mono_16k(row["audio"])
            split_durations.append(len(waveform) / SAMPLE_RATE)
            split_text_lengths.append(len(text))
            token_counts.update(token.casefold() for token in text.split())
            if example_audio is None and split_name == "train":
                example_audio = waveform

        durations[split_name] = split_durations
        transcript_lengths[split_name] = split_text_lengths
        count = len(split_durations)
        report["splits"][split_name] = {
            "rows_analyzed": count,
            "duration_seconds": {
                "mean": float(np.mean(split_durations)) if count else None,
                "median": float(np.median(split_durations)) if count else None,
                "min": float(np.min(split_durations)) if count else None,
                "max": float(np.max(split_durations)) if count else None,
            },
            "transcript_characters": {
                "mean": float(np.mean(split_text_lengths)) if count else None,
                "median": float(np.median(split_text_lengths)) if count else None,
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_overview(durations, transcript_lengths, output_dir / "dataset_overview.png")
    _plot_tokens(token_counts, output_dir / "top_tokens.png")
    if example_audio is not None:
        _plot_example(example_audio, output_dir / "sample_waveform_spectrogram.png")
    return report


def _plot_overview(
    durations: dict[str, list[float]],
    transcript_lengths: dict[str, list[int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    labels = list(durations)
    axes[0].bar(labels, [len(durations[name]) for name in labels], color="#4f79a7")
    axes[0].set_title("Rows analyzed by split")
    axes[0].set_ylabel("Examples")
    axes[0].grid(axis="y", alpha=0.2)

    duration_values = [durations[name] for name in labels if durations[name]]
    duration_labels = [name for name in labels if durations[name]]
    if duration_values:
        axes[1].boxplot(duration_values, tick_labels=duration_labels, showfliers=False)
    axes[1].set_title("Audio duration")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.2)

    text_values = [transcript_lengths[name] for name in labels if transcript_lengths[name]]
    text_labels = [name for name in labels if transcript_lengths[name]]
    if text_values:
        axes[2].boxplot(text_values, tick_labels=text_labels, showfliers=False)
    axes[2].set_title("Transcript length")
    axes[2].set_ylabel("Characters")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("FLEURS Vietnamese dataset overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tokens(counts: Counter[str], output_path: Path, limit: int = 25) -> None:
    most_common = counts.most_common(limit)
    fig, ax = plt.subplots(figsize=(11, 6))
    if most_common:
        tokens, frequencies = zip(*most_common)
        positions = np.arange(len(tokens))
        ax.barh(positions, frequencies, color="#59a14f")
        ax.set_yticks(positions, labels=tokens)
        ax.invert_yaxis()
    ax.set_title("Most common whitespace tokens (Vietnamese syllables)")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_example(waveform: np.ndarray, output_path: Path) -> None:
    seconds = np.arange(len(waveform)) / SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(seconds, waveform, linewidth=0.45, color="#4e79a7")
    axes[0].set_title("Example training clip waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)
    axes[1].specgram(waveform, NFFT=512, Fs=SAMPLE_RATE, noverlap=384, cmap="magma")
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, 8_000)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-split", type=int, default=300,
        help="Rows to inspect per split; use 0 to inspect the full dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()
    report = analyze_dataset(
        max_per_split=args.max_per_split,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved plots and summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
