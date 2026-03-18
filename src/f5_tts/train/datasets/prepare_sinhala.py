"""
Sinhala dataset prep for F5-TTS Option A (full finetune, new vocab).
Run from repo root:
    python src/f5_tts/train/datasets/prepare_sinhala.py
"""
import csv
import json
import os
import sys
import wave as wave_module  # built-in, always available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from pathlib import Path

from datasets.arrow_writer import ArrowWriter
from datasets import Features, Value, Sequence
from tqdm import tqdm


from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────
DATASET_NAME  = os.getenv("DATASET_NAME", "Sinhala_char_custom")
MIN_DURATION  = float(os.getenv("MIN_DURATION", 0.4))
MAX_DURATION  = float(os.getenv("MAX_DURATION", 30.0))
SAVE_DIR      = str(Path(os.getenv("SAVE_DIR_BASE", "data")) / DATASET_NAME)
DATASET_ROOT  = os.getenv("DATASET_ROOT", "2/amalsha-voice-dataset")
# ───────────────────────────────────────────────────────────────────────


def get_duration(wav_path: str) -> float:
    """
    Return audio duration in seconds using the most stable method available.

    Priority order:
      1. Python built-in `wave` module  — zero extra deps, works for all PCM WAV files
      2. soundfile                       — handles more codecs (MP3, FLAC, OGG …)
      3. torchaudio.load()              — always present regardless of torchaudio version;
                                          slightly slower (loads full waveform) but reliable
    `torchaudio.info()` is intentionally avoided because it was only stabilised
    in torchaudio ≥ 0.9 and is absent in many conda/pip environments.
    """
    # ── Method 1: stdlib wave (fastest, no deps) ────────────────────────
    try:
        with wave_module.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        pass

    # ── Method 2: soundfile ─────────────────────────────────────────────
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        pass

    # ── Method 3: torchaudio.load() (version-agnostic) ──────────────────
    import torchaudio
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform.shape[1] / sample_rate


def main():
    # 1. Use local dataset
    print("Using local dataset...")
    dataset_root = Path(DATASET_ROOT).resolve()
    print(f"Dataset root: {dataset_root}")

    # 2. Locate metadata.csv — adjust if the file is nested
    meta_file = dataset_root / "metadata.csv"
    assert meta_file.exists(), f"metadata.csv not found in {dataset_root}"
    print(f"Using metadata: {meta_file}")

    # 3. Parse CSV and build records
    result        = []
    duration_list = []
    vocab_set     = set()
    skipped       = 0

    with open(meta_file, encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        assert header[0].strip() == "audio_file" and header[1].strip() == "text", \
            f"Unexpected header: {header}"

        for row in tqdm(reader, desc="Processing rows"):
            if len(row) < 2:
                continue
            rel_path, text = row[0].strip(), row[1].strip()

            # Resolve to absolute path
            wav_path = (dataset_root / rel_path).resolve()
            if not wav_path.exists():
                print(f"  Missing: {wav_path}, skipping")
                skipped += 1
                continue

            try:
                dur = get_duration(str(wav_path))
            except Exception as e:
                print(f"  Duration error {wav_path}: {e}, skipping")
                skipped += 1
                continue

            if not (MIN_DURATION <= dur <= MAX_DURATION):
                skipped += 1
                continue

            result.append({
                "audio_path": str(wav_path),
                "text":       text,
                "duration":   dur,
            })
            duration_list.append(dur)
            vocab_set.update(list(text))   # codepoint-level, correct for Sinhala

    print(f"\nProcessed: {len(result)} samples, skipped: {skipped}")
    print(f"Total audio: {sum(duration_list)/3600:.2f} hours")

    if not result:
        print("\n[ERROR] No samples were collected. Check that:")
        print("  • dataset_root points to the correct folder")
        print("  • WAV files exist under that folder")
        print("  • metadata.csv paths match the actual file locations")
        sys.exit(1)

    # 4. Save raw.arrow
    # Explicitly declare the schema so ArrowWriter never hits SchemaInferenceError,
    # even if result somehow ends up empty in future edge cases.
    features = Features({
        "audio_path": Value("string"),
        "text":       Value("string"),
        "duration":   Value("float32"),
    })

    os.makedirs(SAVE_DIR, exist_ok=True)
    with ArrowWriter(path=f"{SAVE_DIR}/raw.arrow", features=features) as writer:
        for line in tqdm(result, desc="Writing raw.arrow"):
            writer.write(line)
        writer.finalize()

    # 5. Save duration.json
    with open(f"{SAVE_DIR}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # 6. Build vocab.txt — SPACE MUST BE INDEX 0 (enforced by get_tokenizer assert)
    vocab_set.discard(" ")                  # remove space from sorted set
    sorted_vocab = sorted(vocab_set)        # sort remaining chars
    with open(f"{SAVE_DIR}/vocab.txt", "w", encoding="utf-8") as f:
        f.write(" \n")                      # idx 0 = space
        for ch in sorted_vocab:
            f.write(ch + "\n")

    print(f"\nVocab size: {len(sorted_vocab) + 1} (including space)")
    print(f"Saved to:   {SAVE_DIR}/")
    print("  raw.arrow")
    print("  duration.json")
    print("  vocab.txt")


if __name__ == "__main__":
    main()