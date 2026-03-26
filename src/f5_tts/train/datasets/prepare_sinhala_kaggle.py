"""
Sinhala dataset prep for F5-TTS Option A (full finetune, new vocab) on Kaggle.
"""
import csv
import json
import os
import sys
import wave as wave_module
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

# Use expandvars to resolve ${DATASET_NAME} inside KAGGLE_SAVE_DIR
KAGGLE_SAVE_DIR = os.getenv("KAGGLE_SAVE_DIR", f"/kaggle/working/data/{DATASET_NAME}")
SAVE_DIR = os.path.expandvars(KAGGLE_SAVE_DIR)

KAGGLE_DATASET_VAR = os.getenv("KAGGLE_DATASET", "/kaggle/input/datasets/amalshaf/f5-voice-dataset")
KAGGLE_DATASET = os.path.expandvars(KAGGLE_DATASET_VAR)
# ───────────────────────────────────────────────────────────────────────


def get_duration(wav_path: str) -> float:
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
    print("Preparing Kaggle dataset...")
    kaggle_root = Path(KAGGLE_DATASET)
    
    # 1. Check if the dataset is already mounted on Kaggle
    if not kaggle_root.exists():
        print(f"[INFO] Mounted Kaggle path not found at {kaggle_root}.")
        print("Falling back to downloading via kagglehub...")
        import kagglehub
        # Try to infer dataset handle from the path, e.g. "amalshaf/f5-voice-dataset"
        parts = kaggle_root.parts
        if len(parts) >= 3:
            handle = f"{parts[-2]}/{parts[-1]}"
        else:
            handle = "amalshaf/f5-voice-dataset"
        kaggle_root = Path(kagglehub.dataset_download(handle))
        print(f"Dataset downloaded to: {kaggle_root}")
    else:
        print(f"Using mounted Kaggle dataset at: {kaggle_root}")

    # 2. Locate metadata.csv
    meta_candidates = list(kaggle_root.rglob("metadata.csv"))
    assert meta_candidates, "metadata.csv not found inside downloaded dataset"
    meta_file = meta_candidates[0]
    dataset_root = meta_file.parent
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
            vocab_set.update(list(text))

    print(f"\nProcessed: {len(result)} samples, skipped: {skipped}")
    if result:
        print(f"Total audio: {sum(duration_list)/3600:.2f} hours")
    else:
        print("\n[ERROR] No samples were collected.")
        sys.exit(1)

    # 4. Save raw.arrow
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

    # 6. Build vocab.txt
    vocab_set.discard(" ")
    sorted_vocab = sorted(vocab_set)
    with open(f"{SAVE_DIR}/vocab.txt", "w", encoding="utf-8") as f:
        f.write(" \n")
        for ch in sorted_vocab:
            f.write(ch + "\n")

    print(f"\nVocab size: {len(sorted_vocab) + 1} (including space)")
    print(f"Saved to:   {SAVE_DIR}/")
    print("  raw.arrow")
    print("  duration.json")
    print("  vocab.txt")


if __name__ == "__main__":
    main()