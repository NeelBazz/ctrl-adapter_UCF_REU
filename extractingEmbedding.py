import os
import torch
import torchaudio
from tqdm import tqdm

from msclap.CLAPWrapper import CLAPWrapper

#file path
ROOT_AUDIO_DIR = "/home/ne532326/AudioX/dataset"
ROOT_SAVE_DIR = "/home/ne532326/AudioX/audio_embeddings"

model = CLAPWrapper(version='2023', use_cuda=False)

for subdir in sorted(os.listdir(ROOT_AUDIO_DIR)):
    subdir_path = os.path.join(ROOT_AUDIO_DIR, subdir)

    if not os.path.isdir(subdir_path):
        continue

    print(f"Processing folder: {subdir}")

    save_subdir = os.path.join(ROOT_SAVE_DIR, subdir)
    os.makedirs(save_subdir, exist_ok=True)

    # Get all .wav files in this folder
    audio_files = [f for f in os.listdir(subdir_path) if f.endswith(".wav")]

    for fname in tqdm(audio_files, desc=f"Extracting from {subdir}", leave=False):
        try:
            audio_path = os.path.join(subdir_path, fname)

            # Extract embedding
            embedding = model.get_audio_embeddings([audio_path], resample=True)[0]

            waveform, sample_rate = torchaudio.load(audio_path)
            duration_seconds = waveform.shape[1] / sample_rate

            audio_info = {
                "embedding": embedding.cpu(),
                "duration": float(duration_seconds),
                "sample_rate": int(sample_rate),
                "filename": fname,
                "video_folder": subdir
            }

            # Save as .pt
            save_path = os.path.join(save_subdir, fname.replace(".wav", ".pt"))
            torch.save(audio_info, save_path)

        except Exception as e:
            print(f"Failed to process {fname} in {subdir}: {e}")
