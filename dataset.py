# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa


SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
WINDOW = "hann"

CHARS = "abcdefghijklmnopqrstuvwxyz' ,.-?"
char2idx = {c: i+1 for i,c in enumerate(CHARS)}
idx2char = {i: c for c,i in char2idx.items()}
idx2char[0] = ""

def text_to_int_sequence(text):
    text = text.lower()
    seq = []
    for ch in text:
        if ch in char2idx:
            seq.append(char2idx[ch])
    return seq

def compute_log_mel(waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window=WINDOW
    )
    log_mel = librosa.power_to_db(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)

class LJSpeechDataset(Dataset):
    def __init__(self, root_dir, sample_rate=SAMPLE_RATE, max_len=None, transform=None):
        self.root = root_dir
        metadata = os.path.join(root_dir, "metadata.csv")
        self.items = []
        with open(metadata, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    fname = parts[0] + ".wav"
                    text = parts[2]
                    wav_path = os.path.join(root_dir, "wavs", fname)
                    if os.path.exists(wav_path):
                        self.items.append((wav_path, text))
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        waveform, sr = sf.read(wav_path, dtype='float32')
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        log_mel = compute_log_mel(waveform, sr=self.sample_rate)  # shape (n_mels, T)
        spec = torch.from_numpy(log_mel)  # float32
        target = torch.tensor(text_to_int_sequence(text), dtype=torch.long)
        return spec, target

def collate_fn(batch):
    specs = [b[0].transpose(0,1) for b in batch]  # -> [T, n_mels] for easier padding
    input_lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    max_in = max([s.shape[0] for s in specs])
    n_mels = specs[0].shape[1]
    batch_in = torch.zeros(len(specs), max_in, n_mels, dtype=torch.float32)
    for i,s in enumerate(specs):
        batch_in[i, :s.shape[0], :] = s
    batch_in = batch_in.transpose(1,2)  # [B, n_mels, T]
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return batch_in, input_lengths, targets, target_lengths

if __name__ == "__main__":
    ds_root = "data/LJSpeech-1.1"
    ds = LJSpeechDataset(ds_root)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    for batch in loader:
        print([x.shape for x in batch[:3]])
        break
