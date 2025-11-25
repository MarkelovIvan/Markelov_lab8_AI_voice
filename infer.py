# infer.py
import torch
import numpy as np
import soundfile as sf
import librosa
from model import DeepSpeech2
from dataset import compute_log_mel, idx2char, char2idx

def greedy_decode(log_probs):
    preds = np.argmax(log_probs, axis=-1)
    prev = -1
    out = []
    for p in preds:
        if p != prev and p != 0:
            out.append(p)
        prev = p
    return "".join([idx2char.get(int(i), "") for i in out])

def load_model(path, device):
    m = DeepSpeech2(n_mels=80, rnn_hidden=512, rnn_layers=5, n_classes=len(char2idx)+1)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m

def transcribe_file(model, filepath, device):
    waveform, sr = sf.read(filepath, dtype='float32')
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    if sr != 22050:
        waveform = librosa.resample(waveform, sr, 22050)
    spec = compute_log_mel(waveform)
    import torch
    x = torch.from_numpy(spec).unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        log_probs = model(x)
    lp = log_probs[0].cpu().numpy()
    text = greedy_decode(lp)
    return text

if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("checkpoints/deepspeech2_best.pt", device)
    files = sys.argv[1:] if len(sys.argv)>1 else ["data/LJSpeech-1.1/wavs/LJ001-0001.wav"]
    for f in files:
        print("File:", f)
        txt = transcribe_file(model, f, device)
        print("->", txt)
