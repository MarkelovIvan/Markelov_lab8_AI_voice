import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import LJSpeechDataset, collate_fn, char2idx
from model import DeepSpeech2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 9
LR = 1e-4


def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        x, in_len, y, y_len = batch
        x = x.to(device)
        y = y.to(device)

        log_probs = model(x)
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, C]

        in_len = (in_len.float() / 4).ceil().long()

        optimizer.zero_grad()
        loss = criterion(log_probs, y, in_len, y_len)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def valid_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x, in_len, y, y_len = batch
            x = x.to(device)
            y = y.to(device)

            log_probs = model(x)
            log_probs = log_probs.permute(1, 0, 2)
            in_len = (in_len.float() / 4).ceil().long()

            loss = criterion(log_probs, y, in_len, y_len)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    data_root = "data/LJSpeech-1.1"

    ds = LJSpeechDataset(data_root)
    n = len(ds)
    n_train = int(n * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    model = DeepSpeech2(
        n_mels=80,
        rnn_hidden=512,
        rnn_layers=5,
        n_classes=len(char2idx) + 1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs("checkpoints", exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        train_loss = train_loop(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = valid_loop(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/deepspeech2_best.pt")
            print("Saved best checkpoint")


if __name__ == "__main__":
    main()
