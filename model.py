import torch.nn as nn
import torch.nn.functional as F

class ConvSubsample(nn.Module):
    def __init__(self, in_channels, channels=[32, 64], kernel=(3,3)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0],
                      kernel_size=kernel,
                      stride=(1,2),
                      padding=(kernel[0]//2, kernel[1]//2)),
            nn.BatchNorm2d(channels[0]),
            nn.Hardtanh(0,20, inplace=True),

            nn.Conv2d(channels[0], channels[1],
                      kernel_size=kernel,
                      stride=(1,2),
                      padding=(kernel[0]//2, kernel[1]//2)),
            nn.BatchNorm2d(channels[1]),
            nn.Hardtanh(0,20, inplace=True),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        return self.conv(x)



class DeepSpeech2(nn.Module):
    def __init__(self, n_mels=80, rnn_hidden=512, rnn_layers=5, n_classes=40):
        super().__init__()
        self.sub = ConvSubsample(in_channels=n_mels, channels=[32, 64])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        rnn_in = 64

        self.rnns = nn.ModuleList()
        for i in range(rnn_layers):
            inp = rnn_in if i == 0 else rnn_hidden * 2
            self.rnns.append(nn.GRU(inp, rnn_hidden, batch_first=True, bidirectional=True))
            self.rnns.append(nn.BatchNorm1d(rnn_hidden * 2))

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden*2, rnn_hidden),
            nn.Hardtanh(0,20),
            nn.Linear(rnn_hidden, n_classes)
        )

    def forward(self, x):
        x = self.sub(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)
        x = x.transpose(1,2)

        out = x
        for i in range(0, len(self.rnns), 2):
            rnn = self.rnns[i]
            bn = self.rnns[i+1]
            out, _ = rnn(out)
            out = bn(out.permute(0,2,1)).permute(0,2,1)

        logits = self.fc(out)
        return F.log_softmax(logits, dim=-1)
