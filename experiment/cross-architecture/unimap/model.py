import torch
import torch.nn as nn

class PaperLSTM(nn.Module):
    def __init__(self, input_dim=64, use_embedding=True, vocab_size=682, n_classes=2):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=192, num_layers=2,
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
