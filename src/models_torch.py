import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer | None = None,
):
    """General training/validation epoch"""
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    loss_agg = 0
    for tokens, labels_det, labels_true in loader:
        # Forward pass
        tag_scores: torch.Tensor = model(tokens, labels_det)
        # Reshape for loss calculation
        loss = loss_fn(tag_scores.view(-1, tag_scores.size(-1)), labels_true.view(-1))
        loss_agg += loss.item()
        if optimizer is not None:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_agg / len(loader)


class LSTMTagger(nn.Module):
    def __init__(
        self,
        token_vocab_size,
        label_vocab_size,
        embedding_dim=12,
        hidden_dim=128,
        n_lstm_layers=2,
        dropout_lstm=0.3,
        bidi=True,
    ):
        super(LSTMTagger, self).__init__()
        self.embedding_tokens = nn.Embedding(
            token_vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding_labels = nn.Embedding(
            label_vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            2 * embedding_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidi,
        )
        # double size if bidirectional
        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidi else 1), label_vocab_size)

    def forward(self, tokens, labels):
        embeds_tokens = self.embedding_tokens(tokens)
        embeds_labels = self.embedding_labels(labels)

        embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)
        # Shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        # Shape: (batch_size, seq_len, hidden_dim)
        tag_scores = self.hidden2tag(lstm_out)
        # Shape: (batch_size, seq_len, tagset_size)
        return tag_scores


class LSTMAttTagger(nn.Module):
    def __init__(
        self,
        token_vocab_size,
        label_vocab_size,
        embedding_dim,
        hidden_dim,
        n_lstm_layers=1,
        dropout_lstm=0,
        bidi=False,
        dropout=0.1,
    ):
        super(LSTMAttTagger, self).__init__()
        self.embedding_tokens = nn.Embedding(
            token_vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding_labels = nn.Embedding(
            label_vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            2 * embedding_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidi,
        )

        # double size if bidirectional
        self.attention = AttentionLayer(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidi else 1), label_vocab_size)

    def forward(self, tokens, labels):
        embeds_tokens = self.embedding_tokens(tokens)
        embeds_labels = self.embedding_labels(labels)

        embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)
        # Shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        # Shape: (batch_size, seq_len, hidden_dim)
        # Apply attention to each time step

        context_vectors = []
        attention_weights = []
        for t in range(lstm_out.size(1)):  # Iterate over sequence length
            context_vector, weights = self.attention(lstm_out[:, t, :].unsqueeze(1))
            context_vectors.append(context_vector)
            attention_weights.append(weights)

        context_vectors = torch.stack(
            context_vectors, dim=1
        )  # [batch_size, seq_len, hidden_dim * 2]
        attention_weights = torch.stack(
            attention_weights, dim=1
        )  # [batch_size, seq_len, seq_len]

        # last droupout
        context_vectors = self.dropout(context_vectors)
        tag_scores = self.hidden2tag(context_vectors)  # Pass through classifier
        return tag_scores


class AttentionLayer(nn.Module):
    """Attention on top of LSTM layer"""

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attention_scores = self.attention(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Normalize scores
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out)
        context_vector = context_vector.squeeze(1)
        return context_vector, attention_weights


def seqs2padded_tensor(
    sequences: list[list[int | float]],
    pad_value=0,
    verbose=True,
    device: str | None = None,
):
    t = torch.nn.utils.rnn.pad_sequence(
        (torch.tensor(s) for s in sequences),
        batch_first=True,
        padding_value=pad_value,
    ).to(device)
    if verbose:
        print("padded tensor:", tuple(t.size()), t.device)
    return t
