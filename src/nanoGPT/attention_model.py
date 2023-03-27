"""
"""

from typing import Optional

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig


class Head(torch.nn.Module):
    def __init__(self, n_embedding, head_size, block_size, dropout_rate):
        super().__init__()
        self.key = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.query = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.value = torch.nn.Linear(n_embedding, head_size, bias=False)
        # tril isn't a parameter of the model (ie. it's just a lower triangular matrix of ones)
        # ie. we don't want to do any updates to this matrix, hence regist_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)

        # Comptute affinities
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        # The above [:T, :T] logic is needed at generation time (ie. if the context is a single token, then T_in = 1 and
        # matrix multiplication will fail against the tril already defined block_size shape (1hr21m)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)

        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, num_heads, n_embedding, head_size, block_size, dropout_rate):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(n_embedding, head_size, block_size, dropout_rate) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(n_embedding, n_embedding)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(torch.nn.Module):
    def __init__(self, n_embedding, dropout_rate):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embedding, 4 * n_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embedding, n_embedding),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class Block(torch.nn.Module):
    def __init__(self, n_embedding, n_head, block_size, dropout_rate):
        super().__init__()
        head_size = n_embedding // n_head
        self.mhsa = MultiHeadedAttention(n_head, n_embedding, head_size, block_size, dropout_rate)
        self.ffwd = FeedForward(n_embedding, dropout_rate)
        self.layer_norm_1 = torch.nn.LayerNorm(n_embedding)
        self.layer_norm_2 = torch.nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.mhsa(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x


class MHSAModel(pl.LightningModule):
    """
    Simple model which during inference, takes in the current token (ie. character), pulls out it's row in an
    embedding matrix, and returns that as the logits (ie. unnormalized probabilities) for the next character.
    """

    def __init__(self,
                 vocab_size: int,
                 model_config: Optional[DictConfig] = None,
                 training_config: Optional[DictConfig] = None):
        super().__init__()
        self.block_size = model_config.block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, model_config.n_embedding)
        # Because attention will effectively ignore position, (set operation), reintroduce position via additional embedding
        # 1hr20m25s
        self.position_embedding_table = torch.nn.Embedding(self.block_size, model_config.n_embedding)

        self.blocks = torch.nn.Sequential(
            *[Block(model_config.n_embedding, model_config.n_heads, model_config.block_size, model_config.dropout_rate) for _ in range(model_config.n_blocks)],
            torch.nn.LayerNorm(model_config.n_embedding)
        )
        self.feed_forward = FeedForward(model_config.n_embedding, model_config.dropout_rate)
        self.language_model_head = torch.nn.Linear(model_config.n_embedding, vocab_size)

        if training_config:
            self.learning_rate = training_config.learning_rate

    def training_step(self, batch, batch_idx):
        # https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&t=1553s

        inputs, targets = batch

        logits = self.forward(inputs)  # (B,T,C)

        batch_size, time, channel = logits.size()
        logits = logits.view(batch_size*time, channel)
        targets = targets.view(batch_size*time)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        self.log_dict({"Loss": loss})

        return loss

    def forward(self, idx):
        # https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&t=1553s
        B, T = idx.size()
        token_embedding = self.token_embedding_table(idx)  # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(T, device=next(self.position_embedding_table.parameters()).device))  # (T, C)
        x = token_embedding + position_embedding  # (B, T, C) + (T, C) -> (B, T, C) + (B, T, C) <- auto broadcasting
        x = self.blocks(x)  # (B, T, C)
        logits = self.language_model_head(x)  # (B, T, vocab_size)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # Ignore anything beyond the block size for self-attention
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> int:
    model = MHSAModel(vocab_size=64,
                      model_config=cfg.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
