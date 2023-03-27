"""
"""

from typing import Optional

import torch
import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.nanoGPT.data import get_dataset


torch.manual_seed(42)


class BigramLanguageModel(pl.LightningModule):
    """
    Simple model which during inference, takes in the current token (ie. character), pulls out it's row in an
    embedding matrix, and returns that as the logits (ie. unnormalized probabilities) for the next character.
    """

    def __init__(self, 
                 vocab_size: int, 
                 model_config: Optional[DictConfig] = None,
                 training_config: Optional[DictConfig] = None):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, model_config.n_embedding)
        # Because attention will effectively ignore position, (set operation), reintroduce position via additional embedding
        self.position_embedding_table = torch.nn.Embedding(model_config.block_size, model_config.n_embedding)
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

        return loss

    def forward(self, idx):
        # https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&t=1553s
        B, T = idx.size()
        token_embedding = self.token_embedding_table(idx)  # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(T))  # (T, C)
        x = token_embedding = position_embedding  # (B, T, C) + (T, C) -> (B, T, C) + (B, T, C) <- auto broadcasting
        logits = self.language_model_head(x)  # (B, T, vocab_size)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> int:
    tokenizer, train_dataloader, val_dataloader = get_dataset(input_text_path=cfg.data.input_text_path, block_size=cfg.model.block_size,
                                                              batch_size=cfg.loader.batch_size, split_ratio=cfg.data.split_ratio)
    model = BigramLanguageModel(vocab_size=len(tokenizer.valid_chars),
                                model_config = cfg.model)

    ### LOAD CHECKPOINT

    for inputs, targets in train_dataloader:
        logits = model(inputs)
        break
    generated_text = model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
