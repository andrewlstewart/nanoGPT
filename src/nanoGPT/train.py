"""
"""

import torch
import torch.utils.data
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

import hydra
from omegaconf import DictConfig

# from src.nanoGPT.bigram_model import BigramLanguageModel
from src.nanoGPT.attention_model import MHSAModel
from src.nanoGPT.data import get_dataset


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> int:
    tokenizer, train_dataloader, val_dataloader = get_dataset(input_text_path=cfg.data.input_text_path, block_size=cfg.model.block_size,
                                                              batch_size=cfg.loader.batch_size, split_ratio=cfg.data.split_ratio)
    model = MHSAModel(vocab_size=len(tokenizer.valid_chars), 
                                model_config=cfg.model,
                                training_config=cfg.training)

    limit_train_batches = cfg.training.debug.limit_train_batches if cfg.training.debug.enabled else None

    tensorboard_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.logging.lightning_log_root)

    trainer = pl.Trainer(logger=tensorboard_logger, limit_train_batches=limit_train_batches, max_epochs=cfg.training.epochs)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    generation_init = torch.zeros((1,1), dtype=torch.long)  # Start generation with the break line character
    generated_output = tokenizer.decode(model.generate(idx=generation_init, max_new_tokens=100)[0])
    print(f'{generated_output=}')

    return 0

# def main() -> int:
#     args = parse_args()

#     tokenizer, train_dataloader, val_dataloader = get_dataset(input_text_path='data/tinyshakespeare/input.txt', block_size=8, batch_size=4, split_ratio=0.9)
#     model = BigramLanguageModel(len(tokenizer.valid_chars))

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

#     train_one_epoch(model, optimizer, train_dataloader, val_dataloader, max_iter=10_000)

#     generation_init = torch.zeros((1,1), dtype=torch.long)  # Start generation with the break line character
#     tokenizer.decode(model.generate(idx=generation_init, max_new_tokens=100)[0])
#     return 0


if __name__ == "__main__":
    raise SystemExit(main())
