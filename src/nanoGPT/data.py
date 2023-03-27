"""
"""

from typing import List, Sequence, Tuple

import torch
import torch.utils.data

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


torch.manual_seed(42)


def get_text(input_text_path: str) -> torch.Tensor:
    with open(input_text_path, 'r') as f:
        text = f.read()
    return text


class Tokenizer:
    def __init__(self, input_text_path: str):
        self.input_text_path = input_text_path
        self.valid_chars = self.get_valid_chars(input_text_path)
        self.encoding = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.decoding = {idx: char for idx, char in enumerate(self.valid_chars)}

    @staticmethod
    def get_valid_chars(input_text_path) -> List[str]:
        with open(input_text_path, 'r') as f:
            text = f.read()
        valid_chars = list(set(text))
        valid_chars.sort()
        return valid_chars
    
    def encode(self, text: str) -> List[int]:
        return torch.tensor([self.encoding[char] for char in text], dtype=torch.long)

    def decode(self, indicies: Sequence[int]) -> str:
        try:
            sequence = [self.decoding[idx.item()] for idx in indicies]  # Handle PyTorch
        except SyntaxError:
            sequence = [self.decoding[idx] for idx in indicies]
            
        return "".join(sequence)


class CharLevelDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        if idx < 0:
            raise NotImplementedError("Counting doesn't work")
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]
    

def get_dataset(input_text_path: str, block_size: int, batch_size: int, split_ratio: float) -> Tuple[Tokenizer, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    tokenizer = Tokenizer(input_text_path = input_text_path)
    raw_text = get_text(input_text_path)
    data = tokenizer.encode(raw_text)
    
    n = round(split_ratio*len(data))
    train_data = CharLevelDataset(data=data[:n], block_size=block_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = CharLevelDataset(data[n:], block_size=block_size)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return tokenizer, train_dataloader, val_dataloader

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> int:
    get_dataset(cfg.data.input_text_path, cfg.model.block_size, cfg.loader.batch_size, cfg.data.split_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
