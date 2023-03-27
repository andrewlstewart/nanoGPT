"""
"""

import torch
import torch.nn.functional as F


torch.manual_seed(42)


def main() -> int:
    B, T, C = 4, 8, 2  # Batch = block of text, Time = Token, C = Latent dimension of embedded token
    x = torch.randn(B, T, C)
    x.shape

    """
    t=45m
    If you want to give the current token some information from all prior tokens, a weak information algo could
    be just to average the current, and all previous token embeddings
    """
    x_bag_of_words = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            x_prev = x[b, :t+1]
            x_bag_of_words[b, t] = torch.mean(x_prev, dim=0)

    """
    t=48m
    matmul
    """
    torch.manual_seed(42)
    a = torch.tril(torch.ones(3, 3))
    a = a / torch.sum(a, dim=1, keepdim=True)
    b = torch.randint(0, 10, (3,2)).float()
    c = a @ b

    weights = torch.tril(torch.ones(T, T))
    weights = weights / weights.sum(1, keepdim=True)  # [T, T]
    x_bag_of_words_2 = weights @ x  # [T, T] @ [B, T, C]  -> [B, T, T] @ [B, T, C] <- auto broadcast
    assert torch.allclose(x_bag_of_words, x_bag_of_words_2)

    """
    56m
    softmax
    """
    tril = torch.tril(torch.ones(T, T))
    weights = torch.zeros((T, T))
    weights = weights.masked_fill(tril == 0, float('-inf'))  # Wherever the lower diagonal is zero, set to -inf
    weights = F.softmax(weights, dim=-1)
    x_bag_of_words_3 = weights @ x
    assert torch.allclose(x_bag_of_words, x_bag_of_words_3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
