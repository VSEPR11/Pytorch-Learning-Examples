import nn
import torch

A = torch.tensor([4., 4., 3., 4.])
B = torch.tensor([[4.], [2.], [3.], [4.]])
print(B.shape)
print(A == B)
