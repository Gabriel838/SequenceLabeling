import torch
import torch.nn as nn
import torch.nn.functional as F

N, C = 5, 4

loss = nn.NLLLoss()
data = torch.randn(5, 4, 8)
data.requires_grad = True
target = torch.Tensor(5, 8).random_(0, C).long()

print("target:", target.size())
output = loss(data, target)
print("loss:", output)
output.backward()