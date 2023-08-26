import torch

def weights():
    weight=torch.tensor([962,247,605,914,644,151,1298,617,342])
    weight=weight/weight.sum()
    weight=1/weight
    weight=weight/weight.sum()
    return weight

weights()