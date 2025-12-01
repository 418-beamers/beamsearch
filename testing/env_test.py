# run this to validate cuda install

import torch
import torch, sys

print("torch.__file__ =", torch.__file__)
print("sys.path =", sys.path)

print("torch build:", torch.__version__, "torch cuda:", torch.version.cuda)
print("Cuda available", torch.cuda.is_available())
