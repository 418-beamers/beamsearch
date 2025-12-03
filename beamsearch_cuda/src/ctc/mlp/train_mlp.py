import torch
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np

INPUT_SIZE = 3
HIDDEN_SIZE = 16
OUTPUT_SIZE = 3

class BeamScheduleNet(nn.Module):
    def __init__(self):
        super(BeamScheduleNet, self).__init__()
        # Input -> Hidden
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        # Hidden -> Output
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

def generate_batch(batch_size=1000):
    """
        Function to generate synthetic data based on the hypothesis that
        beam width should decrease as entropy decreases.

        Args:
            batch_size (int) - how much data to generate
        Returns:
            the synthetic data
    """
    inputs = torch.rand(batch_size, INPUT_SIZE) * 5.0
    means = torch.mean(inputs, dim=1, keepdim=True) /  5.0

    alpha = 0.2 + (0.8 * means)
    beta = means
    gamma = 0.0 + (0.3 * means)

    targets = torch.cat((alpha, beta, gamma), dim=1)
    return inputs, targets

model = BeamScheduleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training MLP...")
for epoch in range(500):
    inputs, targets = generate_batch()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.6f}")

print("Saving weights...")

with open("mlp_weights.bin", "wb") as f:
    f.write(struct.pack("iii", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE))
    w1 = model.fc1.weight.detach().numpy().flatten()
    b1 = model.fc1.bias.detach().numpy().flatten()
    f.write(struct.pack(f"{len(w1)}f", *w1))
    f.write(struct.pack(f"{len(b1)}f", *b1))
    w2 = model.fc2.weight.detach().numpy().flatten()
    b2 = model.fc2.bias.detach().numpy().flatten()
    f.write(struct.pack(f"{len(w2)}f", *w2))
    f.write(struct.pack(f"{len(b2)}f", *b2))

print("MLP complete.")
    
