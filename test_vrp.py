import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP


model, _ = load_model('pretrained/cvrp_50/')
torch.manual_seed(1234)
dataset = CVRP.make_dataset(size=50, num_samples=10)


# Need a dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=1000)

# Make var works for dicts
batch = next(iter(dataloader))

# Run the model
model.eval()
model.set_decode_type('greedy')
with torch.no_grad():
    length, log_p, pi = model(batch, return_pi=True)
print(pi)