
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np

from utils import set_seed 

class Autoencoder(nn.Module):
    def __init__(self, input_dim=1280, latent_dim=80, seq_len=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x) 
        reconstructed = self.decoder(latent)
        return reconstructed, latent

if __name__ == "__main__":
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_EPOCHS = 50
    AE_BATCH_SIZE = 64
    AE_LR = 1e-4
    DATA_PATH = 'data2/esm_sequence_embeddings_930.pt'
    SAVE_PATH = 'esmflow/model_out/autoencoder.pt'
    try:
        embeddings = torch.load(DATA_PATH, map_location=device)
    except FileNotFoundError:
        exit()
    
    num_samples, seq_len, input_dim = embeddings.shape
    latent_dim = 80

    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, seq_len=seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0
        for batch_embeddings, in loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_embeddings)
            loss = criterion(reconstructed, batch_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)