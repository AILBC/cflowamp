import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from flow_model import FlowMatchingTransformer, FlowMatching
from utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from autoencoder import Autoencoder # Import Autoencoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from properties_utils import VALID_AMINO_ACIDS, calculate_net_charge, calculate_hydrophobic_moment 


class ESMEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, masks_path, csv_path):
        self.embeddings = torch.load(embeddings_path)
        self.masks = torch.load(masks_path)
        df = pd.read_csv(csv_path)
        self.sequences = df['Sequence'].astype(str).tolist()
        self.properties = self._calculate_raw_properties()
        self.lengths = self.masks.sum(dim=1).int() - 2 
        
    def _calculate_raw_properties(self):
        properties_list = []
        for seq in self.sequences:
            try:
                if not set(seq).issubset(VALID_AMINO_ACIDS):
                    raise ValueError("Sequence contains invalid amino acid letters")

                analysed_seq = ProteinAnalysis(seq)
                net_charge = calculate_net_charge(seq)
                hydrophobicity = analysed_seq.gravy()

                helix_fraction, _, _ = analysed_seq.secondary_structure_fraction() 

                properties_list.append([net_charge, hydrophobicity, helix_fraction])

            except Exception as e:
            
                properties_list.append([0.0, 0.0, 0.0])
        
        properties_tensor = torch.tensor(properties_list, dtype=torch.float32)
        return properties_tensor

    def normalize_properties(self, mean, std):
        self.properties = (self.properties - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'mask': self.masks[idx],
            'length': self.lengths[idx],
            'sequence': self.sequences[idx],
            'properties': self.properties[idx] 
        }

def split_dataset(dataset, train_ratio=0.9, seed=seed):
    val_ratio = 1.0 - train_ratio 
    set_seed(seed)
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset

def generate_condition_drop_pattern(num_epochs, num_batches, drop_prob=0.2):
    pattern = np.random.rand(num_epochs, num_batches) < drop_prob
    return pattern
def train_flow_matching(
    train_dataset, 
    val_dataset, 
    model_config={},
    train_config={},
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_model_config = {
        'd_model': 80, 
        'n_head': 8,
        'num_layers': 16,  
        'max_len': 30,
        'dropout_p': 0.15,
        'num_conditions': 3  
    }
    model_config = {**default_model_config, **model_config}
    default_train_config = {
        'batch_size': 24,      
        'grad_accum_steps': 4,  
        'num_epochs': 100,       
        'lr': 8e-4,              
        'weight_decay': 5e-5,
        'early_stop_patience': 20, 
        'grad_clip_value': 1.0, 
        'val_t_value': 0.5,     
        'save_dir': 'esmflow/model_out',
        'save_name': 'flow_model.pt',
        'gpu_ids': None,
        'warmup_epochs': 5,     
        'min_lr': 1e-6,          
        'warmup_start_lr': 5e-7, 
        'use_amp': True,         
        'autoencoder_path': 'esmflow/model_out/autoencoder.pt' 
    }
    train_config = {**default_train_config, **train_config}
    gpu_ids = train_config.get('gpu_ids')
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    set_seed(train_config.get('seed', seed))
    autoencoder = Autoencoder().to(device)
    if os.path.exists(train_config['autoencoder_path']):
        autoencoder.load_state_dict(torch.load(train_config['autoencoder_path'], map_location=device))
        autoencoder.eval()
    else:
        raise FileNotFoundError()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    model = FlowMatchingTransformer(**model_config) 
    flow_matching = FlowMatching()   
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )   
    num_training_steps = len(train_loader) * train_config['num_epochs']
    warmup_steps = len(train_loader) * train_config.get('warmup_epochs', 5)   
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    scaler = GradScaler() if train_config.get('use_amp', False) and torch.cuda.is_available() else None
    os.makedirs(train_config['save_dir'], exist_ok=True)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    condition_drop_pattern = generate_condition_drop_pattern(
        train_config['num_epochs'], 
        len(train_loader), 
        drop_prob=0.2
    )
    for epoch in range(train_config['num_epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad() 
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}")
        for batch_idx, batch in enumerate(progress_bar):
            H_real_original = batch['embedding'].to(device) 
            M_real = batch['mask'].to(device)
            properties = batch['properties'].to(device) 
            if condition_drop_pattern[epoch, batch_idx]:
                condition_to_use = None
            else:
                condition_to_use = properties
            with torch.no_grad(): 
                _, H_real = autoencoder(H_real_original)
            if scaler is not None:
                with autocast():
                    time_steps = torch.rand(H_real.size(0), device=H_real.device)
                    H_t, H_0 = flow_matching.create_flow(H_real, time_steps) 
                    v_pred = model(H_t=H_t, t=time_steps, condition=condition_to_use) 
                    loss = flow_matching.mse_loss(v_pred, H_real, H_0, mask=M_real)
                    loss = loss / train_config['grad_accum_steps']
                scaler.scale(loss).backward()
                train_loss += loss.item() * train_config['grad_accum_steps']
                if (batch_idx + 1) % train_config['grad_accum_steps'] == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip_value'])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step() 
                    optimizer.zero_grad()
            else:
                time_steps = torch.rand(H_real.size(0), device=H_real.device)
                H_t, H_0 = flow_matching.create_flow(H_real, time_steps) 
                v_pred = model(H_t=H_t, t=time_steps, condition=condition_to_use)
                loss = flow_matching.mse_loss(v_pred, H_real, H_0, mask=M_real)
                loss = loss / train_config['grad_accum_steps']
                loss.backward()
                train_loss += loss.item() * train_config['grad_accum_steps']
                if (batch_idx + 1) % train_config['grad_accum_steps'] == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip_value'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            progress_bar.set_postfix({
                'loss': loss.item() * train_config['grad_accum_steps'],
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                H_real_original = batch['embedding'].to(device)
                M_real = batch['mask'].to(device)
                properties = batch['properties'].to(device)
                _, H_real = autoencoder(H_real_original)
                time_steps = torch.full((H_real.size(0),), train_config['val_t_value'], device=H_real.device)
                H_t, H_0 = flow_matching.create_flow(H_real, time_steps)
                if scaler is not None:
                    with autocast():
                        v_pred = model(H_t=H_t, t=time_steps, condition=properties)
                        loss = flow_matching.mse_loss(v_pred, H_real, H_0, mask=M_real)
                else:
                    v_pred = model(H_t=H_t, t=time_steps, condition=properties)
                    loss = flow_matching.mse_loss(v_pred, H_real, H_0, mask=M_real)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            save_path = os.path.join(train_config['save_dir'], train_config['save_name'])
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config, 
                'train_config': train_config
            }, save_path)
        else:
            early_stop_counter += 1
        if (epoch + 1) % 10 == 0 and epoch+1>=40:
            checkpoint_save_path = os.path.join(train_config['save_dir'], f'flow_model_pro_epoch_{epoch+1}.pt')
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config,
                'train_config': train_config
            }, checkpoint_save_path)
        if early_stop_counter >= train_config['early_stop_patience']:
            break
    checkpoint = torch.load(os.path.join(train_config['save_dir'], train_config['save_name']))
    final_model = FlowMatchingTransformer(**checkpoint['model_config'])
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.to(device)  
    return final_model

if __name__ == "__main__":
    data_dir = "data2/"
    gpu_ids = [0,1,2,3]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    dataset = ESMEmbeddingDataset(
        embeddings_path=os.path.join(data_dir, "esm_sequence_embeddings_930.pt"),
        masks_path=os.path.join(data_dir, "esm_attention_masks_930.pt"),
        csv_path=os.path.join(data_dir, "pos_data930.csv")
    )
    train_dataset, val_dataset = split_dataset(dataset)
    train_indices = train_dataset.indices
    train_properties = dataset.properties[train_indices]
    properties_mean = train_properties.mean(dim=0, keepdim=True)
    properties_std = train_properties.std(dim=0, keepdim=True)
    save_dir = 'esmflow/model_out'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'mean': properties_mean, 'std': properties_std}, os.path.join(save_dir, 'properties_stats.pt'))
    dataset.normalize_properties(properties_mean, properties_std)
    model = train_flow_matching(
        train_dataset, 
        val_dataset,
        model_config={
            'd_model': 80,        
            'num_layers': 8,    
            'dropout_p': 0.25,    
            # 'num_layers': 16,     
            # 'dropout_p': 0.2,      
            'num_conditions': 3  
        },
        train_config={
            'batch_size': 128,     
            'grad_accum_steps': 3, 
            'num_epochs': 200,    
            'lr': 1e-4,           
            'early_stop_patience': 40, 
            'save_dir': 'esmflow/model_out',
            'save_name': 'flow_model.pt', 
            'gpu_ids': gpu_ids,
            'warmup_epochs': 10,    
            'min_lr': 1e-6,       
            'warmup_start_lr': 5e-7,
            'use_amp': True,
            'autoencoder_path': 'esmflow/model_out/autoencoder.pt'
        }
    )