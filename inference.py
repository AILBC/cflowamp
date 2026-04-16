import torch
import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import time
from flow_model import FlowMatchingTransformer, FlowMatching
import torch.nn as nn
from utils import set_seed
from autoencoder import Autoencoder
def get_length_distribution(csv_path):
    df = pd.read_csv(csv_path)
    lengths = df['Sequence'].str.len().tolist()
    return lengths, df['Sequence'].tolist()


def generate_samples(model, num_samples=10, max_len=30, device=None, condition=None, guidance_scale=3.0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    flow_matching = FlowMatching()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    latent_dim = base_model.input_proj.weight.shape[0] 
    total_len = max_len + 2
    shape = (total_len, latent_dim)
    latent_generated = flow_matching.sample_heun(
        model, num_samples, shape, device, condition=condition, num_steps=50, guidance_scale=guidance_scale
    )
    
    return latent_generated


def main():
    default_config = {
        'gpu_ids': [0],
        'seed': seed,
        'num_samples': 50000,
        'max_real_samples': 10000,
        'data_csv': 'data2/pos_data930.csv', 
        'model_path': 'esmflow/model_out/flow_model86.pt',
        'autoencoder_path': 'esmflow/model_out/autoencoder.pt', 
        'properties_stats_path': 'esmflow/model_out/properties_stats.pt', 
        'save_embeddings': True,
        'output_latent_path': 'esmflow/gen_out/generated_latent.pt',
        'output_embeddings_path': 'esmflow/gen_out/generated_embeddings.pt',
        'output_masks_path': 'esmflow/gen_out/generated_masks.pt', 
        'use_condition': True,  
        'guidance_scale': 0,  
        'generation_batch_size': 256, 
        'target_properties': { 
            'net_charge': 3.52, 
            'hydrophobicity': -0.26,
            'alpha_helix_ratio': 0.33 
        }
    }

    if default_config['gpu_ids']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, default_config['gpu_ids']))

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()

    set_seed(default_config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    autoencoder_path = default_config['autoencoder_path']
    autoencoder = Autoencoder().to(device)
    if os.path.exists(autoencoder_path):
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        autoencoder.eval()
    else:
        return
    
    properties_stats_path = default_config['properties_stats_path']
    if os.path.exists(properties_stats_path):
        properties_stats = torch.load(properties_stats_path, map_location='cpu')
        prop_mean = properties_stats['mean']
        prop_std = properties_stats['std']
    else:
        prop_mean = torch.tensor([[3.0, 0.2, 0.4]], dtype=torch.float32)  
        prop_std = torch.tensor([[2.0, 0.5, 0.2]], dtype=torch.float32)
    

    checkpoint = torch.load(default_config['model_path'], map_location=device)
    model_config = checkpoint['model_config']
    
    
    flow_model = FlowMatchingTransformer(**model_config).to(device)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.eval()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        flow_model = nn.DataParallel(flow_model)
    real_lengths, _ = get_length_distribution(default_config['data_csv'])
    sampled_lengths = np.random.choice(
        [l for l in real_lengths if l <= 30], 
        default_config['num_samples'], 
        replace=True
    )
    length_counts = Counter(sampled_lengths)


    condition_tensor_template = None
    if default_config['use_condition']:
        target_properties = np.array([
            [
                default_config['target_properties']['net_charge'],
                default_config['target_properties']['hydrophobicity'],
                default_config['target_properties']['alpha_helix_ratio']
            ]
        ])
        normalized_target = (target_properties - prop_mean.numpy()) / (prop_std.numpy() + 1e-8)
        condition_tensor_template = torch.tensor(normalized_target, dtype=torch.float32)
    start_time = time.time()
    
    all_latents = []
    all_masks = []
    generation_batch_size = default_config['generation_batch_size']

    with tqdm(total=default_config['num_samples']) as pbar:
       
        for length, count in sorted(length_counts.items()):
            
            for i in range(0, count, generation_batch_size):
                current_batch_size = min(generation_batch_size, count - i)
                
                condition_tensor = None
                if default_config['use_condition'] and condition_tensor_template is not None:
                    condition_tensor = condition_tensor_template.repeat(current_batch_size, 1)

                generated_latent_batch = generate_samples(
                    model=flow_model,
                    num_samples=current_batch_size,
                    max_len=length,
                    device=device,
                    condition=condition_tensor,
                    guidance_scale=default_config['guidance_scale']
                )
                all_latents.append(generated_latent_batch.cpu())
                
                mask = torch.zeros(current_batch_size, length + 2)
                mask[:, :length + 2] = 1
                all_masks.append(mask)

                pbar.update(current_batch_size)

    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 0:
        samples_per_second = default_config['num_samples'] / total_time
    else:
        samples_per_second = float('inf')
    
    max_len_overall = max(latent.shape[1] for latent in all_latents)
    
    padded_latents = torch.cat([
        torch.nn.functional.pad(latent, (0, 0, 0, max_len_overall - latent.shape[1]))
        for latent in all_latents
    ], dim=0)

    padded_masks = torch.cat([
        torch.nn.functional.pad(mask, (0, max_len_overall - mask.shape[1]))
        for mask in all_masks
    ], dim=0)

    decoded_embeddings = []
    with torch.no_grad(), tqdm(total=padded_latents.shape[0]) as pbar:
        for i in range(0, padded_latents.shape[0], generation_batch_size):
            batch_latents = padded_latents[i:i+generation_batch_size].to(device)
            batch_decoded = autoencoder.decoder(batch_latents)
            decoded_embeddings.append(batch_decoded.cpu())
            pbar.update(batch_latents.shape[0])
    
    final_embeddings = torch.cat(decoded_embeddings, dim=0)
    if default_config['save_embeddings']:
        os.makedirs(os.path.dirname(default_config['output_latent_path']), exist_ok=True)
        torch.save(padded_latents, default_config['output_latent_path'])
        os.makedirs(os.path.dirname(default_config['output_embeddings_path']), exist_ok=True)
        torch.save(final_embeddings, default_config['output_embeddings_path'])       
        os.makedirs(os.path.dirname(default_config['output_masks_path']), exist_ok=True)
        torch.save(padded_masks, default_config['output_masks_path'])
    
    if default_config['use_condition']:
        condition_info = {
            'target_properties': default_config['target_properties'],
            'normalized_target': normalized_target,
            'mean': prop_mean.numpy(),
            'std': prop_std.numpy(),
            'guidance_scale': default_config['guidance_scale']
        }
        condition_path = os.path.join(os.path.dirname(default_config['output_latent_path']), 'condition_info.pt')
        torch.save(condition_info, condition_path)
    


if __name__ == "__main__":
    main() 