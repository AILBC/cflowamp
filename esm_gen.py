import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from collections import Counter
import re
from flow_model import FlowMatchingTransformer, FlowMatching
from esm_decoder import decode_with_lm_head, load_lm_head, CalibrationModel
from utils import set_seed
import torch.nn as nn
from autoencoder import Autoencoder
from tqdm import tqdm # Added for batch processing progress

def get_length_distribution(csv_path):
    df = pd.read_csv(csv_path)
    lengths = df['Sequence'].str.len().tolist()
    return lengths, df['Sequence'].tolist()

def get_aa_composition(sequences):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_counter = Counter(''.join(sequences))
    composition = {aa: aa_counter.get(aa, 0) for aa in aa_list}  
    total = sum(composition.values())
    if total > 0:
        composition_percent = {aa: count / total * 100 for aa, count in composition.items()}
    else:
        composition_percent = {aa: 0 for aa in aa_list}    
    return composition_percent

def main():
    default_config = {
        'gpu_ids': [0, 1, 2, 3,4,5,6,7],
        'seed': seed,
        'embeddings_path': 'esmflow/gen_out/generated_embeddings.pt',
        'masks_path': 'esmflow/gen_out/generated_masks.pt',
        'data_csv': 'data2/pos_data930.csv',
        'output_dir': 'esmflow/gen_out',
        'use_sampled_lengths': True,
        'fixed_length': -1,
        'temperature': 1.0,
        'use_calibration': True,
        'calibration_path': 'esmflow/model_out/calibration_model.pt',
        'autoencoder_path': 'esmflow/model_out/autoencoder.pt', 
        'plot_stats': True,
        'decode_batch_size': 128 
    }
    if default_config['gpu_ids']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, default_config['gpu_ids'])) 
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
    set_seed(default_config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_head, alphabet, aa_ids, _ = load_lm_head(device=device, freeze=True) 
    calibration_model = None
    if default_config['use_calibration']:
        calibration_model = CalibrationModel().to(device)
        cal_model_path = default_config['calibration_path']
        if os.path.exists(cal_model_path):
            calibration_model.load_state_dict(torch.load(cal_model_path, map_location=device)) 
            calibration_model.eval()
        else:
            calibration_model = None
    embeddings_path = default_config['embeddings_path']
    if not os.path.exists(embeddings_path):
        return   
    H_generated_cpu = torch.load(embeddings_path, map_location='cpu')
    num_to_generate = H_generated_cpu.shape[0]
    max_len = H_generated_cpu.shape[1] - 2  
    masks_path = default_config['masks_path']
    if not os.path.exists(masks_path):
        return
    pred_masks_cpu = torch.load(masks_path, map_location='cpu')
    if H_generated_cpu.shape[0] != pred_masks_cpu.shape[0]:
        return
    real_lengths, real_sequences = get_length_distribution(default_config['data_csv'])
    decoded_sequences = []
    decode_batch_size = default_config['decode_batch_size']
    for i in tqdm(range(0, num_to_generate, decode_batch_size)):
        batch_H_generated = H_generated_cpu[i : i + decode_batch_size].to(device) 
        batch_pred_masks = pred_masks_cpu[i : i + decode_batch_size].to(device)    
        with torch.no_grad():
            batch_decoded_seqs = decode_with_lm_head(
                H_pred=batch_H_generated,
                mask=batch_pred_masks,
                lm_head=lm_head,
                aa_ids=aa_ids,
                calibration_model=calibration_model,
                temperature=default_config['temperature'],
            )
        decoded_sequences.extend(batch_decoded_seqs)
    gen_lengths = [len(seq) for seq in decoded_sequences]
    output_dir = default_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'generated_sequences.txt')
    with open(save_path, 'w') as f:
        for i, seq in enumerate(decoded_sequences):
            f.write(f"{seq}\n")
    

if __name__ == "__main__":
    main()