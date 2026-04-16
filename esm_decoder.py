import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pickle
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

AA20 = "ACDEFGHIKLMNPQRSTVWY"

class CalibrationModel(nn.Module):
    def __init__(self, d_model=1280):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x)

def load_lm_head(device=None, freeze=True):
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device).eval()
    lm_head = esm_model.lm_head  # nn.Linear(1280, |V|)
    if freeze:
        for p in lm_head.parameters():
            p.requires_grad = False
    aa_ids = torch.tensor([alphabet.get_idx(a) for a in AA20], device=device, dtype=torch.long)
    return lm_head, alphabet, aa_ids, device

@torch.no_grad()
def decode_with_lm_head(H_pred, mask, lm_head, aa_ids, calibration_model=None, temperature=1.0, top_k=0):
    device = next(lm_head.parameters()).device
    H_pred = H_pred.to(device)
    mask   = mask.to(device)
    if calibration_model:
        H_pred = calibration_model(H_pred)

    logits  = lm_head(H_pred)
    logits20 = logits.index_select(-1, aa_ids)
    probs   = F.softmax(logits20 / temperature, dim=-1) 

    if top_k and top_k > 0:
        k = min(top_k, probs.size(-1))
        topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        sampled = torch.multinomial(topk_probs.view(-1, k), 1).view(probs.shape[:-1])
        idx20 = topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        idx20 = probs.argmax(dim=-1)

    idx20_cpu = idx20.detach().cpu()
    mask_cpu  = mask.detach().cpu()
    seqs = []
    for b in range(idx20_cpu.shape[0]):
        L_eff = int(mask_cpu[b].sum().item()) 
        if L_eff <= 2:
            seqs.append("")
            continue

        aa_indices = idx20_cpu[b, 1 : L_eff - 1] 
        tokens = [AA20[idx] for idx in aa_indices]
        seqs.append(''.join(tokens))
    return seqs

def loss_fn(H_pred, targets, mask, calibration_model, lm_head, aa_ids):
    device = H_pred.device
    mask = mask.bool()
    H_calibrated = calibration_model(H_pred)
    logits = lm_head(H_calibrated)
    logits20 = logits.index_select(dim=-1, index=aa_ids)

    aa_map = {aa: i for i, aa in enumerate(AA20)}
    
    active_logits = []
    active_targets = []
    for b, seq in enumerate(targets):
        L_eff = int(mask[b].sum())
        if L_eff <= 2: continue    
        aa_logits = logits20[b, 1 : L_eff - 1]     
        target_indices = torch.tensor([aa_map[aa] for aa in seq], device=device, dtype=torch.long)
        active_logits.append(aa_logits)
        active_targets.append(target_indices)
    if not active_logits:
        return torch.tensor(0.0, device=device, requires_grad=True)
    flat_logits = torch.cat(active_logits, dim=0)
    flat_targets = torch.cat(active_targets, dim=0)
    return F.cross_entropy(flat_logits, flat_targets)

def load_pos30_data(data_dir="data2"):
    pkl_path = os.path.join(data_dir, "esm_sequence_data_930.pkl")
    csv_path = os.path.join(data_dir, "pos_data930.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth sequences not found at {csv_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    masks = data["attention_masks"]
    
    df = pd.read_csv(csv_path)
    sequences = df['Sequence'].tolist()
    return embeddings, masks, sequences

def calculate_accuracy(decoded_seqs, original_seqs):
    correct_chars, total_chars, exact_matches = 0, 0, 0
    for decoded, original in zip(decoded_seqs, original_seqs):
        if decoded == original:
            exact_matches += 1
        common_len = min(len(decoded), len(original))
        for i in range(common_len):
            if decoded[i] == original[i]:
                correct_chars += 1
        total_chars += len(original)
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    seq_acc = (exact_matches / len(original_seqs)) * 100 if original_seqs else 0
    return char_acc, seq_acc

if __name__ == "__main__":
    EPOCHS = 20
    LR = 1e-3
    BATCH_SIZE = 32
    from utils import set_seed 
    set_seed(seed)
    lm_head, alphabet, aa_ids, device = load_lm_head()
    embeddings, masks, sequences = load_pos30_data("data2")
    cal_model = CalibrationModel(d_model=1280).to(device)
    cal_model.eval()
    eval_indices = torch.arange(min(BATCH_SIZE, len(sequences)))
    eval_embeddings = embeddings[eval_indices].to(device)
    eval_masks = masks[eval_indices].to(device)
    eval_sequences = [sequences[i] for i in eval_indices]
    decoded_before = decode_with_lm_head(eval_embeddings, eval_masks, lm_head, aa_ids)
    char_acc_before, seq_acc_before = calculate_accuracy(decoded_before, eval_sequences)
    dataset = TensorDataset(embeddings, masks, torch.arange(len(sequences)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(cal_model.parameters(), lr=LR)
    cal_model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for H_batch, M_batch, indices_batch in loader:
            H_batch = H_batch.to(device)
            M_batch = M_batch.to(device)
            target_seqs_batch = [sequences[i] for i in indices_batch]
            optimizer.zero_grad()
            loss = loss_fn(H_batch, target_seqs_batch, M_batch, cal_model, lm_head, aa_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    cal_model.eval()
    decoded_after = decode_with_lm_head(eval_embeddings, eval_masks, lm_head, aa_ids, cal_model)
    char_acc_after, seq_acc_after = calculate_accuracy(decoded_after, eval_sequences)

    save_dir = 'esmflow/model_out'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'calibration_model.pt')
    torch.save(cal_model.state_dict(), save_path)