import os
import pickle
import pandas as pd
import torch
import esm
from tqdm import tqdm
from utils import set_seed


def encode_pos_data30(csv_file_path: str, output_dir: str, total_len: int = 32):
    df = pd.read_csv(csv_file_path)
    sequences = df['Sequence'].astype(str).tolist()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    def build_tokens_and_masks(seq_list):
        tokens_batch = []
        masks_batch = []
        for s in seq_list:
            aa_ids = [alphabet.get_idx(aa) for aa in s]
            token_ids = [alphabet.cls_idx] + aa_ids + [alphabet.eos_idx]
            if len(token_ids) < total_len:
                token_ids += [alphabet.padding_idx] * (total_len - len(token_ids))
            elif len(token_ids) > total_len:
                raise ValueError(
                    f"Sequence length {len(token_ids)} exceeds total_len {total_len}. Consider increasing total_len or truncating sequences."
                )
            tokens_batch.append(token_ids)
            mask = [1 if tid != alphabet.padding_idx else 0 for tid in token_ids]
            masks_batch.append(mask)
        tokens_tensor = torch.tensor(tokens_batch, dtype=torch.long)
        masks_tensor = torch.tensor(masks_batch, dtype=torch.float32)
        return tokens_tensor, masks_tensor
    batch_size = 16
    all_embeddings = []
    all_attention_masks = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_strs = sequences[i:i + batch_size]
        batch_tokens, batch_masks = build_tokens_and_masks(batch_strs)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        last_hidden_state = results["representations"][33].cpu()  
        attention_mask = batch_masks.cpu()                        
        all_embeddings.append(last_hidden_state)
        all_attention_masks.append(attention_mask)
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_attention_masks = torch.cat(all_attention_masks, dim=0)
    out_emb = os.path.join(output_dir, 'esm_sequence_embeddings_930.pt')
    out_mask = os.path.join(output_dir, 'esm_attention_masks_930.pt')
    out_pkl = os.path.join(output_dir, 'esm_sequence_data_930.pkl')
    torch.save(final_embeddings, out_emb)
    torch.save(final_attention_masks, out_mask)
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'embeddings': final_embeddings,
            'attention_masks': final_attention_masks,
            'alphabet': alphabet,
        }, f)
    return final_embeddings, final_attention_masks, alphabet


if __name__ == '__main__':
    set_seed(seed)
    data_dir = 'data2/'
    output_dir = 'data2/'
    encode_pos_data30(os.path.join(data_dir, 'pos_data930.csv'), output_dir, total_len=32)
