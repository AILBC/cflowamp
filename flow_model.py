import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlowMatching:
   
    def create_flow(self, H_real, t, H_0=None):
       
        if H_0 is None:
            H_0 = torch.randn_like(H_real)
        t_reshaped = t.view(-1, 1, 1)
        H_t = (1 - t_reshaped) * H_0 + t_reshaped * H_real
        return H_t, H_0

    def mse_loss(self, v_pred, H_real, H_0, mask=None):
        v_target = H_real - H_0
        loss = F.mse_loss(v_pred, v_target, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.unsqueeze(-1)
            num_active_elements = mask.sum() * v_pred.shape[-1]
            loss = masked_loss.sum() / (num_active_elements + 1e-8)
        else:
            loss = loss.mean()
        return loss

    def sample_heun(self, model, num_samples, shape, device, condition=None, num_steps=100, guidance_scale=3.0):
        model.eval()
        H_t = torch.randn(num_samples, *shape, device=device)
        if condition is not None:
            condition = condition.to(device)  
        dt = 1.0 / num_steps
        with torch.no_grad():
            for i in range(num_steps):
                t_n = torch.ones(num_samples, device=device) * (i * dt)
                v_pred_uncond = model(H_t=H_t, t=t_n, condition=None)
                v_pred_cond = model(H_t=H_t, t=t_n, condition=condition)
                v_pred_n = v_pred_uncond + guidance_scale * (v_pred_cond - v_pred_uncond)
                H_tilde_np1 = H_t + dt * v_pred_n
                t_np1 = torch.ones(num_samples, device=device) * ((i + 1) * dt)
                v_pred_uncond_np1 = model(H_t=H_tilde_np1, t=t_np1, condition=None)
                v_pred_cond_np1 = model(H_t=H_tilde_np1, t=t_np1, condition=condition)
                v_pred_np1 = v_pred_uncond_np1 + guidance_scale * (v_pred_cond_np1 - v_pred_uncond_np1)
                H_t = H_t + (dt / 2.0) * (v_pred_n + v_pred_np1)
        return H_t

class TimeStepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        self.register_buffer(
            "freqs", 
            torch.exp(torch.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
        )

    def forward(self, t):
        args = t.unsqueeze(-1) * self.freqs  
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding 

class FiLMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)    
        self.film_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2)
        )
        self.film_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, src, time_emb):
        gamma_beta_attn = self.film_attn(time_emb)  
        gamma_attn, beta_attn = torch.chunk(gamma_beta_attn, 2, dim=-1) 
        src_modulated = gamma_attn.unsqueeze(1) * src + beta_attn.unsqueeze(1)
        attn_output, _ = self.self_attn(src_modulated, src_modulated, src_modulated)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        gamma_beta_ffn = self.film_ffn(time_emb)  
        gamma_ffn, beta_ffn = torch.chunk(gamma_beta_ffn, 2, dim=-1) 
        src_modulated = gamma_ffn.unsqueeze(1) * src + beta_ffn.unsqueeze(1)
        ffn_output = self.linear2(self.dropout(F.gelu(self.linear1(src_modulated))))
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)
        return src

class FlowMatchingTransformer(nn.Module):
    def __init__(self, d_model=1280, n_head=8, num_layers=16, max_len=30, dropout_p=0.1, num_conditions=3):
        super().__init__()    
        self.time_embed = TimeStepEmbedding(d_model)     
        self.condition_proj = nn.Linear(num_conditions, d_model)      
        total_len = max_len + 2
        self.pos_embed = nn.Parameter(torch.randn(1, total_len, d_model))       
        self.input_proj = nn.Linear(d_model, d_model)       
        self.layers = nn.ModuleList([
            FiLMTransformerEncoderLayer(
                d_model=d_model, 
                n_head=n_head, 
                dim_feedforward=d_model * 4, 
                dropout=dropout_p
            ) for _ in range(num_layers)
        ])       
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)      
        self.uncond_embedding = nn.Parameter(torch.randn(1, d_model))     
        torch.nn.init.constant_(self.output_proj.weight, 0)
        torch.nn.init.constant_(self.output_proj.bias, 0)
        
    def forward(self, H_t, t, condition=None):
        x = self.input_proj(H_t)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        time_embedding = self.time_embed(t)
        if condition is not None:
            condition_embedding = self.condition_proj(condition)
            combined_embedding = time_embedding + condition_embedding
        else:
            uncond_emb_expanded = self.uncond_embedding.expand(t.size(0), -1)
            combined_embedding = time_embedding + uncond_emb_expanded
        for layer in self.layers:
            x = layer(x, combined_embedding)  
        x = self.final_norm(x)
        v_pred = self.output_proj(x)
        
        return v_pred


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    max_len = 30
    d_model = 1280
    num_samples = 2 
    model = FlowMatchingTransformer(d_model=d_model, max_len=max_len).to(device)
    fm = FlowMatching()
    H_real = torch.randn(batch_size, max_len, d_model, device=device)
    t = torch.rand(batch_size, device=device)
    mask = torch.ones(batch_size, max_len, device=device)
    H_t, H_0 = fm.create_flow(H_real, t)   
    t_near_1 = torch.full((batch_size,), 0.999, device=device)
    H_t_near_1, _ = fm.create_flow(H_real, t_near_1)
    diff = torch.mean((H_real - H_t_near_1)**2)
    v_pred = model(H_t, t)
    loss = fm.mse_loss(v_pred, H_real, H_0, mask=mask)
    generated_samples = fm.sample_heun(model, num_samples, (max_len, d_model), device)
    