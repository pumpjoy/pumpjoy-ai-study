# D_train_gpt2_cheap.py
# This version is for normies who have simple GPU like me :').
# No grad_accum 
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

def get_dataset():
  import urllib.request
  url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  with urllib.request.urlopen(url) as response:
      raw_bytes = response.read()
      text = raw_bytes.decode('utf-8')
      # Equivalent of word count (wc third party library)
      print("Lines: "+ str(len(text.splitlines())), '| Words: ' + str(len(text.split())), '| Bytes: '+ str(len(raw_bytes)))
      # This file only has ASCII characters
  return text


class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu   = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # Flag for scaling init 

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

# class CausalSelfAttention(nn.Module):
# Remove the 4 attention lines
# GPT2: scale down at c_proj
class CausalSelfAttention(nn.Module):
  # Multi-Headed Attention

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # Key, Query, Value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # Output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # Flag for scaling init 
    
    # Regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    # Not really a "bias", closer to a mask
    # Following OpenAI/HF naming
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                  .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    # Batch size, sequence length, embedding dimensionality(n_embd)
    B, T, C = x.size()
    # Calculate Key, Query, Value for all heads in batch
    # and move head forward to be batch
    # nh is "Number of heads", hs is "head size", C (number of channels) = nh * ns
    # E.g. GPT-2 (124M), n_heads=12, hs=64, so nh*hs=C=768 channels in Transformer
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
    # New: Flash Attention: Remove these
    # Attention (materializes the large (T, T) matrix for all the queries and keys)
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # Consider tokens before, not after it
    # att = F.softmax(att, dim=-1) # Sum = 1
    # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side
    # Output projection
    y = self.c_proj(y)
    return y

class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config) # Reduction
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config) # Mapping

  def forward(self, x):
    # Residual channel preferably clean
    # from supervision to token input.
    # Gradient flow without normalization in between.
    # From optimization perspective.
    # This is pre-normalization layer. Feed-Forward Network.
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    # Layer Norm is after Attention
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024 # Max sequence length
  vocab_size: int = 50257 # No. tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
  n_layer: int = 12 # Number of layers
  n_head: int = 12 # Number of heads
  n_embd: int = 768 # Embedding dimensions

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd), # Output Embedding
        wpe = nn.Embedding(config.block_size, config.n_embd), # Positional Embedding
        # As seen in the pretrained, h layers can be indexed to create 12 layers
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Nx in image
        ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    # Weight Sharing Scheme
    self.transformer.wte.weight = self.lm_head.weight
    
    # Init param
    self.apply(self._init_weights)
    
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        # Note: that std=0.02 because d_head in GPT2 roughly results the same - Check back to initialization notebook
        torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias) # module.bias is defaulted at uniform, not zero
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  @classmethod
  def from_pretrained(cls, model_type):
    """ Loads pretrained GPT-2 model weights from HuggingFace """
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # Always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # Always 1024 for GPT model checkpoints
    # Create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Discard this mask / buffer, not a param

    # Init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # Copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # Ignore, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Ignore, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # Basically OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear.
    # This means that we have to transpose these weights when we import them.
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # Special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # Vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model
  
  def configure_optimizers(self, weight_decay, learning_rate, device):
    # Start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. All weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    # Decay 2D tensors: Embeddings and matrices that participate in linear (matmul)
    # Don't decay 1D tensors: Biases, LayerNorms, Scales).
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameterss")
    print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Create AdamW optimizer and use the fused version if it is available
    import inspect
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"Using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

  def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # Forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embd)
    x = tok_emb + pos_emb # Broadcasting happening here
    # Forward the blocks of the transformer
    for block in self.transformer.h:
        x = block(x)
    # Forward the final layernorm and the classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # NOTE: cross-entropy uses 2-D
    return logits, loss

  def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # Forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embd)
    x = tok_emb + pos_emb # Broadcasting happening here
    # Forward the blocks of the transformer
    for block in self.transformer.h:
        x = block(x)
    # Forward the final layernorm and the classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # NOTE: cross-entropy uses 2-D
    return logits, loss
  

class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # At init, load tokens from disk and store in memory
        text = get_dataset()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches.")
        
        # State
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # Inputs
        y = (buf[1:]).view(B, T)  # Labels/Targets
        # Advance the position in tensor
        self.current_position += B * T
        # If loading next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -------------------------------------------------------------------
def main():
  # Distributed Data Parallels (DDP)
  import os
  from torch.distributed import init_process_group, destroy_process_group
  
  # Setup DDP
  # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
  ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run? # 
  if ddp:
      # Use of DDP atm demands CUDA, we set the device appropriately according to rank
      assert torch.cuda.is_available(), "For now I think we need CUDA for DDP"
      init_process_group(backend='nccl')
      ddp_rank = int(os.environ['RANK']) # GPU0, rank=0; GPU1, rank=1
      ddp_local_rank = int(os.environ['LOCAL_RANK']) # For multi-node settings (This model is 1 node)
      ddp_world_size = int(os.environ['WORLD_SIZE']) # Stack of 8 GPU, world_size = 8
      device = f'cuda:{ddp_local_rank}'
      torch.cuda.set_device(device)
      master_process = ddp_rank == 0 # This process will do logging, checkpointing, etc.
  else:
      # Vanilla, non-DDP run
      ddp_rank = 0
      ddp_local_rank = 0
      ddp_world_size = 1
      master_process = True
      # Attempt to autodetect device
      device = "cpu"
      if torch.cuda.is_available():
          device = "cuda"
      elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
          device = "mps"
      print(f"Using device: {device}")
  
  torch.manual_seed(1337)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
  
  # Dataset  
  B = 16 # Micro batch size
  T = 32 # Sequence/Context Length 
  # train_loader = DataLoaderLite(B=16, T=32)
  train_loader = DataLoaderLite(B=B, T=T)
  
  # Use T32 Format
  torch.set_float32_matmul_precision('high')
  
  # Model, get logits
  model = GPT(GPTConfig(vocab_size=50304))
  model.to(device) 
  model = torch.compile(model)
  
  # Optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  
  max_lr = 3e-4
  min_lr = max_lr * 0.1
  warmup_steps = 10
  max_steps = 50
  
  # Scheduler
  def get_lr(it):
      # 1) Linear warmup for warmup_iters steps
      if it < warmup_steps:
          return max_lr * (it+1) / warmup_steps
      # 2) If it > lr_decay_iters, return min learning rate
      if it > max_steps:
          return min_lr
      # 3) In between, use cosine decay don to min learning rate
      decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
      assert 0 <= decay_ratio <= 1
      coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff starts at 1 and goes to 0
      return min_lr + coeff * (max_lr - min_lr) 
  
  import time
  for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # <---- This line
        logits, loss = model(x, y) 
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # Time difference in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"Step {step+1:4d} | Loss: {loss.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | Dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
  
if __name__ == '__main__':
  main()