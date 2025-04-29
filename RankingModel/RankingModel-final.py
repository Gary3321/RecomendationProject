import torch 
import torch.nn as nn
from collections import defaultdict

PAD_ID = "<PAD>"
class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, df_data, user_last5, product_emb_dict, user_emb_dict):
        """
        df_data: training set or test set, containing user_id, four labels and product information
        user_last5: dict user_id→list of 5 item_ids (with PAD_ID)
        product_emb_dict: item_id→embedding vector (D-dim). Must include PAD_ID→zero vector
        user_emb_dict: user_id → embedding vector (D-dim)
        user_ids & labels: parallel lists defining samples.      
        """
        self.df = df_data
        self.user_last5 = user_last5
        self.prod_emb = product_emb_dict
        self.user_emb = user_emb_dict

    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        u = self.df.loc[idx, 'user_id']
        
        i = self.df.loc[idx, 'parent_asin']
        labels = self.df.loc[idx, ['verified_purchase', 'clicked', 'add_to_cart', 'favorite']].astype(int).astype(str).tolist()

        # Sequence embeddings (last 5 items)
        items = self.user_last5[u] # length 5

        # building embeddings and mask
        embs = []
        mask = []
        for item in items:
            if item == PAD_ID:
                embs.append(self.prod_emb[PAD_ID])
                mask.append(True) # True = This position is PAD
            else:
                embs.append(self.prod_emb[item])
                mask.append(False) # False = real product
        
        item_embs = torch.tensor(embs, dtype=torch.float32) #[5, D]: last 5 item
        item_padding_mask = torch.tensor(mask) #(S, ) will be broadcast to (B, S); B is batch size, S is sequence length which is 5 here
        
        # User embedding
        user_emb = torch.tensor(self.user_emb[u], dtype=torch.float32)

        # Product embedding
        item_emb = torch.tensor(self.prod_emb[i], dtype=torch.float32)

        # Label
        y = torch.tensor(labels, dtype=torch.float32)

        return item_embs, item_padding_mask, user_emb, item_emb, y


# ## Multi-Head Attention on Last-5 items
# class MultiHeadSeqAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=2):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, item_seq_emb):
#         # Self-attention over sequence
#         attn_output, _ = self.attn(item_seq_emb, item_seq_emb, item_seq_emb)
#         return attn_output.mean(dim=1)  # average pooling over time
    
## Deal with the last 5 items using PositionalEmbedding and SequenceEncoder
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, dim)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.size()
        pos_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)  # [B, S]
        return x + self.pos_emb(pos_ids)

class SequenceEncoder(nn.Module):
    def __init__(self, emb_dim, seq_len=5, num_heads=2, hidden_dim=128):
        super().__init__()
        self.pos_emb = PositionalEmbedding(seq_len, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, seq_embs, key_padding_mask):
        # seq_embs: [B, S, D], key_padding_mask: [B, S]
        x = self.pos_emb(seq_embs)  # Automatically adds position info

        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)

        # mask-aware mean pooling
        mask_inv = ~key_padding_mask  # [B, S]
        mask_inv = mask_inv.unsqueeze(-1)  # [B, S, 1]
        pooled = (attn_out * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)

        return self.fc(pooled)  # [B, D]



# PLE Model
class SharedBottom(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )

    def forward(self, x):
        return self.shared(x)

class TaskTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.tower(x))



### Cross layer (for Two-tower or Wide & Deep)
class CrossLayer(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.cross_weights = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.cross_weights, self.cross_biases):
            xw = w(x)
            x = x0 * xw + b + x
        return x

### Feature Embedding
class EmbeddingLayer(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, category_vocab_size, embedding_dim):
        super().__init__()
        self.user_emb = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_emb = nn.Embedding(item_vocab_size, embedding_dim)
        self.category_emb = nn.Embedding(category_vocab_size, embedding_dim)

    def forward(self, user_ids, item_ids, category_ids):
        return self.user_emb(user_ids), self.item_emb(item_ids), self.category_emb(category_ids)

# ### Get last N items
# def generate_last_n_sequences(df, N=5):
#     user_sequences = defaultdict(list)
#     sequences = []

#     df = df.sort_values(by=['user_id', 'timestamp'])

#     for _, row in df.iterrows():
#         user = row['user_id']
#         item = row['item_id']
#         label = row['label']

#         history = user_sequences[user][-N:]
#         sequences.append((user, item, history, label))

#         user_sequences[user].append(item)

#     return sequences

## Final model
class RecSysModel(nn.Module):
    def __init__(self, user_vocab, item_vocab, cat_vocab, embed_dim):
        super().__init__()
        self.emb_layer = EmbeddingLayer(user_vocab, item_vocab, cat_vocab, embed_dim)
        self.cross = CrossLayer(embed_dim)
        self.seq_attention = MultiHeadSeqAttention(embed_dim)

        self.shared_bottom = SharedBottom(embed_dim * 4, 128)
        self.purchase_tower = TaskTower(128)
        self.click_tower = TaskTower(128)
        self.cart_tower = TaskTower(128)
        self.favorite_tower = TaskTower(128)

    def forward(self, user_id, item_id, category_id, last_n_item_ids):
        user_emb, item_emb, cat_emb = self.emb_layer(user_id, item_id, category_id)
        last_item_embs = self.emb_layer.item_emb(last_n_item_ids)
        seq_emb = self.seq_attention(last_item_embs)

        features = torch.cat([user_emb, item_emb, cat_emb, seq_emb], dim=-1)
        crossed = self.cross(features)
        shared_out = self.shared_bottom(crossed)

        return {
            "purchase": self.purchase_tower(shared_out),
            "click": self.click_tower(shared_out),
            "cart": self.cart_tower(shared_out),
            "favorite": self.favorite_tower(shared_out),
        }