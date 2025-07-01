import torch
import torch.nn as nn

class FrequencyTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, ffn_hidden=128, dropout=0.1):
        super(FrequencyTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_hidden,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, F) → (B, F, C)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class MidFrequencyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MidFrequencyMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, C, F) → (B, F, C)
        #x = x.permute(0, 2, 1)

        attn_output, _ = self.attn(x, x, x)  # self-attention
        x = self.norm(x + self.dropout(attn_output))  # res + LN

        return x
