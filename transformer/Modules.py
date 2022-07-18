import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            #attn中按照mask矩阵取值进行掩码操作，即attn张量中对应索引在mask中取0，则attn中取-1e9
            attn = attn.masked_fill(mask == 0, -1e9)
            #此处attn.shape=(sz_b, n_head, len_q, len_q),mask.shape=(sz_b, 1, len_q, len_q),通过广播语义，
            #mask.shape=>(sz_b, n_head, len_q, len_q)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
