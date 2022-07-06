import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def calc_scores(dk):
    def fn(q, k):
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    return fn


class Attention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        query_size=None,
        dropout=0.1,
        grad_checkpoint=False,
    ):
        super(Attention, self).__init__()

        if input_size is None:
            input_size = attention_size
        if query_size is None:
            query_size = input_size

        self.dk = input_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(query_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x

        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)

        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)

        return out, scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class SymmetricAttention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        dropout=0.1,
    ):
        super(SymmetricAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.kx = nn.Linear(input_size, attention_size, bias=False)
        self.qx = nn.Linear(input_size, attention_size, bias=False)
        self.vx = nn.Linear(input_size, attention_size, bias=False)
        self.ky = nn.Linear(input_size, attention_size, bias=False)
        self.qy = nn.Linear(input_size, attention_size, bias=False)
        self.vy = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def forward(self, mod1, mod2, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        k_mod1 = self.kx(mod1)
        q_mod2 = self.qy(mod2)
        v_mod1 = self.vx(mod1)

        k_mod2 = self.ky(mod2)  # (B, L, A)
        q_mod1 = self.qx(mod1)
        v_mod2 = self.vy(mod2)

        # weights => (B, L, L)

        scores_mod1 = torch.bmm(q_mod2, k_mod1.transpose(1, 2)) / math.sqrt(self.dk)
        scores_mod2 = torch.bmm(q_mod1, k_mod2.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores_mod1 = scores_mod1 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
            scores_mod2 = scores_mod2 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores_mod1 = F.softmax(scores_mod1, dim=-1)
        scores_mod1 = self.drop(scores_mod1)
        scores_mod2 = F.softmax(scores_mod2, dim=-1)
        scores_mod2 = self.drop(scores_mod2)

        # out => (B, L, A)
        out_mod1 = torch.bmm(scores_mod1, v_mod1)
        out_mod2 = torch.bmm(scores_mod2, v_mod2)

        # vilbert cross residual

        # v + attention(v->a)
        # a + attention(a->v)
        out_mod1 += mod2
        out_mod2 += mod1
        return out_mod1, out_mod2

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kx.weight)
        nn.init.xavier_uniform_(self.qx.weight)
        nn.init.xavier_uniform_(self.vx.weight)
        nn.init.xavier_uniform_(self.ky.weight)
        nn.init.xavier_uniform_(self.qy.weight)
        nn.init.xavier_uniform_(self.vy.weight)
