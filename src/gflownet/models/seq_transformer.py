import math

import torch
import torch.nn as nn

from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnvContext
from gflownet.envs.seq_building_env import SeqBatch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()]
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)


class SeqTransformerGFN(nn.Module):
    ctx: GraphBuildingEnvContext

    def __init__(
        self,
        env_ctx,
        cfg: Config,
        num_state_out=1,
    ):
        super().__init__()
        # num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, num_layers, num_head, use_cond, **kwargs
        self.ctx = env_ctx
        num_hid = cfg.model.num_emb
        num_outs = env_ctx.num_outputs + num_state_out
        mc = cfg.model
        self.pos = PositionalEncoding(num_hid, dropout=cfg.model.dropout, max_len=cfg.algo.max_len + 2)
        self.use_cond = env_ctx.num_cond_dim > 0
        self.embedding = nn.Embedding(env_ctx.num_tokens, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(
            num_hid, mc.graph_transformer.num_heads, num_hid, dropout=mc.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, mc.num_layers)
        self.logZ = nn.Linear(env_ctx.num_cond_dim, 1)
        if self.use_cond:
            self.output = MLP(num_hid + num_hid, num_outs, [4 * num_hid, 4 * num_hid], mc.dropout)
            self.cond_embed = nn.Linear(env_ctx.num_cond_dim, num_hid)
        else:
            self.output = MLP(num_hid, num_outs, [2 * num_hid, 2 * num_hid], mc.dropout)
        self.num_hid = num_hid

    def forward(self, xs: SeqBatch, cond, batched=False):
        x = self.embedding(xs.x)
        x = self.pos(x)  # (time, batch, nemb)
        x = self.encoder(x, src_key_padding_mask=xs.mask, mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
        pooled_x = x[xs.lens - 1, torch.arange(x.shape[1])]  # (batch, nemb)

        if self.use_cond:
            cond_var = self.cond_embed(cond)  # (batch, nemb)
            cond_var = torch.tile(cond_var, (x.shape[0], 1, 1)) if batched else cond_var
            final_rep = torch.cat((x, cond_var), axis=-1) if batched else torch.cat((pooled_x, cond_var), axis=-1)
        else:
            final_rep = x if batched else pooled_x

        out: torch.Tensor = self.output(final_rep)
        if batched:
            # out is (time, batch, nout)
            out = out.transpose(1, 0).contiguous().reshape((-1, out.shape[2]))  # (batch * time, nout)
            stop_logits = out[xs.logit_idx, 0:1]  # (proper_time, 1)
            state_preds = out[xs.logit_idx, 1:2]  # (proper_time, 1)
            add_node_logits = out[xs.logit_idx, 2:]  # (proper_time, nout - 1)
            # `time` above is really max_time, whereas proper_time = sum(len(traj) for traj in xs))
            # which is what we need to give to GraphActionCategorical
        else:
            # The default num_graphs is computed for the batched case, so we need to fix it here so that
            # GraphActionCategorical knows how many "graphs" (sequence inputs) there are
            xs.num_graphs = out.shape[0]
            # out is (batch, nout)
            stop_logits = out[:, 0:1]
            state_preds = out[:, 1:2]
            add_node_logits = out[:, 2:]

        return (
            GraphActionCategorical(
                xs,
                logits=[stop_logits, add_node_logits],
                keys=[None, None],
                types=self.ctx.action_type_order,
                slice_dict={},
            ),
            state_preds,
        )


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
