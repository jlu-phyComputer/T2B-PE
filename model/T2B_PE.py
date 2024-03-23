import torch
import torch.nn as nn
from layers.Transformer_EncDec import (EncoderLayer_cross_tv, Encoder_cross_t_v_duli_KL)
from layers.SelfAttention_Family import (FullAttention_with_pos_emb, AttentionLayer_with_enhanced_emb)
from layers.Embed import DataEmbedding_cross_var_time


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        if configs.data in ["PEMS", "Solar"]:
            self.enc_embedding = DataEmbedding_cross_var_time(configs.enc_in, configs.seq_len,
                                                              configs.d_model, configs.embed,
                                                              configs.freq, configs.dropout, configs=configs)
        else:
            self.enc_embedding = DataEmbedding_cross_var_time(configs.enc_in + 4, configs.seq_len,
                                                              configs.d_model, configs.embed,
                                                              configs.freq, configs.dropout, configs=configs)
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder_cross_t_v_duli_KL(
            [
                EncoderLayer_cross_tv(
                    AttentionLayer_with_enhanced_emb(
                        FullAttention_with_pos_emb(False, configs.factor, attention_dropout=configs.dropout,
                                                   output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            [
                EncoderLayer_cross_tv(
                    AttentionLayer_with_enhanced_emb(
                        FullAttention_with_pos_emb(False, configs.factor, attention_dropout=configs.dropout,
                                                   output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            configs,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            emb_layer=self.enc_embedding
        )
        self.projector = nn.Linear(configs.d_model, configs.enc_in, bias=True)
        self.projector_input = nn.Linear(self.seq_len, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # # Normalization from Non-stationary Transformer
        # x_enc_init = x_enc.clone()
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # x_enc1 = x_enc.clone()
        # enc_out_c_var = self.enc_embedding(x_enc, x_mark_enc, flag="cross_var")  # covariates (e.g timestamp) can be also embedded as tokens
        # enc_out_c_time = self.enc_embedding(x_enc1, x_mark_enc, flag="cross_time")

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        (dec_out, dec_out_c_v, dec_out_c_t,
         KL_list) = self.encoder(x_enc, x_mark_enc=x_mark_enc, attn_mask=None)

        # B N E -> B N S -> B S N 
        # dec_out = self.projector(enc_out)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_c_v = dec_out_c_v * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_c_t = dec_out_c_t + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # new
        # res_x_enc = x_enc_init.permute(0, 2, 1)
        # res_x_enc = self.projector_input(res_x_enc).permute(0, 2, 1)
        # dec_out = dec_out + x_enc_init
        return [dec_out, dec_out_c_v, dec_out_c_t, KL_list]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # return [dec_out[0][:, -self.pred_len:, :],
        #         dec_out[1][:, -self.pred_len:, :],
        #         dec_out[2][:, -self.pred_len:, :]]  # [B, L, D]
        # return dec_out[0][:, -self.pred_len:, :]
        if self.configs.output_attention:
            return [dec_out[0][:, -self.pred_len:, :], dec_out[-1]]
        else:
            return dec_out[0][:, -self.pred_len:, :]
