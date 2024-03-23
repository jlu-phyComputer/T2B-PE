import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
from scipy.stats import wasserstein_distance
from utils.utils import compute_tensor_cos_sim


class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :]  # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2)  # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos))  # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


def conv_3x3(inp, oup, groups=1):
    return nn.Conv2d(inp, oup, (3, 3), (1, 1), (1, 1), groups=groups)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, stride=1, padding=1, groups=d_model)
        # self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
        #                            kernel_size=25, stride=1, padding=12, groups=d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None, PE=None):
        # # use DPE
        # x = x.permute(0, 2, 1)
        # x = x + self.pos_embed(x)
        # x = x.permute(0, 2, 1)

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, PE=PE
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_cross_tv(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_cross_tv, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, stride=1, padding=1, groups=d_model)
        # self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
        #                            kernel_size=866, stride=1, padding=433, groups=d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None, use_DPE=True, use_rot=True):
        if use_DPE:
            # use DPE
            x = x.permute(0, 2, 1)
            x = x + self.pos_embed(x)[:, :, :x.shape[-1]]
            x = x.permute(0, 2, 1)

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, use_rot=use_rot
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_cross_tv_supply_yuyi(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_cross_tv_supply_yuyi, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, stride=1, padding=1, groups=d_model)
        # self.pos_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model,
        #                            kernel_size=866, stride=1, padding=433, groups=d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None, use_DPE=True,
                use_rot=True, init_score=None):
        if use_DPE:
            # use DPE
            x = x.permute(0, 2, 1)
            x = x + self.pos_embed(x)[:, :, :x.shape[-1]]
            x = x.permute(0, 2, 1)

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, use_rot=use_rot, init_score=init_score
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_my(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_my, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.pos_embed = conv_3x3(d_model, d_model, groups=d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        b, v, t, c = x.shape
        # 加入DPE编码
        x = x.permute(0, 3, 1, 2)
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b * v, t, c)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)
        x = y
        x = x.reshape(b, v, t, c)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b * t, v, c)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1_2(x)
        y = self.dropout(self.activation(self.conv1_2(y.transpose(-1, 1))))
        y = self.dropout(self.conv2_2(y).transpose(-1, 1))
        y = self.norm2_2(x + y)
        y = y.reshape(b, t, v, c).permute(0, 2, 1, 3)
        return y, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, PE=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, PE=PE)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, PE=PE)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Encoder_cross_t_v(nn.Module):
    def __init__(self, attn_layers, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v, self).__init__()
        self.configs = configs
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer
        # self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in+4)
        if configs.data in ["PEMS", "Solar"]:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in + 4)
        self.de_emb_layer_c_v1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.de_emb_layer_c_t = nn.Linear(configs.d_model, configs.pred_len)
        # self.w_att = nn.Linear(512*2, 512)
        self.w_att = nn.Linear(configs.pred_len * 2, configs.pred_len)
        # self.w_att = nn.Linear((configs.enc_in+4) * 2, configs.enc_in+4)
        self.st_emb_layer = nn.Linear(1, configs.d_model)
        self.st_de_emb_layer = nn.Linear(configs.d_model, 1)

        # self.media_decode_c_v = nn.Linear(configs.enc_in+4, configs.enc_in+4)
        if configs.data in ["PEMS", "Solar"]:
            self.media_decode_c_v = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.media_decode_c_v = nn.Linear(configs.enc_in + 4, configs.enc_in + 4)
        self.media_decode_c_t = nn.Linear(configs.pred_len, configs.pred_len)

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if x_mark_enc is not None:
            x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        for attn_layer in self.attn_layers:
            enc_out_c_var, attn = attn_layer(enc_out_c_var,
                                             attn_mask=attn_mask,
                                             tau=tau, delta=delta,
                                             use_DPE=False, use_rot=True)
            # x_c_t, attn = attn_layer(enc_out_c_time, attn_mask=attn_mask, tau=tau, delta=delta)
            # x_c_v = self.de_emb_layer_c_v(x_c_v)
            # x_c_t = self.de_emb_layer_c_t(x_c_t).permute(0, 2, 1)
            # f = torch.cat([x_c_t, x_c_v], dim=-1)
            # f_att = torch.sigmoid(self.w_att(f))
            # x = f_att * x_c_v + (1 - f_att) * x_c_t
            # attns.append(attn)
        for attn_layer in self.attn_layers:
            enc_out_c_time, attn = attn_layer(enc_out_c_time,
                                              attn_mask=attn_mask,
                                              tau=tau, delta=delta,
                                              use_DPE=True, use_rot=False)
        # x = self.emb_layer(x, x_mark_enc, flag="cross_var")
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        # x_c_v = self.de_emb_layer_c_v1(enc_out_c_var.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = self.de_emb_layer_c_v(enc_out_c_var)
        x_c_v = self.de_emb_layer_c_v1(x_c_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_t = self.de_emb_layer_c_t(enc_out_c_time).permute(0, 2, 1)
        # # 使用TV维度进行融合
        # x_c_t = x_c_t.unsqueeze(-1)
        # x_c_v = x_c_v.unsqueeze(-1)
        # x_c_t = self.st_emb_layer(x_c_t)
        # x_c_v = self.st_emb_layer(x_c_v)
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = self.st_de_emb_layer(x).squeeze(-1)
        # 使用T维度进行融合
        f = torch.cat([x_c_t.permute(0, 2, 1), x_c_v.permute(0, 2, 1)], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        f_att = f_att.permute(0, 2, 1)
        x = f_att * x_c_v + (1 - f_att) * x_c_t

        # # 使用V维度进行融合
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # f_att = f_att
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # # x = x.permute(0, 2, 1)
        # # if self.norm is not None:
        # #     x = self.norm(x)
        x = x[:, :, :self.configs.enc_in]
        x_c_v = self.media_decode_c_v(x_c_v)
        x_c_t = self.media_decode_c_t(x_c_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = x_c_v[:, :, :self.configs.enc_in]
        x_c_t = x_c_t[:, :, :self.configs.enc_in]
        return x, x_c_v, x_c_t, attns


class Encoder_cross_t_v_duli(nn.Module):
    def __init__(self, attn_layers1, attn_layers2, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v_duli, self).__init__()
        self.configs = configs
        self.attn_layers1 = nn.ModuleList(attn_layers1)
        self.attn_layers2 = nn.ModuleList(attn_layers2)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer
        if configs.data in ["PEMS", "Solar"]:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in + 4)
        self.de_emb_layer_c_v1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.de_emb_layer_c_t = nn.Linear(configs.d_model, configs.pred_len)
        # self.w_att = nn.Linear(512*2, 512)
        self.w_att = nn.Linear(configs.pred_len * 2, configs.pred_len)
        # self.w_att = nn.Linear((configs.enc_in+4) * 2, configs.enc_in+4)
        self.st_emb_layer = nn.Linear(1, configs.d_model)
        self.st_de_emb_layer = nn.Linear(configs.d_model, 1)
        if configs.data in ["PEMS", "Solar"]:
            self.media_decode_c_v = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.media_decode_c_v = nn.Linear(configs.enc_in + 4, configs.enc_in + 4)
        self.media_decode_c_t = nn.Linear(configs.pred_len, configs.pred_len)

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if x_mark_enc is not None:
            x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        for attn_layer in self.attn_layers1:
            enc_out_c_var, attn_c_var = attn_layer(enc_out_c_var,
                                                   attn_mask=attn_mask,
                                                   tau=tau, delta=delta,
                                                   use_DPE=False, use_rot=True)
            # x_c_t, attn = attn_layer(enc_out_c_time, attn_mask=attn_mask, tau=tau, delta=delta)
            # x_c_v = self.de_emb_layer_c_v(x_c_v)
            # x_c_t = self.de_emb_layer_c_t(x_c_t).permute(0, 2, 1)
            # f = torch.cat([x_c_t, x_c_v], dim=-1)
            # f_att = torch.sigmoid(self.w_att(f))
            # x = f_att * x_c_v + (1 - f_att) * x_c_t
            # attns.append(attn)
        for attn_layer in self.attn_layers2:
            enc_out_c_time, attn_c_time = attn_layer(enc_out_c_time,
                                                     attn_mask=attn_mask,
                                                     tau=tau, delta=delta,
                                                     use_DPE=True, use_rot=False)
        # x = self.emb_layer(x, x_mark_enc, flag="cross_var")
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        # x_c_v = self.de_emb_layer_c_v1(enc_out_c_var.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = self.de_emb_layer_c_v(enc_out_c_var)
        x_c_v = self.de_emb_layer_c_v1(x_c_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_t = self.de_emb_layer_c_t(enc_out_c_time).permute(0, 2, 1)
        # # 使用TV维度进行融合
        # x_c_t = x_c_t.unsqueeze(-1)
        # x_c_v = x_c_v.unsqueeze(-1)
        # x_c_t = self.st_emb_layer(x_c_t)
        # x_c_v = self.st_emb_layer(x_c_v)
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = self.st_de_emb_layer(x).squeeze(-1)
        # 使用T维度进行融合
        f = torch.cat([x_c_t.permute(0, 2, 1), x_c_v.permute(0, 2, 1)], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        f_att = f_att.permute(0, 2, 1)
        x = f_att * x_c_v + (1 - f_att) * x_c_t

        # # 使用V维度进行融合
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # f_att = f_att
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # # x = x.permute(0, 2, 1)
        # # if self.norm is not None:
        # #     x = self.norm(x)
        x = x[:, :, :self.configs.enc_in]
        x_c_v = self.media_decode_c_v(x_c_v)
        x_c_t = self.media_decode_c_t(x_c_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = x_c_v[:, :, :self.configs.enc_in]
        x_c_t = x_c_t[:, :, :self.configs.enc_in]
        init_score = torch.softmax(torch.matmul(x1, x1.permute(0, 2, 1)) /
                                   (x.shape[-1] ** 0.5), dim=-1)
        attns_diff = 0
        # 计算 L2 范数的平方
        if attn_c_var is not None:
            l2_norm_squared = torch.norm(init_score.unsqueeze(1) - attn_c_var, p=2) ** 2
        else:
            l2_norm_squared = None
        return x, x_c_v, x_c_t, l2_norm_squared


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # import time
        # t_all = time.time()
        # t = time.time()
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda()  # Wasserstein cost function
        # x = x.cpu()
        # y = y.cpu()
        # C = self._cost_matrix(x, y).cpu()  # Wasserstein cost function
        # t1 = time.time()
        # print("t1=", t1 - t)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # t2 = time.time()
        # print("t2=", t2 - t1)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        # t3 = time.time()
        # print("t3=", t3 - t2)
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        # t4 = time.time()
        # print("t4=", t4 - t3)
        # print("t_all=", t4 - t_all)
        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class Encoder_cross_t_v_duli_KL(nn.Module):
    def __init__(self, attn_layers1, attn_layers2, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v_duli_KL, self).__init__()
        self.configs = configs
        self.attn_layers1 = nn.ModuleList(attn_layers1)
        self.attn_layers2 = nn.ModuleList(attn_layers2)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer
        if configs.data in ["PEMS", "Solar"]:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in + 4)
        self.de_emb_layer_c_v1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.de_emb_layer_c_t = nn.Linear(configs.d_model, configs.pred_len)
        # self.w_att = nn.Linear(512*2, 512)
        self.w_att = nn.Linear(configs.pred_len * 2, configs.pred_len)
        # self.w_att = nn.Linear((configs.enc_in+4) * 2, configs.enc_in+4)
        self.st_emb_layer = nn.Linear(1, configs.d_model)
        self.st_de_emb_layer = nn.Linear(configs.d_model, 1)
        if configs.data in ["PEMS", "Solar"]:
            self.media_decode_c_v = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.media_decode_c_v = nn.Linear(configs.enc_in + 4, configs.enc_in + 4)
        self.media_decode_c_t = nn.Linear(configs.pred_len, configs.pred_len)
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
        self.w_distance = SinkhornDistance()

    def distance_loss(self, real_outputs, fake_outputs):
        real_outputs = real_outputs.reshape(-1, real_outputs.shape[-1])
        fake_outputs = fake_outputs.reshape(-1, fake_outputs.shape[-1])
        return torch.mean(torch.norm(real_outputs - fake_outputs, dim=1))
        # return torch.mean(torch.norm(real_outputs - fake_outputs, dim=1, p=1))

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if x_mark_enc is not None:
            x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        init_score_4KL = compute_tensor_cos_sim(enc_out_c_var)
        init_score_c_var = torch.softmax(
            torch.matmul(enc_out_c_var, enc_out_c_var.
                         permute(0, 2, 1)) /
            (enc_out_c_var.shape[-1] ** 0.5), dim=-1)
        init_score_c_t = torch.softmax(
            torch.matmul(enc_out_c_time, enc_out_c_time.
                         permute(0, 2, 1)) /
            (enc_out_c_time.shape[-1] ** 0.5), dim=-1)
        init_score_c_var = init_score_c_var.to(x.device)
        init_score_c_t = init_score_c_t.to(x.device)
        attn_c_var_list = []
        for attn_layer in self.attn_layers1:
            enc_out_c_var, attn_c_var = attn_layer(enc_out_c_var,
                                                   attn_mask=attn_mask,
                                                   tau=tau, delta=delta,
                                                   use_DPE=False, use_rot=True)
            attn_c_var_list.append(attn_c_var)
            # x_c_t, attn = attn_layer(enc_out_c_time, attn_mask=attn_mask, tau=tau, delta=delta)
            # x_c_v = self.de_emb_layer_c_v(x_c_v)
            # x_c_t = self.de_emb_layer_c_t(x_c_t).permute(0, 2, 1)
            # f = torch.cat([x_c_t, x_c_v], dim=-1)
            # f_att = torch.sigmoid(self.w_att(f))
            # x = f_att * x_c_v + (1 - f_att) * x_c_t
            # attns.append(attn)
        attn_c_time_list = []
        for attn_layer in self.attn_layers2:
            enc_out_c_time, attn_c_time = attn_layer(enc_out_c_time,
                                                     attn_mask=attn_mask,
                                                     tau=tau, delta=delta,
                                                     use_DPE=True, use_rot=False)
            attn_c_time_list.append(attn_c_time)
        # x = self.emb_layer(x, x_mark_enc, flag="cross_var")
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        # x_c_v = self.de_emb_layer_c_v1(enc_out_c_var.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = self.de_emb_layer_c_v(enc_out_c_var)
        x_c_v = self.de_emb_layer_c_v1(x_c_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_t = self.de_emb_layer_c_t(enc_out_c_time).permute(0, 2, 1)
        # x_c_t = x_c_t.unsqueeze(-1)
        # x_c_v = x_c_v.unsqueeze(-1)
        # x_c_t = self.st_emb_layer(x_c_t)
        # x_c_v = self.st_emb_layer(x_c_v)
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = self.st_de_emb_layer(x).squeeze(-1)
        f = torch.cat([x_c_t.permute(0, 2, 1), x_c_v.permute(0, 2, 1)], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        f_att = f_att.permute(0, 2, 1)
        x = f_att * x_c_v + (1 - f_att) * x_c_t

        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # f_att = f_att
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # # x = x.permute(0, 2, 1)
        # # if self.norm is not None:
        # #     x = self.norm(x)
        x = x[:, :, :self.configs.enc_in]
        x_c_v = self.media_decode_c_v(x_c_v)
        x_c_t = self.media_decode_c_t(x_c_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = x_c_v[:, :, :self.configs.enc_in]
        x_c_t = x_c_t[:, :, :self.configs.enc_in]
        attns_diff = 0
        import time
        if attn_c_var is not None:
            init_score_c_var = init_score_c_var.unsqueeze(1).repeat(1, 8, 1, 1)
            kl_loss_list = []
            # t = time.time()
            for i in range(len(attn_c_var_list)):
                # temp = attn_c_var_list[i].reshape(-1, attn_c_var_list[i].shape[-1])
                kl_loss = self.distance_loss(attn_c_var_list[i], init_score_c_var)
                # kl_loss = self.kl_criterion(torch.log(temp), init_score_c_var)
                kl_loss_list.append(kl_loss)
            # print(time.time() - t)
            init_score_c_t = init_score_c_t.unsqueeze(1).repeat(1, 8, 1, 1)
            for i in range(len(attn_c_time_list)):
                # temp = attn_c_time_list[i].reshape(-1, attn_c_time_list[i].shape[-1])
                kl_loss = self.distance_loss(attn_c_time_list[i], init_score_c_t)
                # kl_loss = self.kl_criterion(torch.log(temp), init_score_c_t)
                kl_loss_list.append(kl_loss)
        else:
            l2_norm_squared = None
        return x, x_c_v, x_c_t, kl_loss_list


class Encoder_cross_t_v_duli_insert_fe_res(nn.Module):
    def __init__(self, attn_layers1, attn_layers2, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v_duli_insert_fe_res, self).__init__()
        self.configs = configs
        self.attn_layers1 = nn.ModuleList(attn_layers1)
        self.attn_layers2 = nn.ModuleList(attn_layers2)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer
        if configs.data in ["PEMS", "Solar"]:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.de_emb_layer_c_v = nn.Linear(configs.d_model, configs.enc_in + 4)
        self.de_emb_layer_c_v1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.de_emb_layer_c_t = nn.Linear(configs.d_model, configs.pred_len)
        # self.w_att = nn.Linear(512*2, 512)
        self.w_att = nn.Linear(configs.pred_len * 2, configs.pred_len)
        # self.w_att = nn.Linear((configs.enc_in+4) * 2, configs.enc_in+4)
        self.st_emb_layer = nn.Linear(1, configs.d_model)
        self.st_de_emb_layer = nn.Linear(configs.d_model, 1)
        if configs.data in ["PEMS", "Solar"]:
            self.media_decode_c_v = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.media_decode_c_v = nn.Linear(configs.enc_in + 4, configs.enc_in + 4)
        self.media_decode_c_t = nn.Linear(configs.pred_len, configs.pred_len)

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None, init_score=None):
        # x [B, L, D]
        attns = []
        if x_mark_enc is not None:
            x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        init_score = torch.softmax(torch.matmul(enc_out_c_var, enc_out_c_var.permute(0, 2, 1)) /
                                   (enc_out_c_var.shape[-1] ** 0.5), dim=-1)
        init_score_c_t = torch.softmax(torch.matmul(enc_out_c_time, enc_out_c_time.permute(0, 2, 1)) /
                                   (enc_out_c_var.shape[-1] ** 0.5), dim=-1)
        for attn_layer in self.attn_layers1:
            enc_out_c_var, attn_c_var = attn_layer(enc_out_c_var,
                                                   attn_mask=attn_mask,
                                                   tau=tau, delta=delta,
                                                   use_DPE=False, use_rot=True,
                                                   init_score=init_score)
            # x_c_t, attn = attn_layer(enc_out_c_time, attn_mask=attn_mask, tau=tau, delta=delta)
            # x_c_v = self.de_emb_layer_c_v(x_c_v)
            # x_c_t = self.de_emb_layer_c_t(x_c_t).permute(0, 2, 1)
            # f = torch.cat([x_c_t, x_c_v], dim=-1)
            # f_att = torch.sigmoid(self.w_att(f))
            # x = f_att * x_c_v + (1 - f_att) * x_c_t
            # attns.append(attn)
        for attn_layer in self.attn_layers2:
            enc_out_c_time, attn_c_time = attn_layer(enc_out_c_time,
                                                     attn_mask=attn_mask,
                                                     tau=tau, delta=delta,
                                                     use_DPE=True, use_rot=False,
                                                     init_score=init_score_c_t)
        # x = self.emb_layer(x, x_mark_enc, flag="cross_var")
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        # x_c_v = self.de_emb_layer_c_v1(enc_out_c_var.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = self.de_emb_layer_c_v(enc_out_c_var)
        x_c_v = self.de_emb_layer_c_v1(x_c_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_t = self.de_emb_layer_c_t(enc_out_c_time).permute(0, 2, 1)
        # # 使用TV维度进行融合
        # x_c_t = x_c_t.unsqueeze(-1)
        # x_c_v = x_c_v.unsqueeze(-1)
        # x_c_t = self.st_emb_layer(x_c_t)
        # x_c_v = self.st_emb_layer(x_c_v)
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = self.st_de_emb_layer(x).squeeze(-1)
        # 使用T维度进行融合
        f = torch.cat([x_c_t.permute(0, 2, 1), x_c_v.permute(0, 2, 1)], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        f_att = f_att.permute(0, 2, 1)
        x = f_att * x_c_v + (1 - f_att) * x_c_t

        # # 使用V维度进行融合
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # f_att = f_att
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # # x = x.permute(0, 2, 1)
        # # if self.norm is not None:
        # #     x = self.norm(x)
        x = x[:, :, :self.configs.enc_in]
        x_c_v = self.media_decode_c_v(x_c_v)
        x_c_t = self.media_decode_c_t(x_c_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = x_c_v[:, :, :self.configs.enc_in]
        x_c_t = x_c_t[:, :, :self.configs.enc_in]
        init_score = torch.softmax(torch.matmul(x1, x1.permute(0, 2, 1)) /
                                   (x.shape[-1] ** 0.5), dim=-1)
        attns_diff = 0
        # 计算 L2 范数的平方
        if attn_c_var is not None:
            l2_norm_squared = torch.norm(init_score.unsqueeze(1) - attn_c_var, p=2) ** 2
        else:
            l2_norm_squared = None
        return x, x_c_v, x_c_t, l2_norm_squared


class Encoder_cross_t_v_duli_try(nn.Module):
    def __init__(self, attn_layers1, attn_layers2, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v_duli_try, self).__init__()
        self.configs = configs
        self.attn_layers1 = nn.ModuleList(attn_layers1)
        self.attn_layers2 = nn.ModuleList(attn_layers2)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer

        self.fc_emb1 = nn.Linear(configs.seq_len, configs.d_model)
        if configs.data in {"PEMS", "Solar"}:
            self.fc_emb2 = nn.Linear(configs.enc_in, configs.d_model)
        else:
            self.fc_emb2 = nn.Linear(configs.enc_in + 4, configs.d_model)

        self.fc_dec_1_1 = nn.Linear(configs.d_model, configs.pred_len)
        self.fc_dec_2_2 = nn.Linear(configs.d_model, configs.pred_len)
        if configs.data in {"PEMS", "Solar"}:
            self.fc_dec_1_2 = nn.Linear(configs.d_model, configs.enc_in)
            self.fc_dec_2_1 = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.fc_dec_1_2 = nn.Linear(configs.d_model, configs.enc_in + 4)
            self.fc_dec_2_1 = nn.Linear(configs.d_model, configs.enc_in + 4)

        self.w_att = nn.Linear(configs.d_model * 2, configs.d_model)
        if configs.data in {"PEMS", "Solar"}:
            self.fc_dec_fuse1 = nn.Linear(configs.d_model, configs.enc_in)
        else:
            self.fc_dec_fuse1 = nn.Linear(configs.d_model, configs.enc_in + 4)
        self.fc_dec_fuse2 = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if x_mark_enc is not None:
            x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        for attn_layer in self.attn_layers1:
            enc_out_c_var, attn = attn_layer(enc_out_c_var,
                                             attn_mask=attn_mask,
                                             tau=tau, delta=delta,
                                             use_DPE=False, use_rot=True)
        for attn_layer in self.attn_layers2:
            enc_out_c_time, attn = attn_layer(enc_out_c_time,
                                              attn_mask=attn_mask,
                                              tau=tau, delta=delta,
                                              use_DPE=True, use_rot=False)
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        x_decode1 = self.fc_emb1(enc_out_c_var.permute(0, 2, 1))
        x_decode1_1 = self.fc_dec_1_1(x_decode1)
        x_decode1_2 = self.fc_dec_1_2(x_decode1_1.permute(0, 2, 1))
        x_decode2 = self.fc_emb2(enc_out_c_time.permute(0, 2, 1))
        x_decode2_1 = self.fc_dec_2_1(x_decode2)
        x_decode2_2 = self.fc_dec_2_2(x_decode2_1.permute(0, 2, 1)).permute(0, 2, 1)
        x_decode2 = x_decode2.permute(0, 2, 1)
        # 使用T维度进行融合
        f = torch.cat([x_decode1, x_decode2], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        x = f_att * x_decode1 + (1 - f_att) * x_decode2
        x = self.fc_dec_fuse2(x)
        x = self.fc_dec_fuse1(x.permute(0, 2, 1))
        return x, x_decode1_2, x_decode2_2, attns


class Encoder_cross_t_v_Patch(nn.Module):
    def __init__(self, attn_layers, configs, conv_layers=None, norm_layer=None, emb_layer=None):
        super(Encoder_cross_t_v_Patch, self).__init__()
        self.configs = configs
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.emb_layer = emb_layer
        self.de_emb_layer_c_v = nn.Linear(configs.d_model, (configs.enc_in + 4) * 6)
        self.de_emb_layer_c_v1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.de_emb_layer_c_t = nn.Linear(configs.d_model, configs.pred_len)
        # self.w_att = nn.Linear(512*2, 512)
        self.w_att = nn.Linear(configs.pred_len * 2, configs.pred_len)
        self.st_emb_layer = nn.Linear(1, configs.d_model)
        self.st_de_emb_layer = nn.Linear(configs.d_model, 1)

        self.media_decode_c_v = nn.Linear(configs.enc_in + 4, configs.enc_in + 4)
        self.media_decode_c_t = nn.Linear(configs.pred_len, configs.pred_len)

    def forward(self, x, x_mark_enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        x = torch.cat([x, x_mark_enc], -1)
        x1 = x.clone()
        # enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_var_Patch",
        #                                num=0)  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_c_var = self.emb_layer(x, x_mark_enc, flag="cross_time_part",
                                       num=0)  # covariates (e.g timestamp) can be also embedded as
        batch, var_num = enc_out_c_var.shape[:2]
        # enc_out_c_var = enc_out_c_var.reshape([batch*var_num, enc_out_c_var.shape[2], enc_out_c_var.shape[3]])
        enc_out_c_time = self.emb_layer(x1, x_mark_enc, flag="cross_time")
        for attn_layer in self.attn_layers:
            enc_out_c_var, attn = attn_layer(enc_out_c_var,
                                             attn_mask=attn_mask,
                                             tau=tau, delta=delta,
                                             use_DPE=True, use_rot=True)
            # x_c_t, attn = attn_layer(enc_out_c_time, attn_mask=attn_mask, tau=tau, delta=delta)
            # x_c_v = self.de_emb_layer_c_v(x_c_v)
            # x_c_t = self.de_emb_layer_c_t(x_c_t).permute(0, 2, 1)
            # f = torch.cat([x_c_t, x_c_v], dim=-1)
            # f_att = torch.sigmoid(self.w_att(f))
            # x = f_att * x_c_v + (1 - f_att) * x_c_t
            # attns.append(attn)
        for attn_layer in self.attn_layers:
            enc_out_c_time, attn = attn_layer(enc_out_c_time,
                                              attn_mask=attn_mask,
                                              tau=tau, delta=delta,
                                              use_DPE=True, use_rot=True)
        # x = self.emb_layer(x, x_mark_enc, flag="cross_var")
        if self.norm is not None:
            enc_out_c_var = self.norm(enc_out_c_var)
            enc_out_c_time = self.norm(enc_out_c_time)
        # x_c_v = self.de_emb_layer_c_v1(enc_out_c_var.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = self.de_emb_layer_c_v(enc_out_c_var)
        x_c_v = x_c_v.reshape(x_c_v.shape[0], x_c_v.shape[1] * 6, x_c_v.shape[2] // 6)
        x_c_v = self.de_emb_layer_c_v1(x_c_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_t = self.de_emb_layer_c_t(enc_out_c_time).permute(0, 2, 1)
        # # 使用TV维度进行融合
        # x_c_t = x_c_t.unsqueeze(-1)
        # x_c_v = x_c_v.unsqueeze(-1)
        # x_c_t = self.st_emb_layer(x_c_t)
        # x_c_v = self.st_emb_layer(x_c_v)
        # f = torch.cat([x_c_t, x_c_v], dim=-1)
        # f_att = torch.sigmoid(self.w_att(f))
        # x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = self.st_de_emb_layer(x).squeeze(-1)
        # 使用T维度进行融合
        f = torch.cat([x_c_t.permute(0, 2, 1), x_c_v.permute(0, 2, 1)], dim=-1)
        f_att = torch.sigmoid(self.w_att(f))
        f_att = f_att.permute(0, 2, 1)
        x = f_att * x_c_v + (1 - f_att) * x_c_t
        # x = x.permute(0, 2, 1)
        # if self.norm is not None:
        #     x = self.norm(x)
        x = x[:, :, :self.configs.enc_in]
        x_c_v = self.media_decode_c_v(x_c_v)
        x_c_t = self.media_decode_c_t(x_c_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_c_v = x_c_v[:, :, :self.configs.enc_in]
        x_c_t = x_c_t[:, :, :self.configs.enc_in]
        return x, x_c_v, x_c_t, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
