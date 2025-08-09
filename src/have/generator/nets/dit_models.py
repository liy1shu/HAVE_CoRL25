# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import numpy as np
import rpad.pyg.nets.pointnet2 as pnp_original
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

import failure_recovery.nets.pn2 as pnp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type="Rotary1D"):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = (
            torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        )
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(
                0, self.feature_dim, 2, dtype=torch.float, device=x_position.device
            )
            * (-math.log(10000.0) / (self.feature_dim))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx],
        )
        position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    def __init__(self, feature_dim, pe_type="Rotary3D"):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        """
        @param XYZ: [B,N,3]
        @return:
        """
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(
                0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device
            )
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz],
        )

        position_code = torch.stack(
            [
                torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
                torch.cat([sinx, siny, sinz], dim=-1),  # sin_pos
            ],
            dim=-1,
        )

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # print("shift: ", shift)
        # print("scale: ", scale)
        # print("After norm:", self.norm_final(x)[0, 0, :])
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PN2DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=[30, 40],
        patch_size=1,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        pos_embed_freq_L=10,
        n_points=1200,
        time_embed_dim=64,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.n_points = n_points

        self.h, self.w = input_size[0], input_size[1]

        # # 0) Pure input
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels + pos_embed_freq_L * 6, hidden_size, bias=True)
        # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 1) Point Cloud features
        self.x_embedder = pnp_original.PN2Dense(
            in_channels=3,
            out_channels=hidden_size,
            p=pnp_original.PN2DenseParams(),
        )

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels + 3, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        num_patches = self.n_points
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.pos_embed_freq_L = pos_embed_freq_L

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def pcd_positional_encoding(self, xyz):
        """
        Apply sinusoidal positional encoding to the given 3D coordinates.

        :param xyz: A numpy array of shape (N, 3), where N is the number of points, and each point has x, y, z coordinates.
        :return: A numpy array of shape (N, 6*L) containing the positional embeddings.
        """

        L = self.pos_embed_freq_L

        # Initialize an array to hold the positional encodings
        embeddings = np.zeros((xyz.shape[0], 6 * L))

        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        frequencies = 2 ** np.arange(L)

        # Apply sinusoidal encoding
        for i, freq in enumerate(frequencies):
            for j, func in enumerate([np.sin, np.cos]):
                embeddings[:, (6 * i + 2 * j) : (6 * i + 2 * j + 3)] = func(
                    2 * np.pi * xyz.detach().cpu().numpy() * freq
                )

        return torch.from_numpy(embeddings).float().cuda()

    def forward(self, x, t, pos, context, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        pos: (N, H*W, C)
        """
        # # 0) Takes original point cloud
        # pos_embed = self.pcd_positional_encoding(torch.flatten(pos, start_dim=0, end_dim=1))  # N*T * D
        # x = torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1)  # (N, H*W, C)
        # x = torch.concat((x, pos_embed.reshape(x.shape[0], x.shape[1], -1)), dim=-1)
        # x = self.x_embedder(x.reshape(x.shape[0], -1, 30, 40))
        # 1) Take pointnet++ encoded point cloud
        context.x = (
            torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1).reshape(-1, 3)
        )
        encoded_pcd = self.x_embedder(context.cuda())
        x = encoded_pcd.reshape(x.shape[0], 1200, -1)

        # # 2) Take DGCNN encoded point cloud
        # # print(torch.flatten(x, start_dim=2, end_dim=3).shape, pos.permute(0, 2, 1).shape)
        # x = torch.cat(
        #     (torch.flatten(x, start_dim=2, end_dim=3), pos.permute(0, 2, 1)), dim=1
        # )
        # encoded_pcd = self.x_embedder(x, t, pos.permute(0, 2, 1)).sample
        # x = encoded_pcd.permute(0, 2, 1)

        t = self.t_embedder(t)  # (N, D)

        # print("x", x.shape, "t", t.shape)

        # y = self.y_embedder(y, self.training)    # (N, D)
        c = t
        # print("c", c.shape)                              # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        # print("after final layer:", x.shape)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, cfg_scale, pos, context):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        model_out = self.forward(x, t, pos, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        return torch.cat([eps, rest], dim=1)


class PN2HisDiT(nn.Module):  # With history latent everywhere version
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        history_embed_dim=128,
        input_size=[30, 40],
        patch_size=1,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        pos_embed_freq_L=10,
        n_points=1200,
        time_embed_dim=64,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.n_points = n_points

        self.h, self.w = input_size[0], input_size[1]

        # # 0) Pure input
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels + pos_embed_freq_L * 6, hidden_size, bias=True)
        # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 1) Point Cloud features
        self.x_embedder = pnp.PN2DenseLatentEncodingEverywhere(
            history_embed_dim=history_embed_dim,
            in_channels=3,
            out_channels=hidden_size,
        )

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels + 3, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        num_patches = self.n_points
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.pos_embed_freq_L = pos_embed_freq_L

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def pcd_positional_encoding(self, xyz):
        """
        Apply sinusoidal positional encoding to the given 3D coordinates.

        :param xyz: A numpy array of shape (N, 3), where N is the number of points, and each point has x, y, z coordinates.
        :return: A numpy array of shape (N, 6*L) containing the positional embeddings.
        """

        L = self.pos_embed_freq_L

        # Initialize an array to hold the positional encodings
        embeddings = np.zeros((xyz.shape[0], 6 * L))

        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        frequencies = 2 ** np.arange(L)

        # Apply sinusoidal encoding
        for i, freq in enumerate(frequencies):
            for j, func in enumerate([np.sin, np.cos]):
                embeddings[:, (6 * i + 2 * j) : (6 * i + 2 * j + 3)] = func(
                    2 * np.pi * xyz.detach().cpu().numpy() * freq
                )

        return torch.from_numpy(embeddings).float().cuda()

    def forward(self, x, t, pos, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        pos: (N, H*W, C)
        """
        # # 0) Takes original point cloud
        # pos_embed = self.pcd_positional_encoding(torch.flatten(pos, start_dim=0, end_dim=1))  # N*T * D
        # x = torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1)  # (N, H*W, C)
        # x = torch.concat((x, pos_embed.reshape(x.shape[0], x.shape[1], -1)), dim=-1)
        # x = self.x_embedder(x.reshape(x.shape[0], -1, 30, 40))
        # 1) Take pointnet++ encoded point cloud
        context.x = (
            torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1).reshape(-1, 3)
        )
        encoded_pcd = self.x_embedder(context.cuda(), latents=context.history_embed)
        x = encoded_pcd.reshape(x.shape[0], 1200, -1)

        # # 2) Take DGCNN encoded point cloud
        # # print(torch.flatten(x, start_dim=2, end_dim=3).shape, pos.permute(0, 2, 1).shape)
        # x = torch.cat(
        #     (torch.flatten(x, start_dim=2, end_dim=3), pos.permute(0, 2, 1)), dim=1
        # )
        # encoded_pcd = self.x_embedder(x, t, pos.permute(0, 2, 1)).sample
        # x = encoded_pcd.permute(0, 2, 1)

        t = self.t_embedder(t)  # (N, D)

        # print("x", x.shape, "t", t.shape)

        # y = self.y_embedder(y, self.training)    # (N, D)
        c = t
        # print("c", c.shape)                              # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        # print("after final layer:", x.shape)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


    def forward_with_particle(self, x, t, pos, context):  # With particle guidance https://arxiv.org/abs/2310.13102
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        pos: (N, H*W, C)
        """
        assert x.shape[0] > 1, "Needs multiple samples for particle guidance sampling"
        context.x = (
            torch.flatten(x, start_dim=2, end_dim=3).permute(0, 2, 1).reshape(-1, 3)
        )
        encoded_pcd = self.x_embedder(context.cuda(), latents=context.history_embed)
        x = encoded_pcd.reshape(x.shape[0], 1200, -1)
        t = self.t_embedder(t)  # (N, D)
        c = t
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        power = 2
        coeff = 10
        noise, sigma = torch.split(x, x.shape[1] // 2, dim=1)
        x_vec = noise.reshape(len(noise), -1)
        diff = x_vec.unsqueeze(1) - x_vec.unsqueeze(0)
        diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])
        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
        N = x_vec.shape[0]
        h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(N - 1)
        weights = torch.exp(- (distance ** power / h_t))

        grad_phi = 2 * weights * diff / h_t * sigma.reshape(len(noise), 1, -1) * coeff
        grad_phi = grad_phi.sum(dim=1)
        grad_phi = grad_phi.reshape_as(noise)
        return torch.concatenate([noise + grad_phi, sigma], dim=1)


    def forward_with_cfg(self, x, t, pos, context, cfg_scale=0.8):    # Unused
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # print("Cfg scale: ", cfg_scale)
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        half_t = t[: len(t) // 2]
        combined_t = torch.cat([half_t, half_t], dim=0)
        model_out = self.forward(combined, combined_t, pos, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = cond_eps + cfg_scale * (cond_eps - uncond_eps)  # Original cfg paper
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)  # Original DiT
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



    # def forward_with_cfg(self, x, t, cfg_scale, pos, context):    # Unused
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     # half = x[: len(x) // 2]
    #     model_out = self.forward(x, t, pos, context)

    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
    #     return torch.cat([eps, rest], dim=1)




class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        # x_embedder is conv1d layer instead of 2d patch embedder
        self.x_embedder = nn.Conv1d(
            in_channels, hidden_size, kernel_size=1, stride=1, padding=0, bias=True
        )
        # no pos_embed, or y_embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, pos):
        """
        Forward pass of DiT.
        x: (N, 3, L) tensor of noisy flows
        t: (N,) tensor of diffusion timesteps
        pos: (N, 3, L) tensor of 3D coordinates
        """
        # NOTE: the patchify/unpatchify layers handle the dimension swapping, so we need to manually do that here

        # concat x and pos
        # print(x.shape, pos.shape)
        x = torch.cat((x, pos), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)
        t = self.t_embedder(t)  # (N, D)
        c = t  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, L, D)
        x = self.final_layer(x, c)  # (N, L, patch_size ** 2 * out_channels)
        # transpose back to (N, out_channels, L)
        x = torch.transpose(x, -1, -2)
        return x


class RoPEDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.rope_embedder = RotaryPositionEncoding3D(hidden_size)
        # x_embedder is conv1d layer instead of 2d patch embedder
        self.x_embedder = nn.Conv1d(
            in_channels, hidden_size, kernel_size=1, stride=1, padding=0, bias=True
        )
        # no pos_embed, or y_embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, pos):
        """
        Forward pass of DiT.
        x: (N, 3, L) tensor of noisy flows
        t: (N,) tensor of diffusion timesteps
        pos: (N, 3, L) tensor of 3D coordinates
        """
        # NOTE: the patchify/unpatchify layers handle the dimension swapping, so we need to manually do that here

        # concat x and pos
        # print(x.shape, pos.shape)
        x = self.x_embedder(x)
        x += self.rope_embedder(pos)
        x = torch.transpose(x, -1, -2)
        t = self.t_embedder(t)  # (N, D)
        c = t  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, L, D)
        x = self.final_layer(x, c)  # (N, L, patch_size ** 2 * out_channels)
        # transpose back to (N, out_channels, L)
        x = torch.transpose(x, -1, -2)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}


if __name__ == "__main__":
    model = DiT(
        input_size=[30, 40], depth=28, hidden_size=1152, patch_size=1, num_heads=16
    )
    input = torch.randn(1, 3, 30, 40)
    pos = torch.randn(1, 1200, 3)
    t = torch.randint(100, size=(1,))
    print(model(input, t, pos).shape)
