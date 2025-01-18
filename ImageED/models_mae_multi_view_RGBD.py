# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from functools import partial

import torch
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from timm.models.vision_transformer import Block
from torch import nn as nn


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


class ViewAgnosticPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, V, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = torch.stack([self.proj(x[:, i]) for i in range(V)], dim=2)  # single project for all views
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def generate_2d_sincos_pos_embeddings(
    embedding_dim: int,
    length: int,
    scale: float = 10000.0,
    use_class_token: bool = True,
    num_views: int = 1,
) -> torch.nn.Parameter:
    """Reference: https://github.com/recursionpharma/maes_microscopy/blob/b3e68bb14cccbb05d478e9dd39bcbe4a113cd14d/vit.py#L6
    Generate 2Dimensional sin/cosine positional embeddings

    Parameters
    ----------
    embedding_dim : int
        embedding dimension used in vit
    length : int
        number of tokens along height or width of image after patching (assuming square)
    scale : float
        scale for sin/cos functions
    use_class_token : bool
        True - add zero vector to be added to class_token, False - no vector added
    num_views: number of views. If 0, a single modality is assumed.
        Otherwise one-hot modality encoding is added and sincos encoding size is appropriately reduced.

    Returns
    -------
    positional_encoding : torch.Tensor
        positional encoding to add to vit patch encodings
        [num_modality*length*length, embedding_dim] or [1+num_modality*length*length, embedding_dim]
        (w/ or w/o cls_token)
    """

    linear_positions = torch.arange(length, dtype=torch.float32)
    height_mesh, width_mesh = torch.meshgrid(
        linear_positions, linear_positions, indexing="ij"
    )
    positional_dim = embedding_dim // 4  # accomodate h and w x cos and sin embeddings
    positional_weights = (
        torch.arange(positional_dim, dtype=torch.float32) / positional_dim
    )
    positional_weights = 1.0 / (scale**positional_weights)

    height_weights = torch.outer(height_mesh.flatten(), positional_weights)
    width_weights = torch.outer(width_mesh.flatten(), positional_weights)

    positional_encoding = torch.cat(
        [
            torch.sin(height_weights),
            torch.cos(height_weights),
            torch.sin(width_weights),
            torch.cos(width_weights),
        ],
        dim=1,
    )[None, :, :]

    # repeat positional encoding for multiple channel modalities
    positional_encoding = positional_encoding.repeat(1, num_views, 1)

    if use_class_token:
        class_token = torch.zeros([1, 1, embedding_dim], dtype=torch.float32)
        positional_encoding = torch.cat([class_token, positional_encoding], dim=1)

    positional_encoding = torch.nn.Parameter(positional_encoding, requires_grad=False)

    return positional_encoding


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 n_views=6, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_loss_weight=1, recovery_loss_weight=1):
        super().__init__()

        self.in_chans = in_chans
        self.n_views = n_views
        self.mask_loss_weight = mask_loss_weight
        self.recovery_loss_weight = recovery_loss_weight
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ViewAgnosticPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = generate_2d_sincos_pos_embeddings(
            embedding_dim=embed_dim,
            length=self.patch_embed.grid_size[0],
            use_class_token=True,
            num_views=self.n_views,
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = generate_2d_sincos_pos_embeddings(
            embedding_dim=decoder_embed_dim,
            length=self.patch_embed.grid_size[0],
            use_class_token=True,
            num_views=self.n_views,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, n_view, in_chans, H, W)
        x: (N, L, patch_size**2 * in_chans)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

        h = w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.n_views, self.in_chans, h, p, w, p))
        x = torch.einsum('nvchpwq->nvhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1] * h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * in_chans)
        imgs: (N, views, in_chans, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int((x.shape[1] // self.n_views)**.5)
        assert h * w * self.n_views == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], self.n_views, h, w, p, p, self.in_chans))
        x = torch.einsum('nvhwpqc->nvchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.n_views, self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        mask_loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        unmask_loss = (loss * (1-mask)).sum() / (1-mask).sum()  # recovery loss

        loss = self.mask_loss_weight * mask_loss + self.recovery_loss_weight * unmask_loss

        return loss, mask_loss, unmask_loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

