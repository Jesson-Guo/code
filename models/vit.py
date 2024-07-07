import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .layers import Block, OverlapPatchEmbed, Head

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Block as NaiveBlock
import math


class ViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[768, 1536],
                 patch_sizes=[2, 3, 3, 3], paddings=[0, 1, 1, 1], num_heads=[12, 16], mlp_ratios=[4, 2], 
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[12, 2], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_stages = num_stages
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        a = dpr[:depths[0]]
        b = dpr[depths[0]:]
        self.visual = VisionTransformer(
            img_size=img_size,
            num_classes=num_classes,
            embed_dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[:depths[0]],
            norm_layer=norm_layer，
        )

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size // (2**num_stages),
                patch_size=patch_sizes[i],
                stride=2,
                padding=paddings[i],
                in_chans=embed_dims[-2],
                embed_dim=embed_dims[-1])

            cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
            pos_embed = nn.Parameter(torch.randn(1, patch_embed.num_patches+1, embed_dims[1]) * .02)

            block = nn.Sequential(*[
                NaiveBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depths[0]:],
                    norm_layer=norm_layer,
                )
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])

            # classification head
            head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

            setattr(self, f"cls_token{i+1}", cls_token)
            setattr(self, f"pos_embed{i+1}", pos_embed)
            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}", block)
            setattr(self, f"norm{i+1}", norm)
            setattr(self, f"head{i+1}", head)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_list = []
        for k, _ in self.named_parameters():
            k_split = k.split('.')
            if 'pos_embed' in k_split or 'cls_token' in k_split:
                no_weight_list.append(k)
        return no_weight_list

    def forward_head(self, i):
        cls_token = getattr(self, f"cls_token{i+1}")
        pos_embed = getattr(self, f"pos_embed{i+1}")
        patch_embed = getattr(self, f"patch_embed{i+1}")
        block = getattr(self, f"block{i+1}")
        norm = getattr(self, f"norm{i+1}")
        head = getattr(self, f"head{i+1}")

    def forward(self, x):
        x = self.visual(x)

        prompt = None
        for i in range(self.num_stages):

            out, prompt = classifier(x, prompt)

            if i == self.num_stages-1:
                outputs.append(out.squeeze())
            else:
                outputs[0].append(out)

        return x
