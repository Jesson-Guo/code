import numpy as np
import torch
import torch.nn as nn
from functools import partial

from .layers import Block, OverlapPatchEmbed, Head

from timm.models.layers import trunc_normal_
import math


class Encoder(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        num_branches=None,
        embed_dim=768,
        patch_embed=None,
        depth=12,
        ca_num_heads=4,
        sa_num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        use_layerscale=False,
        layerscale_value=1e-4,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        ca_attentions=[1],
        expand_ratio=2,
        norm_layer=nn.LayerNorm,
        has_cls_token=False,
        use_local_conv=True,
        **kwargs
    ):
        super().__init__()
        self.has_cls_token = has_cls_token
        self.num_branches = num_branches
        self.patch_embed = patch_embed

        self.block = nn.ModuleList([
            Block(
                dim=embed_dim,
                ca_num_heads=ca_num_heads,
                sa_num_heads=sa_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[j],
                norm_layer=norm_layer,
                ca_attention=ca_attentions[j],
                use_local_conv=use_local_conv,
                expand_ratio=expand_ratio
            ) for j in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if has_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, num_branches, embed_dim))
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward(self, x, pre_embed=None):
        x, H, W = self.patch_embed(x)
        info = {'H': H, 'W': W}

        if self.has_cls_token:
            if pre_embed == None:
                pre_embed = torch.zeros(x.shape[0], 0, x.shape[2]).cuda()
            info['prompt_size'] = pre_embed.shape[1]
            info['n_cls_token'] = self.num_branches
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = torch.cat([x, pre_embed], dim=1)

        for blk in self.block:
            x = blk(x, info)
        x = self.norm(x)

        if self.has_cls_token:
            split_size = [self.num_branches, x.shape[1]-self.num_branches]
            x = x.split(split_size, dim=1)
            out = self.head(x[0])
            # out = torch.nn.functional.softplus(out)
            return out, x[1]
        else:
            return x, H, W


class MyModel(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_coarses=[1, 4], num_vocab=1, embed_dims=[64, 128, 256, 512],
                 patch_sizes=[3, 3, 3, 3], paddings=[1, 1, 1, 1], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], 
                 mlp_ratios=[8, 6, 4, 2], qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, head_conv=3, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[4, 6, 28, 4], ca_attentions=[1, 1, 1, 0], num_stages=4, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_coarses = num_coarses
        self.num_stages = num_stages
        self.num_plans = 2
        self.embed_dims = embed_dims

        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages-1):
            if i == 0:
                patch_embed = Head(head_conv, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(
                    img_size=img_size if i == 0 else img_size // (2**(i+1)),
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i-1],
                    embed_dim=embed_dims[i]
                )

            ca_attentions = [0 if i==2 and j%2!=0 else ca_attentions[i] for j in range(depths[i])]

            enc = Encoder(
                embed_dim=embed_dims[i],
                patch_embed=patch_embed,
                depth=depths[i],
                ca_num_heads=ca_num_heads[i],
                sa_num_heads=sa_num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur: cur+depths[i]],
                ca_attentions=ca_attentions,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
            )
            cur += depths[i]
            setattr(self, f"enc{i+1}", enc)

        dpr = dpr[cur:]
        ca_attentions = [0 for _ in range(depths[-1])]
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size // (2**num_stages),
                patch_size=patch_sizes[i],
                stride=2,
                padding=paddings[i],
                in_chans=embed_dims[-2],
                embed_dim=embed_dims[-1]
            )
            classifier = Encoder(
                num_classes=num_classes if i == num_stages-1 else num_coarses[i+1],
                num_branches=1 if i == num_stages-1 else 2**(i+1),
                embed_dim=embed_dims[-1],
                patch_embed=patch_embed,
                depth=depths[-1],
                ca_num_heads=ca_num_heads[-1],
                sa_num_heads=sa_num_heads[-1],
                mlp_ratio=mlp_ratios[-1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr,
                ca_attentions=ca_attentions,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                has_cls_token=True,
                use_local_conv=False,
            )

            setattr(self, f"classifier{i+1}", classifier)

        modules = []
        for i in range(num_stages-1):
            modules.append(getattr(self, f"enc{i+1}"))
        for module in modules:
            for k, p in module.named_parameters():
                p.requires_grad = False

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

    def forward(self, x):
        outputs = [[]]

        B = x.shape[0]

        for i in range(self.num_stages-1):
            enc = getattr(self, f"enc{i+1}")
            x, H, W = enc(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        prompt = None
        for i in range(self.num_stages):
            classifier = getattr(self, f"classifier{i+1}")
            # out = classifier(x, prompt)
            # prompt = out[1].detach()
            out, prompt = classifier(x, prompt)

            if i == self.num_stages-1:
                outputs.append(out.squeeze())
            else:
                outputs[0].append(out)

        return outputs

    def infer(self, x):
        all_probs = {"ori": 0, "coarses": [], "leaves": []}
        probs = {"ori": 0, "coarses": [], "leaves": []}
        preds = {"ori": 0, "coarses": [], "leaves": []}
        alpha = []

        B = x.shape[0]
        for i in range(self.num_stages-1):
            enc = getattr(self, f"enc{i+1}")
            x, H, W = enc(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        enc = getattr(self, f"enc{self.num_stages}")
        x_enc = enc(x)[0]

        # infer coarse
        y_in = []
        y_next = x.new_zeros(B, 1).to(torch.long)
        for i in range(self.num_stages):
            y_in.append(y_next)
            embed, split_size = self.construct_embed(y_in)
            out = self.text_dec.infer(embed, x_enc, i+1, split_size)

            all_prob = out.softmax(dim=-1)
            prob = all_prob.max(dim=-1)[0]
            pred = all_prob.max(dim=-1)[1]

            all_probs["coarses"].append(all_prob)
            probs["coarses"].append(prob)
            preds["coarses"].append(pred)

            y_next = pred + sum(self.num_coarses[:i+1])

        # infer leaves
        y_in.append(y_next)
        for i in range(self.num_stages):
            classifier = getattr(self, f"classifier{i+1}")
            embed = self.construct_embed(y_in[:i+2])[0]
            out = classifier(x, embed)[0]

            all_prob = out.softmax(dim=-1)
            prob = all_prob.max(dim=-1)[0]
            pred = all_prob.max(dim=-1)[1]

            all_probs["leaves"].append(all_prob)
            probs["leaves"].append(prob)
            preds["leaves"].append(pred)
            alpha.append(out+1)

        return all_probs, probs, preds, alpha
