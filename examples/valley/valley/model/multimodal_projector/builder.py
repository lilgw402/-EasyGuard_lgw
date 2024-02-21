import torch
import torch.nn as nn
import re
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class PoolAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size=4):
        super().__init__()
        self.pool_h, self.pool_w = pool_out_size, pool_out_size

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )
                
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        # print(x.shape)  # torch.Size([image_num, vit_token_num, dim_in])  [8, 257, 1024]
        f, v, d = x.shape
        s = int(math.sqrt(v-1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d)

        if s % self.pool_h == 0 and s % self.pool_w == 0:
            x = x.reshape(f, self.pool_h, s//self.pool_h, self.pool_w, s//self.pool_w, d)
            x = x.permute([0, 1, 3, 5, 2, 4]).reshape(f, self.pool_h * self.pool_w, d, -1).mean(-1)
            x = self.mlp(x)                 # [f, h*w, d]
        #else:
        #    x = x.flatten(0, 2).permute(0, 3, 1, 2)
        #    x = torch.nn.functional.adaptive_avg_pool2d(x, (self.pool_h, self.pool_w))
        #    x = x.permute(0, 2, 3, 1).flatten(1, 2)
        #    x = self.mlp(x)                 # [b, t, f, h*w, d]
        else:
            raise ValueError()

        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
        
    elif projector_type == 'pool_adapter':
        return PoolAdapter(config.mm_hidden_size, config.hidden_size, config.pool_out_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
