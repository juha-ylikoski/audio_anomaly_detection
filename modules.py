from typing import List, Tuple
import torch.nn as nn
import torch

# missing from __all__ in compressai/layers/layers.py
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class SCCTXModel(nn.Module):
    """
    Spatial-channel context model according to ELIC paper.
    """
    def __init__(self, M: int, block_sizes: List[int]) -> None:
        super().__init__()
        self.block_sizes = block_sizes
        self.g_sps = nn.ModuleList([
            CheckerboardContext(
                in_channels=self.block_sizes[i], 
                out_channels=self.block_sizes[i]*2,
                kernel_size=5, 
                stride=1, 
                padding=2) 
          for i in range(len(self.block_sizes))])

        self.g_chs = nn.ModuleList([
            ChannelContext(
                Mk=self.block_sizes[i], 
                y_hat_channels=sum(self.block_sizes[:i])) 
            for i in range(1,len(self.block_sizes))])
            
        # not sure if param agg should be different for blocks
        self.param_aggs = nn.ModuleList([
                ParamAgg(self.block_sizes[i], M) 
            for i in range(len(self.block_sizes))])

    def predict_non_anchor(self, 
                            y_hat: List[torch.Tensor], 
                            block_i: int, 
                            psi: torch.Tensor, 
                            y_anchor: torch.Tensor,
                            mask: bool = False) -> Tuple[torch.Tensor]:
        """
        @params:
        y_hat: List of tensors shape (bs,ch,h,w)
        block_i: index of current block
        psi: Tensor shape (bs,2*M,h,w)
        y_anchor: Tensor of shape (bs,ch,h,w)
        
        returns: tuple of tensors mu, sigma paramateres to de/encode the 
        non_anchor parts of block_i
        """
        sp_cx = self.g_sps[block_i](y_anchor)
        if block_i == 0:
            ch_cx = torch.zeros_like(sp_cx)
        else:
            ch_cx = self.g_chs[block_i-1](torch.cat(y_hat, dim=1))
        if mask:
            # masks the anchor. Used during training to get one pass encoding
            sp_cx[:, :, 0::2, 1::2] = 0
            sp_cx[:, :, 1::2, 0::2] = 0
        cat = torch.cat((sp_cx, ch_cx, psi), dim=1)
        theta = self.param_aggs[block_i](cat)
        scales, means = theta.chunk(2,1)
        return means, scales

    def predict_anchor(self, 
                       y_hat: List[torch.Tensor], 
                       block_i: int, 
                       psi: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        returns tuple of tensors mu, sigma parametres to de/encode the anchors of 
        block_i+1

        get current encoded block of y and calculate theta(chX, sp1) using psi 
        and channel context from g_ch[<block_i](y_hat).
        """
        if block_i == 0:
            bs,c,h,w = psi.shape
            ch_cx = torch.zeros(bs, 2*self.block_sizes[block_i], h, w)
        else:
            ch_cx = self.g_chs[block_i-1](torch.cat(y_hat, dim=1))

        sp_cx = torch.zeros_like(ch_cx) # no spatial context
        cat = torch.cat((sp_cx, ch_cx, psi), dim=1)
        theta = self.param_aggs[block_i](cat)
        scales, means = theta.chunk(2,1)
        return means, scales


class ParamAgg(nn.Module):
    def __init__(self, Mk: int, M: int) -> None:
        super().__init__()
        # LINEARY reduces the number of channels
        # should this be the same for all blocks? 
        # input and output size depends on block.
        i = 4*Mk+2*M
        o = 2*Mk
        s = (i-o)//3
        self.block = nn.Sequential(
              conv1x1(i, i-s),
              nn.ReLU(inplace=True),
              conv1x1(i-s, o+s),
              nn.ReLU(inplace=True),
              conv1x1(o+s, o)
          )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CheckerboardContext(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

class ChannelContext(nn.Module):
    def __init__(self, Mk: int, y_hat_channels: int) -> None:
        super().__init__()
        # only last layer doubles the channels? is this feasible?  
        self.block = nn.Sequential(
            nn.Conv2d(y_hat_channels, Mk, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(Mk, Mk, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(Mk, 2*Mk, kernel_size=5, stride=1, padding=2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)