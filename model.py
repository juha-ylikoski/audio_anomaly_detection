from typing import Dict
import torch.nn as nn
import torch

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    #conv1x1
)
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import (
    GaussianConditional,
    EntropyBottleneck,
)
from compressai.models import JointAutoregressiveHierarchicalPriors

from modules import SCCTXModel, conv1x1

class ELICModel(JointAutoregressiveHierarchicalPriors):
    """
    Args:
        N (int): Number of channels
      """
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.block_sizes = [16,16,32,64,M-128]
        self.g_a = nn.Sequential(
            # something like this
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.h_a = nn.Sequential(
            
        )

        self.h_s = nn.Sequential(
            
        )

        self.g_a = nn.Sequential(
            
        )

        self.scctx = SCCTXModel(self.M, self.block_sizes)     

        # inherited
        # self.gaussian_conditional = GaussianConditional(None)
        # self.entropy_bottleneck = EntropyBottleneck(N)
        # self.N = int(N)
        # self.M = int(M)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        executes the full forward pass for training. compress+decompress  
        returns dict with keys 'x_hat' and 'likelihoods'
        'likelihoods' is dict with keys 'y' and 'z'
        """
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        psi = self.h_s(z_hat)
        
        yhat = []
        y_likelihoods = []
        # needs to somehow keep track of y_likelihoods?
        # can we take shortcut in forward pass and only derive y_likelihoods from scctx
        # and use just y_hat with self.gaussian_conditional.quantize(y, "noise")
        
        # do we need to sort the channels according to energy?
        y_split = y.split(self.block_sizes, dim=1)
        for i, block in enumerate(y_split):
            block_hat = self.gaussian_conditional.quantize(
                block, "noise" if self.training else "dequantize")
            
            mean, scale = self.sccxt.predict_anchor(yhat, i, psi)
            _, y_l = self.gaussian_conditional(y_anchor, scale, mean=mean)

            yhat.append()

        yhat = torch.cat(yhat, dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x: torch.Tensor) -> Dict:
        """
        returns dict with keys 'strings' and 'shape'
        'strings' is list of compressed y and z
        'shape' is h,w shape of z
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        psi = self.h_s(z_hat)

        blocks = y.split(self.block_sizes, dim=1)
        y_hat=[]
        for i, block in enumerate(blocks):
            m,s = self.scctx.predict_anchor(y_hat,i,psi)
            anchor_hat = self.compress_anch(block,s,m,...)
            m,s=self.scctx.predict_non_anchor(y_hat,i,psi,anchor_hat)
            non_anchor_hat = self.compress_non_anch(block,s,m,...)
            encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
            # how these handle for in loop?
            y_string = encoder.flush()
            y_strings.append(y_string)
        # ....

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings: list, shape: torch.Size) -> Dict:
        """
        returns dict with key 'x_hat'
        """
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        psi = self.h_s(z_hat)
        # ....

        return {
            "x_hat": x_hat
        }
 
