from typing import Dict, List
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

        self.g_s = nn.Sequential(
            
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
            
            # should we call also the anchor prediction to get these likelihoods?
            # mean, scale = self.sccxt.predict_anchor(yhat, i, psi)
            # _, y_l = self.gaussian_conditional(block, scale, mean=mean)

            # mask anchor before param agg. like in checkerboard.
            mean, scale = self.scctx.predict_non_anchor(yhat, i, psi, block, mask=True)
            _, y_l = self.gaussian_conditional(block, scale, mean=mean)

            yhat.append(block_hat)
            y_likelihoods.append(y_l)

        yhat = torch.cat(yhat, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(yhat)

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
        y_hat = []

        for i, block in enumerate(blocks):
            m, s = self.scctx.predict_anchor(y_hat, i, psi)
            anchor_hat = self.compress_anchor(block,s,m,symbols_list,indexes_list)
            m, s = self.scctx.predict_non_anchor(y_hat,i,psi,anchor_hat)
            non_anchor_hat = self.compress_nonanchor(block,s,m,symbols_list,indexes_list)
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

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()

        y_strings = strings[0] #[0] ?
        assert len(y_strings) == len(self.block_sizes)
        z_strings = strings[1]
        decoder.set_stream(y_strings)

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        psi = self.h_s(z_hat)

        y_hat = []
        for i, block in enumerate(y_strings):
            m, s = self.scctx.predict_anchor(y_hat, i, psi)
            anchor_hat = self.decompress_anchor(s,m,decoder,cdf,cdf_lengths,offsets)
            m, s = self.scctx.predict_non_anchor(y_hat, i, psi, anchor_hat)
            non_anch_hat = self.decompress_nonanchor(s,m,decoder,cdf,cdf_lengths,offsets)
            block_hat = anchor_hat + non_anch_hat
            y_hat.append(block_hat)

        y_hat = torch.cat(y_hat, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat
        }
 

    #####################################
    # Following is adopted from 
    # https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/
    #####################################
    def compress_anchor(self, 
                        anchor: torch.Tensor, 
                        scales: torch.Tensor, 
                        means: torch.Tensor, 
                        symbols: List, 
                        indexes_list: List) -> torch.Tensor:

        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_squeeze = self.ckbd_anchor_sequeeze(scales)
        means_squeeze = self.ckbd_anchor_sequeeze(means)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_squeeze)
        symbols.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_squeeze)
        return anchor_hat

    def compress_nonanchor(self, 
                            nonanchor: torch.Tensor, 
                            scales_nonanchor: torch.Tensor, 
                            means_nonanchor: torch.Tensor,
                            symbols_list: List, 
                            indexes_list: List) -> torch.Tensor:
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat + means_squeeze)
        return nonanchor_hat

    def decompress_anchor(self, 
                            scales_anchor: torch.Tensor, 
                            means_anchor: torch.Tensor, 
                            decoder: RansDecoder, 
                            cdf: List, 
                            cdf_lengths: List, 
                            offsets: List) -> torch.Tensor:
        scales_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_squeeze.shape).to(
            scales_anchor.device) + means_squeeze
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(self, 
                            scales_nonanchor: torch.Tensor, 
                            means_nonanchor: torch.Tensor, 
                            decoder: RansDecoder, 
                            cdf: List, 
                            cdf_lengths: List, 
                            offsets: List) -> torch.Tensor:
        scales_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_squeeze.shape).to(
            scales_nonanchor.device) + means_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat

    def ckbd_anchor_sequeeze(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor
