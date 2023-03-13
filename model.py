from typing import Dict, List
import torch

from compressai.ans import BufferedRansEncoder, RansDecoder

from compressai.models import JointAutoregressiveHierarchicalPriors

from modules import SCCTXModel

from neural_nets import g_analysis, g_synthesis, h_analysis, h_synthesis

class ELICModel(JointAutoregressiveHierarchicalPriors):
    """
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        block_sizes: List of blocks sizes to use in SCCTX-model
    """
    def __init__(self, N=192, M=192, block_sizes: List[int] = None, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.block_sizes = block_sizes if block_sizes else [16,16,32,64,M-128]

        # actually we can just use the inherited h_a and h_s from
        # JointAutoregressiveHierarchicalPriors
        # as they stated in last paragraph of supplementary material 1.1
        # self.h_a = h_analysis.Model(N)
        # self.h_s = h_synthesis.Model(M)

        self.g_a = g_analysis.Model(self.M, self.N)
        self.g_s = g_synthesis.Model(self.M, self.N)
        self.scctx = SCCTXModel(self.M, self.block_sizes)     

        # inherited
        # self.gaussian_conditional = GaussianConditional(None)
        # self.entropy_bottleneck = EntropyBottleneck(N)
        # self.N = int(N)
        # self.M = int(M)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Executes the model forward pass with one pass latent encoding
        returns dict with keys 'x_hat' and 'likelihoods'
        'likelihoods' is dict with keys 'y' and 'z'
        """
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        psi = self.h_s(z_hat)
        
        yhat = []
        y_likelihoods = []
        
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
            _, y_l = self.gaussian_conditional(block, scale, means=mean)

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
        Compresses the input x to bytestrings.
        Compression is done in img wise mannor where every image in batch is compressed serially.

        returns dict with keys 'strings' and 'shape'
        'strings' is list of compressed y and z. One element for each img in batch
        'shape' is h,w shape of z
        """
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        psi = self.h_s(z_hat) # can we use directly z here

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_img(y[[i],:,:,:], psi[[i],:,:,:])
            y_strings.append(string)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }
    

    def _compress_img(self, x: torch.Tensor, psi: torch.Tensor) -> str:
        """
        compresses one image to bytestring
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        blocks = x.split(self.block_sizes, dim=1)
        y_hat = []

        for i, block in enumerate(blocks):
            m, s = self.scctx.predict_anchor(y_hat, i, psi)
            anchor_hat = self.compress_anchor(block,s,m,symbols_list,indexes_list)
            m, s = self.scctx.predict_non_anchor(y_hat,i,psi,anchor_hat)
            non_anchor_hat = self.compress_nonanchor(block,s,m,symbols_list,indexes_list)
            y_hat.append(anchor_hat+non_anchor_hat)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        return y_string

    def decompress(self, strings: list, shape: torch.Size) -> Dict:
        """
        Decompresses batch of bytestring into images.
        Decompression is done sequentially for each image in batch.

        returns dict with key 'x_hat'
        """
        assert isinstance(strings, list) and len(strings) == 2
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        psi = self.h_s(z_hat)

        y_batches = []
        for i, y_string in enumerate(strings[0]):
            y_b = self._decompress_image(y_string, psi[[i],:,:,:])
            y_batches.append(y_b)

        y_hat = torch.cat(y_batches, dim=0)
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}
    

    def _decompress_image(self, batch: str, psi: torch.Tensor) -> torch.Tensor:
        """
        Decompress one bytestring into image
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(batch)

        y_hat = []
        for i in range(len(self.block_sizes)):
            m, s = self.scctx.predict_anchor(y_hat, i, psi)
            anchor_hat = self.decompress_anchor(s,m,decoder,cdf,cdf_lengths,offsets)
            m, s = self.scctx.predict_non_anchor(y_hat, i, psi, anchor_hat)
            non_anch_hat = self.decompress_nonanchor(s,m,decoder,cdf,cdf_lengths,offsets)
            block_hat = anchor_hat + non_anch_hat
            y_hat.append(block_hat)

        y_hat = torch.cat(y_hat, dim=1)
        return y_hat
 

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
            scales_anchor.device).to(scales_anchor.dtype) + means_squeeze
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
            scales_nonanchor.device).to(scales_nonanchor.dtype) + means_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat

    def ckbd_anchor_sequeeze(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2], dtype=y.dtype).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2], dtype=y.dtype).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2], dtype=anchor.dtype).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2], dtype=nonanchor.dtype).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor
