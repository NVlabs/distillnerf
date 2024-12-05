import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin.ops.spc as spc_ops
import time
do_print = False

def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)

class SpcConvWrapper(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_vectors, op, jump=0, bias=True, activate=False, do_norm=False):
        '''
        op = spc_ops.Conv3d or spc_ops.ConvTranspose3d
        '''
        super(SpcConvWrapper, self).__init__()
        self.conv = op(in_channel, out_channel, kernel_vectors, jump=jump, bias=bias)
        self.activate, self.do_norm = activate, do_norm
        if self.activate:
            self.activation = nn.LeakyReLU(0.2)
        if self.do_norm:
            self.norm = nn.LayerNorm(out_channel)

    def forward(self, octrees, point_hierarchies, level, pyramids, exsum, x):
        # import pdb; pdb.set_trace()
        if self.conv.kernel_vectors.dtype != torch.int16:
            self.conv.kernel_vectors = self.conv.kernel_vectors.to(dtype=torch.int16)
        out, _ = self.conv(octrees, point_hierarchies, level, pyramids, exsum, x)
        if self.activate:
            out = self.activation(out)
        if self.do_norm:
            out = self.norm(out)
        return out

class SparseVoxelTopDownConvNet(nn.Module):
    def __init__(self, config):
        super(SparseVoxelTopDownConvNet, self).__init__()
        self.config = config

        # create kernel vectors for sparse Conv3d
        vectors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vectors.append([i, j, k])
        # kernel_vectors = torch.LongTensor(vectors).to(dtype=torch.int16)
        kernel_vectors = torch.LongTensor(vectors)


        self.octree_levels = config.octree_levels
        feat_size = (config.feature_size - 1) // len(self.octree_levels) + 1

        self.model_hidden_dim = config.model_hidden_dim
        self.octree_final_dim = self.config.octree_final_dim

        self.init_conv = nn.ModuleList()
        self.processing_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        reverse_octree_levels = self.octree_levels[::-1]
        for ind, level in enumerate(reverse_octree_levels):
            self.init_conv.append(
                SpcConvWrapper(feat_size, self.model_hidden_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=True)
            )

            in_channel = self.model_hidden_dim * 2 if ind > 0 else self.model_hidden_dim
            self.processing_layers.append(
                SpcConvWrapper(in_channel, self.model_hidden_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=True, do_norm=True)
            )
            for proc_layer_ind in range(self.config.num_layers-1):
                self.processing_layers.append(
                    SpcConvWrapper(self.model_hidden_dim, self.model_hidden_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=True, do_norm=True)
                )
            if ind > 0:
                jump = self.octree_levels[ind]-self.octree_levels[ind-1]

                self.downsample_layers.append(
                    SpcConvWrapper(self.model_hidden_dim, self.model_hidden_dim, kernel_vectors, spc_ops.Conv3d, jump=jump, bias=True, activate=True),
                )

        self.output_head = SpcConvWrapper(self.model_hidden_dim, self.octree_final_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=False)
        # self.output_head_level7 = SpcConvWrapper(self.model_hidden_dim, self.octree_final_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=False)
        # self.output_head_level9 = SpcConvWrapper(self.model_hidden_dim, self.octree_final_dim, kernel_vectors, spc_ops.Conv3d, jump=0, bias=True, activate=False)


    def forward(self, inputs, octrees, point_hierarchies, levels, pyramids, exsum, dist=None, div_batch=1):
        #     [torch.Size([1045536, 33]), torch.Size([88346, 33])]
        #               x,              octrees, point_hierarchies,     level, pyramids, exsum
        # [octree_feat9, octree_feat7], octrees9, point_hierarchies9, [7, 9], pyramids9, exsum9 
        '''
            inputs = [featN, ...feat0] where featN is the finest octree level
        '''
        pyramids = pyramids.unsqueeze(0)
        out_features = []
        processing_layer_ind = 0
        reverse_octree_levels = self.octree_levels[::-1]
        for ind, level in enumerate(reverse_octree_levels):
            torch.cuda.synchronize()
            start = time.time()
            out = self.init_conv[ind](octrees, point_hierarchies, level, pyramids, exsum, inputs[ind])
            if do_print: print("            Multi-view fusion - Sparse Convolution - init_conv:", round(time.time()-start,6))

            torch.cuda.synchronize()
            start = time.time()
            if ind > 0:
                prev_out = self.downsample_layers[ind-1](octrees, point_hierarchies, reverse_octree_levels[ind-1], pyramids, exsum, prev_out)
                out = torch.cat([prev_out, out], dim=1)
            torch.cuda.synchronize()
            if do_print: print("            Multi-view fusion - Sparse Convolution - downsample layer:", round(time.time()-start,6))

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(self.config.num_layers): # 2
                out = self.processing_layers[processing_layer_ind](octrees, point_hierarchies, level, pyramids, exsum, out)
                processing_layer_ind += 1
            torch.cuda.synchronize()
            if do_print: print("            Multi-view fusion - Sparse Convolution - process layer:", round(time.time()-start,6))
            prev_out = out


        torch.cuda.synchronize()
        start = time.time()
        final_out = self.output_head(octrees, point_hierarchies, reverse_octree_levels[-1], pyramids, exsum, out)
        torch.cuda.synchronize()
        if do_print: print("            Multi-view fusion - Sparse Convolution - output layer:", round(time.time()-start,6))
        return final_out
        # return self.output_head_level7(octrees, point_hierarchies, reverse_octree_levels[-1], pyramids, exsum, out)
                # self.output_head_level9(octrees, point_hierarchies, reverse_octree_levels[0], pyramids, exsum, prev_out)
