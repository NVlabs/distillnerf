import os, math
import torch
import clip
from torchvision import transforms
from PIL import Image
from torch import Tensor, nn
from typing import Tuple
import torch.nn.modules.utils as nn_utils
import types
import numpy as np
import torch.nn.functional as F

def save_image_horizontally(recon_imgs, directory='', name='rgb', cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # create directory and file name
    directory += name
    if not os.path.exists(directory):
        os.makedirs(directory)
    num_file = len(os.listdir(directory))
    file_name = directory + '/{}.png'.format(num_file)

    # Create a new figure
    N_cam = recon_imgs.shape[0]
    if N_cam == 3:
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))  # Adjust figsize and colurms as needed
    else:
        fig, axes = plt.subplots(1, 6, figsize=(15, 3))  # Adjust figsize and colurms as needed
    # Plot each image
    for i, image in enumerate([one_img for one_img in recon_imgs]):
        if cmap is not None:
            axes[i].imshow(image, cmap=cmap)
        else:
            axes[i].imshow(image)  # Assuming grayscale images
        axes[i].axis('off')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # print(file_name)
    plt.savefig(file_name)

    # crop the white background
    combined_image = Image.open(file_name)
    grayscale_image = np.array(combined_image.convert('L'))
    # Find bounding box of non-white pixels
    coords = np.argwhere(grayscale_image < 255)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # Add 1 to make it inclusive
    # Crop the image
    cropped_image = combined_image.crop((y0, x0, y1, x1))
    # Save the cropped image
    cropped_image.save(file_name)

    # plt.show()
    plt.close()

class DenseCLIPWrapper:
    def __init__(self, model_name="ViT-B/16", stride: int = 8, input_size=(640, 960)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model_name, device=self.device)

        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    input_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.preprocess = preprocess

        self.model = DenseCLIPWrapper.wrap_dense_model(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device).float()

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def tokenize_text(self, text_list):
        return clip.tokenize(text_list).to(self.device)

    def get_image_embeddings(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return image_features

    def get_text_embeddings(self, text_list):
        text_tokens = self.tokenize_text(text_list)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def get_similarity(self, image_path, text_list, temperature=1.0):
        image_features = self.get_image_embeddings(image_path)
        # image_features = np.load("data/waymo/processed/training/023/lseg/084_1.npy")
        # image_features = torch.from_numpy(image_features).to(self.device).float()
        # image_features = image_features.unsqueeze(0)
        text_list = [f"a photo of a {text}" for text in text_list]
        text_features = self.get_text_embeddings(text_list)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity_logits = image_features @ text_features.T
        similarity = (similarity_logits / temperature).softmax(dim=-1)
        return similarity

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        # from vit extractor
        def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
            npatch = x.shape[1] - 1
            N = self.positional_embedding.shape[0] - 1
            if npatch == N and w == h:
                return self.positional_embedding
            class_pos_embed = self.positional_embedding[0]
            patch_pos_embed = self.positional_embedding[1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            output = torch.cat(
                (class_pos_embed.unsqueeze(0).unsqueeze(0), patch_pos_embed), dim=1
            )
            return output

        return interpolate_pos_encoding

    @staticmethod
    def _dense_forward():
        def forward(self, x: Tensor) -> Tensor:
            B, nc, h, w = x.shape
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            h0, w0 = x.shape[-2:]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            # interpolate position embedding
            x = x + self.interpolate_pos_encoding(x, w, h)

            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND

            ### ---modification--- ###
            # from f3rm:
            # https://github.com/f3rm/f3rm/blob/218f290a6dd3d9a5db27402ad640e08590b11a10/f3rm/features/clip/model.py#L277
            *layers, last_resblock = self.transformer.resblocks
            penultimate = nn.Sequential(*layers)
            x = penultimate(x)
            v_in_proj_weight = last_resblock.attn.in_proj_weight[
                -last_resblock.attn.embed_dim :
            ]
            v_in_proj_bias = last_resblock.attn.in_proj_bias[
                -last_resblock.attn.embed_dim :
            ]
            v_in = F.linear(last_resblock.ln_1(x), v_in_proj_weight, v_in_proj_bias)
            x = F.linear(
                v_in,
                last_resblock.attn.out_proj.weight,
                last_resblock.attn.out_proj.bias,
            )
            ##### ----- #####

            x = x.permute(1, 0, 2)  # LND -> NLD

            # Extract the patch tokens, not the class token
            x = x[:, 1:, :]
            x = self.ln_post(x)
            if self.proj is not None:
                # This is equivalent to conv1d
                x = x @ self.proj
            x = x.reshape(B, h0, w0, -1)

            # original:
            # x = self.transformer(x)
            # x = x.permute(1, 0, 2)  # LND -> NLD

            # x = self.ln_post(x[:, 0, :])

            # if self.proj is not None:
            #     x = x @ self.proj
            return x

        return forward

    @staticmethod
    def wrap_dense_model(model: nn.Module, stride: int) -> nn.Module:
        patch_size = model.visual.conv1.kernel_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.visual.conv1.stride = stride
        # fix the positional encoding code
        model.visual.interpolate_pos_encoding = types.MethodType(
            DenseCLIPWrapper._fix_pos_enc(patch_size, stride), model.visual
        )
        # override forward fn
        model.visual.forward = types.MethodType(
            DenseCLIPWrapper._dense_forward(), model.visual
        )
        return model


def load_color_pca(file_path):
    # Load the .npz file
    data = np.load(file_path)
    # Access the dictionary of arrays
    data_dict = dict(data)
    # Access individual arrays from the dictionary
    color_reduction_mat = data_dict['color_reduction_mat']
    color_clip_min = data_dict['color_clip_min']
    color_clip_max = data_dict['color_clip_max']
    return color_reduction_mat, color_clip_min, color_clip_max

# load pca
def load_pca(file_path):
    # Load the .npz file
    data = np.load(file_path)
    # Access the dictionary of arrays
    data_dict = dict(data)
    # Access individual arrays from the dictionary
    reduction_mat = data_dict['reduction_mat']
    min = data_dict['min']
    max = data_dict['max']
    return reduction_mat, min, max


def reduce_feature_dim(feature, reduction_mat, fm_min, fm_max):
    reduced_feature = feature @ reduction_mat
    reduced_feature = (reduced_feature - fm_min) / (fm_max - fm_min)
    return reduced_feature


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)

''' visualization of foundation model features (clip or dino) '''
def visualize_foundation_feat(recon_imgs, novel_view=False, feat='clip', directory='./vis/'):
    color_reduction_mat, color_fm_min, color_fm_max = get_robust_pca(recon_imgs.reshape(-1, 64))
    if novel_view:   # recon_imgs.shape[1] == 18
        for i in range(3):
            reduced_recon_fm_imgs_vis = reduce_feature_dim(recon_imgs[:, i*6:(i+1)*6].cpu(), color_reduction_mat.cpu(), color_fm_min.cpu(), color_fm_max.cpu())[0]
            save_image_horizontally(reduced_recon_fm_imgs_vis, directory=directory, name='novel_{}'.format(feat))
    else:
        reduced_recon_fm_imgs_vis = reduce_feature_dim(recon_imgs.cpu(), color_reduction_mat.cpu(), color_fm_min.cpu(), color_fm_max.cpu())[0]
        save_image_horizontally(reduced_recon_fm_imgs_vis, directory=directory, name='recon_{}'.format(feat))


def language_query(recon_imgs, directory='./vis/'):
    ''' load the pca reduction matrix that was used to reduce the clip feature dimension to 64 '''
    # model_path = f"../EmerNerf_clip_85.pth"
    # model_weight = torch.load(model_path, map_location=torch.device('cpu'))
    # reduction_mat = model_weight['model']['feats_to_target_reduction_mat'].cpu().detach().numpy()
    # clip_min = model_weight['model']['feat_min'].cpu().detach().numpy()
    # clip_max = model_weight['model']['feat_max'].cpu().detach().numpy()
    pca_path = './aux_models/clip/clip_reduction_pca.npz'
    reduction_mat, clip_min, clip_max = load_pca(pca_path)
    
    ''' load the clip model to encode the text queries '''
    model_name = "ViT-L/14@336px"
    # queries = ["other","sky","person","car","red car","truck","road","bicycle","motorcycle","bus","traffic light","traffic sign"]
    # queries = ["car", 'bus', 'road', 'vegetation', 'terrain', 'barrier']
    queries = ['road', 'car', 'building', 'pedestrian', 'vegetation']
    img_shape = [602, 1071]
    stride = 7
    clip_wrapper = DenseCLIPWrapper(
        model_name=model_name, stride=stride, input_size=img_shape
    )
    temperature = 0.001
    text_list = [f"a photo of a {text}" for text in queries]
    text_features = clip_wrapper.get_text_embeddings(text_list)

    ''' calculate similarity '''
    # reduce the text feature dimension
    text_reduced_features = reduce_feature_dim(text_features.cpu(), reduction_mat, clip_min, clip_max)
    # Normalize the features
    recon_imgs /= recon_imgs.norm(dim=-1, keepdim=True)
    text_reduced_features /= text_reduced_features.norm(dim=-1, keepdim=True)
    # Compute similarity
    similarity_logits = recon_imgs.cpu() @ text_reduced_features.T
    similarity = (similarity_logits / temperature).softmax(dim=-1)

    ''' save the similarity image '''
    for query_idx, query in enumerate(queries):
        similarity_for_one_query = similarity[0, ..., query_idx]
        similarity_for_one_query /= similarity_for_one_query.reshape(6,-1).max(dim=-1)[0].unsqueeze(1).unsqueeze(1)
        save_image_horizontally(similarity_for_one_query, directory=directory, name=f'languate_query_{query}', cmap='turbo')
