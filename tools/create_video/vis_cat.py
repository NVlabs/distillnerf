import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F

# # Sample tensor
# input_tensor = torch.randn(1, 3, 256, 256)  # Example: batch size of 1, 3 channels, 256x256 image
# 
# # Resize the tensor to a new size (e.g., 512x512)
# new_size = (512, 512)
# output_tensor = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)

if not os.path.exists('combine'):
    os.makedirs('combine')

# img_type = ['../../GT/gt_rgb', '../../OURS/rgb', '../../OURS/depth', '../../OURS/clip', '../../OURS/dino', \
#         '../../OURS/languate_query_car', '../../OURS/languate_query_building', '../../OURS/languate_query_road', \
#         '../../OURS/occupancy', '../../OURS/semantic_occupancy']
# text_list = ['GT RGB', 'Rendered RGB', 'Rendered Depth', 'Rendered CLIP', 'Rendered DINOv2', \
#             'Text Query: Car', 'Text Query: Building', 'Text Query: Road', 'Occupancy', 'Semantic Occupancy']


# img_type = ['../../GT/gt_rgb', '../../OURS/rgb', '../../OURS/depths_input', '../../OURS/recon_clip', '../../OURS/recon_dino', \
#         '../../OURS/languate_query_car', '../../OURS/languate_query_building', '../../OURS/languate_query_road', \
#         '../../OURS/binary_occupancy_scene_0103', '../../OURS/semantic_occupancy_scene_0103']
# text_list = ['GT RGB', 'Rendered RGB', 'Rendered Depth', 'Rendered CLIP', 'Rendered DINOv2', \
#             'Text Query: Car', 'Text Query: Building', 'Text Query: Road', 'Occupancy', 'Semantic Occupancy']

img_type = ['../../GT/gt_rgb', '../../OURS/rgb', '../../OURS/depths_input', '../../OURS/recon_clip', \
        '../../OURS/languate_query_car', \
       '../../OURS/semantic_occupancy_scene_0103']
text_list = ['GT RGB', 'Rendered RGB', 'Rendered Depth', 'Rendered CLIP', \
            'Text Query: Car', 'Semantic Occupancy']

text_color = ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']
first_scene_frame_num = 40

num_frame = len(os.listdir(img_type[0]))

for one_frame in range(num_frame):

    fig, axes = plt.subplots(len(img_type), 1, figsize=(15, 6))  # Adjust figsize as needed
    for i, one_img_type in enumerate(img_type):
        if 'occupancy' in one_img_type:
            scene_idx = '0103'
            frame_idx = one_frame if one_frame < first_scene_frame_num else one_frame - first_scene_frame_num
            if "binary" in one_img_type:
                one_img_path = '{}/occ_{}_scene{}_binary.png'.format(one_img_type, one_frame, scene_idx)
            else:
                one_img_path = '{}/occ_{}_scene{}.png'.format(one_img_type, one_frame, scene_idx)
            one_img = mpimg.imread(one_img_path)
        else:
            one_img_path = '{}/{}.png'.format(one_img_type, one_frame)
            one_img = mpimg.imread(one_img_path)
        if 'occupancy' in one_img_type:
        # if one_img_type == 'voxels1':
            one_img = F.interpolate(torch.tensor(one_img).permute(2,0,1).unsqueeze(0), size=(109, 1162), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
            one_img = one_img.numpy()
        # (109, 1162, 4)
        # (125, 1162, 4)
        axes[i].imshow(one_img)
        # Add the patch to the Axes
        # import pdb; pdb.set_trace()
        # print(one_img_type.split('/')[-1], len(one_img_type.split('/')[-1]))
        # text_object = ax.text(0, 0, text, fontdict=fontdict)
        # fig.canvas.draw()
        text_object = axes[i].text(5, 4, text_list[i], horizontalalignment='left', verticalalignment='top', color=text_color[i], fontsize=10)
        fig.canvas.draw()
        bbox = text_object.get_window_extent()
        width = bbox.width
        if 'gt_rgb' in one_img_type:
            width += 5
        black_box = patches.Rectangle((-5, -5), width*1.2+4, 28, linewidth=1, edgecolor='none', facecolor='black')
        axes[i].add_patch(black_box)

        axes[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.savefig('combine/{}.png'.format(one_frame), bbox_inches='tight')
    # plt.show()
    plt.close()

exit()

num_file = len(os.listdir(directory))
file_name = directory + '/{}.png'.format(num_file)



frame_lst = [i for i in range(6)]
camera_lst = [i for i in range(6)]

for one_frame in frame_lst:
    one_frame_dir = './{}/'.format(one_frame)
    one_frame_image_lst = []
    one_frame_6_image_name = '{}.png'.format(one_frame)

    for one_camera in camera_lst:
        one_camera_path = one_frame_dir + '{}.png'.format(one_camera)
        one_frame_image_lst.append(mpimg.imread(one_camera_path))

    # Create a new figure
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))  # Adjust figsize as needed
    # Plot each image
    for i, image in enumerate([one_img for one_img in one_frame_image_lst]):
        axes[i].imshow(image)  # Assuming grayscale images
        axes[i].axis('off')
        axes.text(0.5, 0.5, text_list[i], horizontalalignment='center', verticalalignment='center', color=text_color[i])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(one_frame_6_image_name)

    # crop the white background
    combined_image = Image.open(one_frame_6_image_name)
    grayscale_image = np.array(combined_image.convert('L'))
    # Find bounding box of non-white pixels
    coords = np.argwhere(grayscale_image < 255)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # Add 1 to make it inclusive
    # Crop the image
    cropped_image = combined_image.crop((y0, x0, y1, x1))
    # Save the cropped image
    cropped_image.save(one_frame_6_image_name)

    # plt.show()
    plt.close()