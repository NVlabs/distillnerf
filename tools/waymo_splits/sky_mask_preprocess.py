from glob import glob
from os.path import join
import pdb
import shutil, os

tfrecord_pathnames = sorted(
    glob(join("./data/waymo/waymo_format/emernerf_notr", '*.tfrecord')))
total_list = open("tools/waymo_splits/waymo_train_list.txt", "r").readlines()
total_list = [line.strip() for line in total_list]
scene_ids = [total_list.index(pathanme.split('/')[-1].split('.')[0]) for pathanme in tfrecord_pathnames]

for scene_count, scene_id in enumerate(scene_ids):
    import pdb; pdb.set_trace()
    sky_masks = sorted(glob(join(f'./data/waymo/waymo_format/waymo/training/{str(scene_id).zfill(3)}/sky_masks/', '*.png')))
    for sky_mask in sky_masks:
        filename = sky_mask.split('/')[-1]
        frame_idx = int(filename.split('_')[0])
        cam_idx = int(filename.split('_')[1].split('.')[0])
        save_dir = f'./data/waymo/kitti_format/emernerf_notr/sky_mask_{cam_idx}/'
        save_name = f'0{str(scene_count).zfill(3)}{str(frame_idx).zfill(3)}.png'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # copy one file to another place with a new name
        shutil.copy(sky_mask, os.path.join(save_dir, save_name))