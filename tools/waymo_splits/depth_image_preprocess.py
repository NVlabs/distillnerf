from glob import glob
from os.path import join
import pdb
import shutil, os

# ln -s /home/letian/DistillNeRF_rebuttal/EmerNeRF-20240804T012020Z-001/EmerNeRF/data/waymo/processed/training_0_0_0_0_0_0 /home/letian/DistillNeRF_0612/DistillNeRF/data/waymo/waymo_format/waymo/
tfrecord_pathnames = sorted(
    glob(join("./data/waymo/waymo_format/emernerf_notr", '*.tfrecord')))
total_list = open("tools/waymo_splits/waymo_train_list.txt", "r").readlines()
total_list = [line.strip() for line in total_list]
scene_ids = [total_list.index(pathanme.split('/')[-1].split('.')[0]) for pathanme in tfrecord_pathnames]

for scene_count, scene_id in enumerate(scene_ids):
    depth_imgs = sorted(glob(join(f'./data/waymo/waymo_format/waymo/training_0_0_0_0_0_0/depths/{str(scene_id).zfill(3)}/images/', '*.npy')))
    for depth_img in depth_imgs:
        filename = depth_img.split('/')[-1]
        frame_idx = int(filename.split('_')[0])
        cam_idx = int(filename.split('_')[1].split('.')[0])
        save_dir = f'./data/waymo/kitti_format/emernerf_notr/depth_img_{cam_idx}/'
        save_name = f'0{str(scene_count).zfill(3)}{str(frame_idx).zfill(3)}.npy'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # copy one file to another place with a new name
        # import pdb; pdb.set_trace()
        shutil.copy(depth_img, os.path.join(save_dir, save_name))