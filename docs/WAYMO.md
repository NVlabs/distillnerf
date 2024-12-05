# Waymo dataset Preparation
Following EmerNeRF, we used the Waymo NOTR dataset to evaluate our DistillNeRF.

## 1. Register on Waymo Open Dataset

### Sign Up for a Waymo Open Dataset Account and Install gcloud SDK

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details.

### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/waymo_format   
mkdir -p ./data/waymo/kitti_format 
```

## 2. Download the raw data

Below we download the Waymo data to the directory of `./data/waymo`. Another option is the download the data in your preferred directory, and create symbolic link to it
```
ln -s [YOUR-DATA-DIR] ./data/waymo
```

We first introduce an example to download the any specified data samples as follows:

### Downloading Specific Scenes from Waymo Open Dataset

For example, to obtain the 114th, 700th, and 754th scenes from the Waymo Open Dataset, execute:

```shell
python datasets/download_waymo.py \
    --target_dir ./data/waymo/raw \
    --scene_ids 114 700 754
```

### Download the NOTR split Proposed by EmerNeRF

The actual data needed in our paper is The NOTR dataset, which comes in multiple splits. Specify the split_file argument to download your desired split:

- **Static32 Split:**
```
python tools/download_waymo.py --target_dir ./data/waymo/waymo_format --split_file tools/waymo_splits/static32.txt
```
- **Dynamic32 Split:**
```
python tools/download_waymo.py --target_dir ./data/waymo/waymo_format --split_file tools/waymo_splits/dynamic32.txt
```
- **Diverse56 Split:**
```
python tools/download_waymo.py --target_dir ./data/waymo/waymo_format --split_file tools/waymo_splits/diverse56.txt
```
Ensure you modify the paths and filenames to align with your project directory structure and needs.



## 3. Download the Sky Mask
We use ViT-adapater to extract sky masks. We refer readers to [their repo](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) for more details. Follow [EmerNeRF](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) to download precomputed sky masks for the NOTR dataset. After you download them, unzip them and put them under `data/waymo/waymo_format/`. Example scripts to download and unzip the files is:

```
# download the sky masks from https://drive.google.com/drive/folders/11hJDPqd5XhaI7EGbq0twhb0sgfUrmpQQ?usp=share_link or:
# gdown 1ZEU1B_MdTeFHC2EM97jatnWfE7zkWm8a # static32
# gdown 1zJBWeEoAFvEfD02sQsUrRqT8r28oAyrX # dynamic32
# gdown 1nfSTIxK-RFffx-rDLPp-LeB5XBO9GN7K # diverse56

# in the downloading directory, run
tar -xf diverse56.tar.gz
for file in diverse56/*.tar.gz; do tar -xvf $file -C ./data/waymo/waymo_format; done
rm -rf diverse56
rm diverse56.tar.gz

tar -xf static32.tar.gz
for file in static32/*.tar.gz; do tar -xvf $file -C ./data/waymo/waymo_format; done
rm -rf static32 && rm static32.tar.gz

tar -xf dynamic32.tar.gz
for file in dynamic32/*.tar.gz; do tar -xvf $file -C ./data/waymo/waymo_format; done
rm -rf dynamic32 && rm dynamic32.tar.gz
```


## 3. Data Preprocess
We follow mmdetection3D to preprocess the waymo dataset to be in the KITTI format.

- **Preprocess the Raw Data:**

``` 
TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

Some Explanation of this command: this command execute:
```
1. store the data in the kitti format, 
   1. things that we modified
      1. tools/create_data.py: splits = ["emernerf_notr"]
      2. tools/data_converter/waymo_converter.py: save_lidar() -> parse_range_image_and_camera_projection(frame) returns 4 variables
   2. before the next step: generate frame indexes, for generaing the data info pkl: notr_frame_idx
      1. put to ./data/waymo/kitti_format/ImageSets/notr_frame_idx.txt
2. save the data info pkl
   1. things to modify
      1. tools/data_converter/kitti_converter: train_img_ids = _read_imageset_file(str(imageset_folder / 'notr_frame_idx.txt'))
3. save the ground-truth data base and info pkl
   1. tools/data_converter/create_gt_databse: split='training' -> split='emernerf_notr'
```

- **Preprocess the Sky Masks:**
```
python tools/waymo_splits/sky_mask_preprocess.py 
```

## 5. Run the Code

1. **Training Script**

With parameterized space
```
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_waymo.py --seed 0 --work-dir=../work_dir_debug
```

No parameterized space
```
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_waymo.py --seed 0 --work-dir=../work_dir_debug
```

2. **Visualize the Images**

We used the checkpoint `./checkpoint/model.pth`
```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_waymo.py ./checkpoint/model.pth --cfg-options model.visualize_imgs=True
```

We can also use the checkpoint `./checkpoint/model_linearspace.pth`, where we do not use the parameterized space
```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_waymo.py ./checkpoint/model_linearspace.pth --cfg-options model.visualize_imgs=True
```

3. **Visualize the Images after Color Calibration**

```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_waymo.py ./checkpoint/model_linearspace.pth --cfg-options model.recoloring_img=True model.visualize_imgs=True
```


4. **Testing**
   
We used the checkpoint `./checkpoint/model.pth`

```
python ./tools/test.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_waymo.py ./checkpoint/model.pth --eval psnr
```

5. **Testing via the Visualization Code**
   
This command does not launch complicated mmdetection3d process, so it is more memory friendly and better suited in local machine. But only `PSNR` and `SSIM` are supported by now

With parameterized space
```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_waymo.py ./checkpoint/model.pth
```

Without parameterized space
```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_waymo.py ./checkpoint/model_linearspace.pth
```


6. **More Efficient Visualization (TODO)**

This command only load one-frame data, should be more efficient in the data loading process. But it has not been fully tested yet
```
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_waymo.py ./checkpoint/model.pth --cfg-options data.test.num_prev_frames=0 data.test.num_next_frames=0 model.num_input_seq=1 model.target_cam_temporal_idx=0 model.input_cam_temporal_index=0
```
