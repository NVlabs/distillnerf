
## NuScenes Dataset preparation

The NuScenes dataset is a popular autonomous driving dataset. Follow the steps below to set it up and prepare it for use:

1. **Download NuScenes Data**

The dataset can be downloaded from the [NuScenes official website](https://www.nuscenes.org/nuscenes). Once downloaded, unzip it and place it at your favored place.

2. **Setup Directories**

Create directories for NuScenes data, and create a symbolic link to your saved Nuscenes data (symbolic link makes my life easier, I like it)

```shell
cd $DistillNeRF_Repo
mkdir -p data/nuscenes
ln -s $PATH_TO_NUSCENES data/nuscenes
```

3. **Preprocess data for mmdetection3d**
   
For NuScenes mini set, run
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```
Alternatively, you can also directly download the preprocessed files [here](https://drive.google.com/drive/folders/1Ohy6Z9-NkWW0NkKH5t1QWXm6Pk9-Yo6z?usp=sharing), and put them under `data/nuscenes/`.
Note that, please only use these data if you have agreed to the terms for non-commercial use from nuScenes https://www.nuscenes.org/nuscenes. The preprocessed dataset are under the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


For NuScenes full set, run
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

4. **Download Sky Masks**

Sky masks help in addressing the ill-defined depth for sky. Follow [EmerNeRF](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NUSCENES.md) to download the sky masks, and unzip them in the `data/nuscenes/` directory.

Once done, you're all set to integrate and use the NuScenes dataset in your project!
