# If you're running locally with limited computation, add `model.num_camera=1 data.train.subselect_group_num=1 data.val.subselect_group_num=1 data.test.subselect_group_num=1` to the end of the scipt

''' RGB Synthesis '''
# w\o parameterization, w\o depth disllation, w\o virtual cam distillation
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_no_depth_distilll.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\o parameterization, w\ depth disllation, w\o virtual cam distillation
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\ parameterization, w\ depth disllation, w\ virtual cam distillation
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_virtual_cam.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\o parameterization, w\ depth disllation, w\ virtual cam distillation
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_virtual_cam_linearspace.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation, larger resoltuion
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_400_224.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 


''' Foundation Model Feature Lifting '''
# w\o parameterization, w\ depth disllation, w\o virtual cam distillation, w\ CLIP feature prediction
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_clip.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\o parameterization, w\ depth disllation, w\o virtual cam distillation, w\ DINO feature prediction
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_dino.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation, w\ CLIP feature prediction
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_400_224_clip.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation, w\ DINO feature prediction
python tools/train.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_400_224_dino.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True 
