# w\o parameterization, w\ depth disllation, w\o virtual cam distillation
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper.py ./checkpoint/model.pth --cfg-options model.visualize_imgs=True

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation
python ./tools/visualization.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace.py ./checkpoint/model_linearspace.pth --cfg-options model.visualize_imgs=True
