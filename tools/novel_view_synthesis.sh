

#  List of indices
INDICES=(0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60)

# Run the Python script for each index
for INDEX in "${INDICES[@]}"
do
    echo "Running with index: $INDEX"

    # RGB novel view - model with parameterized space
    python ./tools/novel_view_synthesis.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper.py ./checkpoint/model.pth "$INDEX" --cfg-options model.visualize_imgs=True

    # # RGB novel view - model without parameterized space
    # python ./tools/novel_view_synthesis.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace.py ./checkpoint/model_linearspace_virtual_cam.pth "$INDEX" --cfg-options model.visualize_imgs=True

    # # DINO novel view
    # python ./tools/novel_view_synthesis.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_dino.py ./checkpoint/model_linearspace_dino.pth --cfg-options model.visualize_foundation_model_feat=True 

    # # CLIP novel view
    # python ./tools/novel_view_synthesis.py ./projects/DistillNeRF/configs/model_wrapper/model_wrapper_linearspace_clip.py ./checkpoint/model_linearspace_clip.pth --cfg-options model.visualize_foundation_model_feat=True 



done
