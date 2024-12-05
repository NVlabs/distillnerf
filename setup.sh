pip install -r requirements_distillnerf.txt

pip install pytz==2023.4
pip install rich==13.4.2
pip install requests==2.28.2

pip install setuptools==60.2.0
pip install openmim
pip install mmcv-full==1.7
pip install mmdet==2.28
pip install mmsegmentation==0.30
pip install mmdet3d==1.0.0rc6

pip install networkx==2.5
pip install motmetrics
pip install IPython
pip install casadi
pip install torchmetrics
pip install hydra-core==1.3.2
pip install 'git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f'
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu116.html
pip install yapf==0.40.1
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
pip install git+https://github.com/nerfstudio-project/nerfacc.git@8340e19daad4bafe24125150a8c56161838086fa
pip install huggingface_hub==0.19.4
pip install safetensors
pip install lpips
pip install einops
pip install open3d
pip install git+https://github.com/openai/CLIP.git
pip install numpy==1.22.4
pip install protobuf==3.20.0
pip uninstall -y lyft-dataset-sdk
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

pip uninstall -y opencv-python
# pip install "opencv-python-headless<4.3" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python-headless

pip install matplotlib==3.5.3