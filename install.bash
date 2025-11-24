conda create -n melody python=3.10
conda activate melody
which pip
echo "Will install in this environment. Sleeping for 10 seconds. Press Ctrl+C to cancel."
sleep 60
# TODO HERE: install pytorch via one of the following commands:
# 1. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # if your NVIDIA driver is old
# 2. pip install torch torchvision torchaudio # if your NVIDIA driver is new enough.
# torch 2.5 is OK, but newer versions is not tested.
pip install transformers einops ninja seaborn loguru echo_logger
# TODO HERE: install cupy-cuda12x or cupy-cuda11x according to your NVIDIA driver version
pip install scikit-learn tensorboard matplotlib jupyter tqdm pandas accelerate fire
conda install zlib
pip install h5py h5sparse pyBigWig tensorboardX torchvision medpy pytabix pyfaidx wandb plotly liftover # swanlab (optional, if your wandb works well, you do not need swanlab)
pip install selene-sdk
pip cache purge # cleanup