read  -n 1 -p"
Step 1: Switch 'Software&Updates'/'Additional drivers' to 440 (prop) and apply changes.
Step 2: Download cuDNN v7.6.5 for CUDA 10.1/cuDNN Library for Linux from https://developer.nvidia.com/rdp/cudnn-download#_=_ .
" mainmenuinput
sudo apt install nvidia-cuda-toolkit
nvcc -V
whereis cuda
cd ~/Downloads
tar -xvzf ./cudnn-10.1-linux-x64-v7.6.5.32.tgz
sudo cp ./cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp ./cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
sudo rm -R ./cuda
sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
