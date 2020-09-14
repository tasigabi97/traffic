read  -n 1 -p "Step 1: Switch 'Additional drivers' in Software & Updates to 440 (prop) and apply changes." mainmenuinput
cd /home/gabi/WINDOWS/STATIC
sudo apt install nvidia-cuda-toolkit
nvcc -V
tar -xvzf /home/gabi/WINDOWS/STATIC/cudnn-10.1-linux-x64-v7.6.5.32.tgz
sudo cp /home/gabi/WINDOWS/STATIC/cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp /home/gabi/WINDOWS/STATIC/cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
rm -R /home/gabi/WINDOWS/STATIC/cuda