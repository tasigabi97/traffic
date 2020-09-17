sudo docker run --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_REQUIRE_CUDA="cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"    -t -it --rm --ipc=host -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.12.0-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py
#    python /traffic/traffic/examples/tensorflow_1.py
#   -e az environment
#    echo $NVIDIA_REQUIRE_CUDA
#   -e NVIDIA_REQUIRE_CUDA="cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"
