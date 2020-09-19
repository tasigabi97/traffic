echo GUI+TF
xhost +local:
sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -it --privileged --net=host --ipc=host --gpus all -v /home/gabi/PycharmProjects/traffic/:/traffic jjanzic/docker-python3-opencv  bash /traffic/traffic/bash_scripts/install_docker_opencv.sh

#sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -it --privileged --net=host --ipc=host --gpus all -v /home/gabi/PycharmProjects/traffic/:/traffic tensorflow/tensorflow:1.3.0-gpu-py3  bash /traffic/traffic/bash_scripts/install_docker_opencv.sh

read  -n 1 -p"just GUI"

xhost +local:
sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -it --privileged --net=host --ipc=host --gpus all -v /home/gabi/PycharmProjects/traffic/:/traffic opencvcourses/opencv:440  python /traffic/traffic/examples/original_camera.py

read  -n 1 -p"just Tensorflow"

echo 1.3.0 working
sudo docker run -it --privileged --net=host --ipc=host --gpus all -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.3.0-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py

echo 1.12.0 working
sudo docker run -it --privileged --net=host --ipc=host --gpus all -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.12.0-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py

# hosszú
#sudo docker run -d=false -it --privileged --net=host --ipc=host --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_REQUIRE_CUDA="cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"   -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.15.2-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py


# --rm=true # kilépéskor törli a fájlokat, de nem működik eddig
# --privileged #  bekapcsol minden eszközt
# -it # -t -i elvileg
# --gpus all # --runtime=nvidia # videókártya támogatás, elvileg mindkettő ugyanaz de az első újabb
# -e # környezeti változó beállítás
# -v "/home/gabi/PycharmProjects/traffic/:/traffic" # eléri a mappát a konténer
# --ipc=host # nem tűnik fontosnak eddig
# -d=false # előtérben fut