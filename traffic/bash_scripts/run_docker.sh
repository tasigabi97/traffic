# hosszú
#sudo docker run -d=false -it --privileged --net=host --ipc=host --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_REQUIRE_CUDA="cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"   -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.12.0-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py
echo 1.12.0 working
sudo docker run -it --privileged --net=host --ipc=host --gpus all -v "/home/gabi/PycharmProjects/traffic/:/traffic" tensorflow/tensorflow:1.12.0-gpu-py3 python /traffic/traffic/examples/tensorflow_1.py



# --rm=true # kilépéskor törli a fájlokat, de nem működik eddig
# --privileged #  bekapcsol minden eszközt
# -it # -t -i elvileg
# --gpus all # --runtime=nvidia # videókártya támogatás, elvileg mindkettő ugyanaz de az első újabb
# -e # környezeti változó beállítás
# -v "/home/gabi/PycharmProjects/traffic/:/traffic" # eléri a mappát a konténer
# --ipc=host # nem tűnik fontosnak eddig
# -d=false # előtérben fut


