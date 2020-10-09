# This file must be used with "source bin/activate" *from bash* you cannot run it directly
cd /home/gabi/PycharmProjects/traffic/
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.5
sudo apt install -y python3-virtualenv
sudo apt install -y python3-tk
sudo apt-get install -y libfreetype6-dev
sudo apt-get install -y python3.5-dev
sudo apt install -y libcurl4-openssl-dev libssl-dev
sudo apt-get install -y python3-distutils-extra
virtualenv -p /usr/bin/python3.5 /home/gabi/PycharmProjects/traffic/venv3.5
sudo chmod 777 /home/gabi/PycharmProjects/traffic/venv3.5/bin/activate
echo export PYTHONPATH="/home/gabi/PycharmProjects/traffic:/home/gabi/PycharmProjects/traffic/mrcnn:/home/gabi/PycharmProjects/traffic/mrcnn/samples/coco" >> /home/gabi/PycharmProjects/traffic/venv3.5/bin/activate
. /home/gabi/PycharmProjects/traffic/venv3.5/bin/activate
echo INSTALL PYPI PACKAGES
echo install --upgrade pip @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install --upgrade pip
echo install opencv-python @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install opencv-python
echo install cython @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install cython
echo install scikit-image @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install scikit-image
echo install imgaug @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install imgaug
echo install IPython[all] @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install IPython[all]
echo install keras==2.0.8 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install keras==2.0.8
echo install tensorflow-gpu==1.12.0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install tensorflow-gpu==1.12.0
echo INSTALL PYCOCOTOOLS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
cd /home/gabi/PycharmProjects/traffic/pycocotools/PythonAPI && python setup.py build_ext install
cp /home/gabi/PycharmProjects/traffic/jdk.table.xml /home/gabi/.config/JetBrains/PyCharmCE2020.2/options/jdk.table.xml
