apt update
# -y # yes mindenre
apt install -y wireless-tools
apt install -y python-opencv
pip install --upgrade pip
pip install opencv-python
pip install cython
pip install scikit-image
pip install imgaug
pip install IPython[all]
echo "alias python=python3" >> /etc/bash.bashrc
echo export PYTHONPATH="/traffic" >> /etc/bash.bashrc
