echo INSTALL PROGRAMS
echo echo apt update @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
apt update
echo install -y wireless-tools @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
apt install -y wireless-tools
echo install -y python-opencv @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
apt install -y python-opencv
echo install -y python3-tk @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
apt install -y python3-tk
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
echo install pytest @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip install pytest
echo INSTALL PYCOCOTOOLS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
cd /traffic/pycocotools/PythonAPI && python3 setup.py build_ext install
echo SAVE PACKAGES @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
pip freeze > "/traffic/requirements.$(pip show tensorflow-gpu | grep  Version | grep -o 1.*).txt"
chmod 777 "/traffic/requirements.$(pip show tensorflow-gpu | grep  Version | grep -o 1.*).txt"
cat "/traffic/requirements.$(pip show tensorflow-gpu | grep  Version | grep -o 1.*).txt"
echo SET BASH DEFAULTS
echo "alias python=python3" >> /etc/bash.bashrc
echo export PYTHONPATH="/traffic:/traffic/mrcnn" >> /etc/bash.bashrc
echo alias @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
bash -i -c "alias"
echo env @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
bash -i -c "env"

#ipython /traffic/mrcnn/samples/demo.py

echo CONVERT:
#   ipython nbconvert --to script /traffic/mrcnn/samples/demo.ipynb && chmod 777  /traffic/mrcnn/samples/demo.py

echo OPEN JUPYTERNOTEBOOK:
#   cd /traffic && jupyter notebook --allow-root
echo COPY WITH TOKEN


#cd traffic/mrcnn
#python /traffic/mrcnn/setup.py install

