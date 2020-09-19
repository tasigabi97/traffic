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
# beállítja az ezt beolvasó bash aliasát
shopt -s expand_aliases
alias python=python3
echo Running tests:
# csak ebben a bashben van az alias ezért kell a source
. /traffic/traffic/bash_scripts/programs.sh
echo Bash:
bash