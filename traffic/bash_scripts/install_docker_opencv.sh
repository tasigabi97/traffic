apt update
read  -n 1 -p"apt install libopencv-*"
apt install libopencv-*
read  -n 1 -p"apt install libgtk2.0-dev"
apt install libgtk2.0-dev
pip reinstall opencv-python







python /traffic/traffic/examples/original_camera.py

read  -n 1 -p"END"

#export QT_DEBUG_PLUGINS=1
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.5/dist-packages/cv2/qt/plugins
echo $QT_QPA_PLATFORM_PLUGIN_PATH
read  -n 1 -p"apt install libxcb-xinerama0"
apt install libxcb-xinerama0
read  -n 1 -p"apt install libgl1-mesa-glx"
apt install libgl1-mesa-glx
read  -n 1 -p"pip install --upgrade pip"
pip install --upgrade pip
pip install opencv-python
python /traffic/traffic/examples/original_camera.py
bash
