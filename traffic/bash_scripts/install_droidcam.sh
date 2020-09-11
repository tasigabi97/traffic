sudo apt-get install make
sudo apt-get install linux-headers-`uname -r`
sudo apt-get install gcc
cd /tmp/
wget https://files.dev47apps.net/linux/droidcam_latest.zip
echo "73db3a4c0f52a285b6ac1f8c43d5b4c7 droidcam_latest.zip" | md5sum -c --
unzip droidcam_latest.zip -d droidcam && cd droidcam
sudo ./install
lsmod | grep v4l2loopback_dc