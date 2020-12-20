#Ez a script letölti és feltelepíti a droidcam alkalmazást. (nem a konténeren belül)
#Csak azért kell ez a script, mert nem sikerült rájönnöm,
# hogyan lehetne feltelepíteni a kamera drivert a konténeren belül.
sudo apt-get install -y make linux-headers-`uname -r`gcc
cd /tmp/
wget https://files.dev47apps.net/linux/droidcam_latest.zip
echo "952d57a48f991921fc424ca29c8e3d09 droidcam_latest.zip" | md5sum -c --
unzip droidcam_latest.zip -d droidcam && cd droidcam
sudo ./install-client
sudo ./install-video
lsmod | grep v4l2loopback_dc
