#!/bin/bash
sudo apt update
sudo apt upgrade -y

# Environment variables
export MCSTRACK_DETECTOR_CONFIGURATION_FILEPATH="/home/admin/MCSTrack/data/configuration/detector/rpicam_aruco.json"

# Firewall setup
sudo ufw enable
sudo ufw allow 8001

# Python setup
cd /home/admin/MCSTrack
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --break-system-packages .[component]
deactivate

# Run startup script on boot
sudo echo "@reboot root /usr/local/bin/mcstrack_startup >> mcstrack_log.log" > /etc/cron.d/startup
