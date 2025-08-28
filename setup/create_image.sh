#!/bin/bash

# Setup script for custom Raspbian Lite image for Pi detectors

# If this is your first time running this script and it does not work, try to download these dependencies
#apt-get install coreutils quilt parted qemu-user-static debootstrap zerofree zip \
#dosfstools libarchive-tools libcap2-bin grep rsync xz-utils file git curl bc \
#gpg pigz xxd arch-test

# If it not your first time and it still doesn't work, try deleting the pi-gen directory first
# The below version clones the arm64 branch specfically because I have been building this on a Raspberry Pi 5
# You might need to change this depending on your build system
git clone --branch arm64 https://github.com/RPI-Distro/pi-gen.git
pushd pi-gen
chmod +x build.sh

# Start build process
SECONDS=0
clean=false

# NOTE: This config currently configures the detectors to have default user/pass and ssh open at first boot
# This is obviously a security risk on certain networks, be careful
cat > config <<EOL
export IMG_NAME=detector-raspbian
export RELEASE=bookworm
export DEPLOY_ZIP=1
export LOCALE_DEFAULT=en_US.UTF-8
export TARGET_HOSTNAME=detector-pi
export KEYBOARD_KEYMAP=us
export KEYBOARD_LAYOUT="English (US)"
export TIMEZONE_DEFAULT=America/Toronto
export TARGET_HOSTNAME=pi
export DISABLE_FIRST_BOOT_USER_RENAME=1
export FIRST_USER_NAME=pi
export FIRST_USER_PASS="password"
export ENABLE_SSH=1
EOL

# Skip stages 3-5, only want Raspbian lite
touch ./stage3/SKIP ./stage4/SKIP ./stage5/SKIP
touch ./stage4/SKIP_IMAGES ./stage5/SKIP_IMAGES

pushd stage2

# don't need NOOBS
rm -f EXPORT_NOOBS || true

# Add stage to the end of build process that performs the necessary detector setup
step="04-detector-setup"
if [ -d "$step" ]; then rm -Rf $step; fi
mkdir $step && pushd $step
cat > 00-run.sh <<RUN
#!/bin/bash
on_chroot << CHROOT
# Update package list and upgrade packages
apt-get update
apt-get upgrade -y

apt-get install -y python3-venv python3-pip build-essential libgtk-3-dev libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev python3-picamera2 ufw

# Download and set up the MCSTrack repository
cd /home/pi
wget --no-check-certificate -O MCSTrack.zip https://github.com/PerkLab/MCSTrack/archive/refs/heads/main.zip
unzip MCSTrack.zip
mv MCSTrack-main MCSTrack
chmod 777 MCSTrack

# Set up Python virtual environment and install dependencies
pushd MCSTrack
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip3 install --break-system-packages .[component]
popd

# Create startup script
cat > /usr/local/bin/startup << EOF
#!/bin/bash
sudo ufw allow 8001
cd /home/pi/MCSTrack
source .venv/bin/activate
python -m src.main_detector
EOF

chmod +x /usr/local/bin/startup

# Schedule the script to run at boot
echo '@reboot root /usr/local/bin/startup >> startup_log.log' > /etc/cron.d/startup
CHROOT
RUN

chmod +x 00-run.sh

popd
popd # stage 02

# run build
if [ "$clean" = true ] ; then
    echo "Running build with clean to rebuild last stage"
    CLEAN=1 ./build.sh
else
    echo "Running build"
    ./build.sh
fi

exitCode=$?

duration=$SECONDS
echo "Build process completed in $(($duration / 60)) minutes"

if [ $exitCode -ne 0 ]; then
    echo "Custom Raspbian lite build failed with exit code ${exitCode}" ; exit -1
fi

ls ./deploy
