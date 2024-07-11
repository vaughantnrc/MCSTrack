# Environment Setup

This set up is necessary for running the pose solver and GUI. Please see the section that pertains to your operating system below.

### Windows

Please install a recent version of Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

You may need to manually modify the installation and ensure the C++ workload is selected.

```
py -3.11 -m venv venv
cd venv/Scripts
activate
cd ../..
pip install -r requirements.txt
```

### Linux

Install python3.11. If you are on a Debian-based distribution and you cannot find that version of Python, you can try the deadsnakes ppa: https://askubuntu.com/questions/1398568/installing-python-who-is-deadsnakes-and-why-should-i-trust-them

You may need to install additional packages depending on your distribution. The following list is a work in progress and it will become more complete over time:
- libgtk-3-dev
- python3.11
- python3.11-dev

```
py -3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Troubleshooting

wxPython wheel failure, try: https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html


# Detector Setup

This system is intended to be used over a LAN connecting Raspberry Pi -based detector(s) and a main computer. The Raspberry Pi's run the ```src.main_detector```, while the main computer runs the ```src.main_pose_solver``` and optionally the ```src.gui.gui``` module.

To setup the Raspberry Pi -based detectors, download the latest OS image under Releases. Alternatively, run the setup/create_image.sh script on a compatible linux system. Flash the resulting image to a microSD card (or multiple) for use.

Connect the Raspberry Pi and host machine to the LAN. On the host machine, with the python virtualenv activated (see platform-specific instructions above), run the ```src.main_pose_solver``` and ```src.gui.gui``` modules - either through command line, or your IDE of choice for debugging. 

# GUI Usage

Using the GUI, add the IP address and port (8001) of the detector. For the pose solver, the IP address will be localhost and port 8000. To use the pose solver renderer (Windows only), the detector must first be calibrated using a board of markers - as described by https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html. In the detector panel, with the 'preview image' option checked, capture calibration images of the board at a variety of angles. In the calibrator panel, you can select which ones to use to generate a calibration. Once that is complete, you can set the reference and tracked markers in the pose solver panel, and begin tracking. 