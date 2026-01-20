
To Do:
- Add the Mounting Frame screw and specifications to the material list
- Add instructions for Detector software MicroSD card flash
- Add the Mounting Frame stl file to the repository and update text
- Add details about Paper Printer selection (toner versus ink)
- Replace XXXXMaterialXXXX
- Replace XXXXToolXXXX
- Replace XXXXDetSoftLinkXXXX
- Replace XXXXMountFrameLinkXXXX
- Add images showing how to attach the Raspberry Pi 5 to the Mounting Frame

## List of Materials

(Let D be the number of detectors you wish you have)

- D x Raspberry Pi 5 - https://www.pishop.ca/product/raspberry-pi-5-8gb/
- D x Raspberry Pi Global Shutter Camera - https://www.pishop.ca/product/raspberry-pi-global-shutter-camera/
- D x Camera Lens - https://www.pishop.ca/product/6mm-wide-angle-lens-for-raspberry-pi-hq-camera-cs/
- D x Raspberry Pi "Drive" - https://www.pishop.ca/product/microsd-card-16-gb-class-10-blank/
- D x Raspberry Pi Camera Cable - https://www.pishop.ca/product/camera-cable-for-raspberry-pi-5 (200 mm, alternate link https://www.canakit.com/raspberry-pi-5-camera-cable.html)
- D x PoE hat for Raspberry Pi 5 - https://www.waveshare.com/poe-hat-f.htm
- D x Gooseneck clamps (choose SCP-BH, SCP-GN18HDB, SCP-TC) - https://snakeclamp.com/collections/camera
- D x Ethernet Cable - https://www.cdw.ca/product/tripp-lite-7ft-cat6-gigabit-molded-patch-cable-rj45-m-m-550mhz-24awg-yellow/5991672
- D x 8 x Mounting Screws - https://www.digikey.com/en/products/detail/fix-supply/0404MPP188/21635254
- 1 x 3D Printer
- 1 x Filament in XXXXMaterialXXXX
- 1 x XXXXToolXXXX
- 1 x Paper Printer
- White Printer Paper
- Full Page Printable Labels
- 1 x Switch with PoE support* - https://www.cdw.ca/product/netgear-gs516pp-ethernet-switch/6252835
- 1 x suitable Router*
- 1 x laptop or other computing device on which to run the pose solver
- 2 x Ethernet cables to connect router, switch, and laptop - https://www.cdw.ca/product/tripp-lite-cat6-gigabit-snagless-molded-patch-cable-rj45-m-m-blue-7ft/622270
- 1 x Raspberry Pi display cable - https://www.pishop.ca/product/micro-hdmi-to-hdmi-cable-for-pi-4-3ft-black/
- (Optional, but recommended) 1 x Raspberry Pi power supply - https://www.pishop.ca/product/raspberry-pi-27w-usb-c-power-supply-white-us/

* Alternatively, if you do not anticipate needing many detectors, then you may be able to use a router with sufficient PoE support (power budget and ethernet ports)

## SD Card Preparation

1. Gather necessary materials. You will need:
    - Workstation computer (computer other than the one you are preparing)
    - Mouse & Keyboard
    - MicroSD cards, Blank (min. 16 GB)
    - You may also need a MicroSD card adapter for your workstation (if your workstation does not have a MicroSD card slot)
1. On the workstation computer, download and install Raspberry Pi Imager ([link](https://www.raspberrypi.com/software/)).
1. Raspberry Pi OS will be helpful for previewing camera images and focusing lenses. The following steps describe how to flash Raspberry Pi OS to a MicroSD card.
    a. Connect a blank MicroSD card, using an adapter if necessary
    a. Run Raspberry Pi Imager
    a. When asked to select a Device, choose "Raspberry Pi 5" and click "Next"
    a. When asked to select an operating system, select "Raspberry Pi OS (64-bit)" and click "Next"
    a. When asked to select a storage device, choose the blank MicroSD card and click "Next"
    a. When asked to enter customization options, you may enter fields as you see fit
    a. When asked to confirm writing, review the information and proceed through the prompts
    a. When the software indicates that it is safe to eject the MicroSD card, remove it and close Raspberry Pi Imager.
1. The Detector software must be flashed to MicroSD cards, one for each Detector.
    a. Download the latest Detector software image from XXXXDetSoftLinkXXXX
    a. Follow the same procedure as above for flashing Raspberry Pi OS, except when asked to select an operating system, scroll down and select "custom image" and select the image file you just downloaded

## Mounting Frame Preparation

1. Download the Mounting Frame STL file from XXXXMountFrameLinkXXXX
1. Use the 3D printer to print the Mounting Frame
    - Print one Mounting Frame for each Detector
    - It is recommended to print one Mounting Frame at a time
    - It is recommended to print in XXXXMaterialXXXX due to its rigidity
1. Use XXXXToolXXXX to widen the existing holes so that they fit the screws

## Paper Tools

1. Print a full-page [Siemens Star](https://en.wikipedia.org/wiki/Siemens_star)

## Detector Setup

1. Gather necessary materials.
    - MicroSD Card flashed with Raspberry Pi OS
    - Mouse & Keyboard
    - Display & Suitable cable and/or adapters to connect to a Raspberry Pi 5
    - Raspberry Pi 5 Power Supply
    - Each Detector will additionally need:
        - Raspberry Pi 5
        - Raspberry Pi Global Shutter Camera
        - Raspberry Pi Camera Cable
        - Camera Lens
        - PoE HAT for Raspberry Pi 5
        - Mounting Frame
        - MicroSD card flashed with detector software
1. Install the PoE HAT on the Raspberry Pi 5 as per product instructions
1. Install the Camera Lens on the Raspberry Pi 5 Global Shutter Camera
1. Connect the Raspberry Pi 5 to the Mounting Frame
1. Connect the Raspberry Pi 5 Global Shutter Camera to the Mounting Frame
1. Connect the Raspberry Pi 5 to the Raspberry Pi 5 Global Shutter Camera using the Raspberry Pi Camera Cable
1. Focus the Camera Lens
    a. Place the assembled Detector such that the camera is directed toward a full-page Siemens Star, and is at a distance similar to that of markers in the intended use case
    a. Insert the Raspberry Pi OS MicroSD Card into the Raspberry Pi 5.
    a. Connect the Rasperry Pi 5 to a keyboard, mouse, display, and lastly power
    a. Wait for Raspberry Pi 5 to start up and for Raspberry Pi OS to become visible on the display
    a. Open a command line interface and run the command `rpicam-hello -t 1000000`. This is expected to show a live camera image, though it will likely be out of focus.
    a. Turn the shutter ring all the way open
    a. Turn the focus ring such that the star pattern is in maximal focus (lines and boundaries appear as sharp as possible)
    a. Power down the Raspberry Pi 5, and disconnect keyboard, mouse, display, and power.
1. Insert the Detector software image

## Router and PoE Setup

1. Gather materials. You will need:
    - PoE Switch
    - Router
    - Ethernet cables supporting PoE
1. Place the PoE Switch as desired, and connect to power with an appropriate power cable or AC adapter.
1. Place the router as desired, connect to the PoE switch with an Ethernet cable, and connect to power with an appropriate cable and AC adapter.
1. Use your workstation to configure the router
    a. Connect your workstation to the PoE Switch with an Ethernet cable.
    a. If your workstation is on any network, th
    a. Open an Internet browser on your workstation and navigate to your router's configuration page (consult your router's documentation)
    a. 

# Detector Setup

This system is intended to be used over a LAN connecting Raspberry Pi -based detector(s) and a main computer. The Raspberry Pi's run the ```src.main_detector```, while the main computer runs the ```src.main_pose_solver``` and optionally the ```src.gui.gui``` module. For a diagram, please see below.

![diagram_1](diagram.png)

To setup the Raspberry Pi -based detectors, 
1. Download the latest OS image (.zip file) under Releases on the main web page of this repository. 
   - Alternatively, run the setup/create_image.sh script on a compatible linux system, which will generate the OS image from scratch
2. Flash the resulting image to a microSD card (or multiple) for use.
   - Plug microSD card into a compatible port, or adapter, on the computer where you downloaded the OS image
   - Use a flashing software, such as (Etcher)[https://etcher.balena.io/], to flash the OS image downloaded in step (1) to the microSD card
   - Once flashing is complete, remove the microSD card from the computer, and plug it into the Raspberry Pi.
   - Power up the Pi by ethernet.
   - To confirm the Pi is working, run a network scan (e.g., (nmap)[https://nmap.org/]) before and after connecting the Pi to the network. Consult the IP address assigned in `Router Setup` to ensure the Pi is connected. 

# Software Environment Setup

This set up is necessary for running the pose solver and GUI on the host machine. After cloning this repository, please see the section that pertains to your operating system below.

### Windows

Please install a recent version of Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

You may need to manually modify the installation and ensure the C++ workload is selected.

```
py -3.11 -m venv .venv
cd venv/Scripts
activate
cd ../..
pip install .[gui,component]
```

### Linux

Install python3.11. If you are on a Debian-based distribution and you cannot find that version of Python, you can try the deadsnakes ppa: https://askubuntu.com/questions/1398568/installing-python-who-is-deadsnakes-and-why-should-i-trust-them

You may need to install additional packages depending on your distribution. The following list is a work in progress and it will become more complete over time:
- libgtk-3-dev
- python3.11
- python3.11-dev

```
py -3.11 -m venv .venv
source venv/bin/activate
pip install .[gui,component]
```

### Troubleshooting

wxPython wheel failure, try: https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html


# GUI Usage

On the host machine, with the python virtualenv activated (see platform-specific instructions above), run the ```src.main_pose_solver``` and ```src.gui.gui``` modules - either through command line, or your IDE of choice for debugging. 

Using the GUI, add the IP address and port (8001) of the detector. For the pose solver, the IP address will be localhost and port 8000. To use the pose solver renderer (Windows only), the detector must first be calibrated using a board of markers - as described by https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html. In the detector panel, with the 'preview image' option checked, capture calibration images of the board at a variety of angles. In the calibrator panel, you can select which ones to use to generate a calibration. Once that is complete, you can set the reference and tracked markers in the pose solver panel, and begin tracking. 
