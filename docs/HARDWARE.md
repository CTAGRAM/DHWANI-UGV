# Hardware

<p align="center"><img src="images/rover.jpg" width="520" alt="The DHWANI rover"></p>

## Bill of materials

| Component | Model | Role | Price (Rs) |
|---|---|---|---|
| Rover kit | Waveshare JetRacer Pro ROS AI kit | Jetson Nano 4 GB, Ackermann chassis, CSI camera, IMU, wheel encoders, OLED display, battery sensor | 26,399 |
| Lidar | RPLidar A1M8 | 360° scan from 0.15 to 12 m; the reference map for grading the camera | 6,777 |
| Stereo camera | Luxonis OAK-D-IoT-40 | stereo depth and on-chip neural inference | about 27,000 |
| Long-range radio | EBYTE E32 LoRa transmitter, 1 W | permission, position and alerts when Wi-Fi is gone | under 1,000 |
| **Total** | | | **about 61,000** |

Everything except the camera is sold in India. The camera is imported, and Indian listings for it
range from Rs 21,000 to 52,000. No GPU is needed, because depth and detection run on the camera's
own chip.

## How it is connected

| Part | Connection | Notes |
|---|---|---|
| OAK-D-IoT-40 | USB 3 to the Nano | mounted at the front bumper, about 8 cm above the floor and pitched about 8° down; it shows up as USB vendor `03e7` and needs a udev rule for that vendor before a normal user can open it |
| RPLidar A1M8 | USB | publishes `/scan` at about 8 Hz |
| EBYTE E32 (1 W) | Nano 40-pin header UART, `/dev/ttyTHS1`, pins 8 and 10, 9600 baud | M1 on header pin 7 (`gpio216`), held low for transparent mode; `tools/uart_loopback.py` finds the right two pins with one jumper wire and `tools/e32_probe.py` reads the module's settings |
| OLED, 0.91 in, SSD1306 128 x 32 | I2C bus 1, address `0x3C` | on the kit's expansion board |
| INA219 battery sensor | I2C bus 1, address `0x41` | on the kit's expansion board; gives the real pack voltage |
| Base station radio | a second E32 on a USB serial adapter (CP2102) at the laptop | used by `tools/lora_base.py`; a link between the two radios has not been tested yet |

## Power

The kit's 3-cell lithium-ion pack powers everything. Under motor load the voltage sags, and with a
low battery the rover has lost power several times with no warning in the system log. We now charge to about
12.6 V before autonomous runs. The voltage from the INA219 is smoothed over 30 seconds and turned
into a percentage with a typical lithium-ion curve from 3.4 V to 4.2 V per cell; that curve is not
calibrated on our cells yet. A regulated supply for the camera and the radio is on our roadmap.

## Things we learned the hard way

- The Nano has no clock battery. It wakes up at the time it was switched off and jumps forward when
  it finds a network, so nothing on the rover may trust wall-clock differences across a boot.
- Turning off Wi-Fi power saving on the rover cut the average round trip from 502 ms to 48 ms.
- The wheel odometry has a scale error of about 60 % until it is calibrated, so AMCL is set to trust
  the laser more than the wheels, and every distance we report comes from the laser map.
- Under DepthAI 2 every sensor of our OAK-D returned flat frames at the right rate and size. Under
  DepthAI 3 the same camera streamed real depth on the first try.
