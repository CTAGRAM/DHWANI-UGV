#!/bin/bash
# DHWANI OLED status. Started at power-on by the jetson user's crontab: @reboot /home/jetson/dhwani_oled/start.sh
# flock keeps it to one copy; the log only gets a line when the display (re)opens or I2C hiccups.
source /opt/ros/melodic/setup.bash
export ROS_MASTER_URI=http://localhost:11311 ROS_IP=127.0.0.1
exec flock -n /tmp/dhwani_oled.lock python3 /home/jetson/dhwani_oled/oled.py >> /home/jetson/dhwani_oled/oled.log 2>&1
