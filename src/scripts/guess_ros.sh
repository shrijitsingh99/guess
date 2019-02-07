#! /bin/bash

GUESS_ROOT=/home/sapienzbot/ws/guess
pyscript=${GUESS_ROOT}/src/py/scan_guesser_socket.py
cmd_vel_topic=/teleop_velocity_smoother/raw_cmd_vel

xterm -hold -e "rosrun scan_guesser_node guesser _cmd_vel_topic:=${cmd_vel_topic}" &

export PYTHONPATH="/usr/local/:${GUESS_ROOT}/src/py/:/usr/lib/python3/dist-packages/"
python3 ${pyscript}
