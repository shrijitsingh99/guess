#! /bin/bash

GUESS_ROOT=/home/sapienzbot/ws/guess
rcfg=${GUESS_ROOT}/config/rviz/stage_guess.config.rviz
map_file=${GUESS_ROOT}/config/world/diag_floor_b1
# urdf_file=${GUESS_ROOT}/config/ros/urdf/

xterm -hold -e "roslaunch scan_guesser_node guess_stage_env.launch worldfile:=${map_file}.world mapfile:=${map_file}.yaml urdffile:=${urdffile} rviz_cfg:=${rcfg}" &
