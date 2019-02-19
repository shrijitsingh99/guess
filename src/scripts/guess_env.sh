#! /bin/bash

GUESS_ROOT=/home/sapienzbot/ws/guess
rcfg=${GUESS_ROOT}/config/rviz/stage_guess.config.rviz

#diag_floor_b1
map_name=simple_corridor
map_file=${GUESS_ROOT}/config/world/${map_name}
# urdf_file=${GUESS_ROOT}/config/ros/urdf/

cmd_vel_topic=/cmd_vel

echo -e "Launching Guess-stage environment"

xterm -hold -e "roslaunch scan_guesser_node guess_stage_env.launch worldfile:=${map_file}.world mapfile:=${map_file}.yaml urdffile:=${urdffile} cmd_vel_topic:=${cmd_vel_topic} rviz_cfg:=${rcfg}" &
