#! /bin/bash


rcfg=/home/sapienzbot/ws/guess/config/stage_guess.config.rviz
map_file=/home/sapienzbot/ws/guess/src/ros/world/diag_floor_b1
# cmd_vel_topic=/teleop_velocity_smoother/raw_cmd_vel

xterm -hold -e "roslaunch scan_guesser_node guess_stage_env.launch worldfile:=${map_file}.world mapfile:=${map_file}.yaml rviz_cfg:=${rcfg}" &
