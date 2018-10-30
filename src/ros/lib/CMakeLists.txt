cmake_minimum_required(VERSION 2.8.3)
project(guess)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11")
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(guess_lib SHARED
  laser_reading.hpp
  point.hpp
  projector2d.h projector2d.cpp
  )

#target_link_libraries()
