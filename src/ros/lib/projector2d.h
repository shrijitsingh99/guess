#pragma once

#include "laser_reading.hpp"
#include "point.hpp"

namespace guess {

  class Projector2d {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct Config {
      real min_range = 30;
      real max_range = 0.1;
      size_t num_ranges = 1024;
      real fov = 1.5 * M_PI;
    };

    Projector2d() {}
    ~Projector2d() {}

    const Config& config() const {return _config;}
    Config& mutableConfig() {return _config;}
    
    // TODO
    void project(LaserReading& laser_reading_,
                 const Isometry2& T_,
                 const LabelCloud& cloud_);

    // TODO
    void unproject(LabelCloud& cloud_,
                   const LaserReading& laser_reading_);

  private:
    Config _config;
    
  };

}
