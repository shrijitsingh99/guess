#pragma once

#include "def.h"
#include "labels.h"

namespace guess {

  struct LaserBeam {
    LaserBeam(const real& value_,
              const Label label_) {
      value = value_;
      label = label_;
    }
    real value;
    Label label;
  protected:
    LaserBeam();
  };

  struct LaserReading : public std::vector<LaserBeam> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double timestamp = 0.0;
    Isometry2 pose = Isometry2::Identity();
  };
  
}
