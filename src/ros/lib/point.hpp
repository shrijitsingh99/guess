#pragma once

//#include<iostream>
#include "def.h"
#include "labels.h"

namespace guess {

  struct Point {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Point(const Vector3& p_,          
          const Vector3& n_ = Vector3::Zero()) {
      p = p_;
      n = n_;
    }
    Vector3 p;
    Vector3 n;
  protected:
    Point();
  };

  struct LabPoint : public Point {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    LabPoint(const Label label_,
             const Vector3& p_,
             const Vector3& n_ = Vector3::Zero()) :
      Point(p_, n_) {
      label = label_;
    }
    Label label;
  protected:
    LabPoint();
  };

  using LabelCloud = std::vector<LabPoint, Eigen::aligned_allocator<LabPoint> >;
  
}
