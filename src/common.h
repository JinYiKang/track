#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#include <Eigen/Dense>

#include "cyber/common/log.h"

namespace bayes_track {

struct Point2D {
  double x;
  double y;

  Point2D() : x(0), y(0) {}
  Point2D(double p_x, double p_y) : x(p_x), y(p_y) {}
  Point2D operator+(const Point2D& pt) { return Point2D(x + pt.x, y + pt.y); }
  Point2D operator-(const Point2D& pt) { return Point2D(x - pt.x, y - pt.y); }
  void operator+=(const Point2D& pt) { x += pt.x, y += pt.y; }
  void operator-=(const Point2D& pt) { x -= pt.x, y -= pt.y; }
};

struct Box {
  double x, y, z;
  double width, height, length;
  double speed_x, speed_y, speed_z, speed;
  double yaw;
  uint64_t id;
  double acc_x, acc_y, acc_z;

  Box()
      : x(0),
        y(0),
        z(0),
        width(0),
        height(0),
        length(0),
        speed_x(0),
        speed_y(0),
        speed_z(0),
        speed(0),
        yaw(0),
        acc_x(0),
        acc_y(0),
        acc_z(0) {}

  /**rotate around the z axis*/
  static Point2D rotate(double x, double y, double yaw) {
    return Point2D(x * std::cos(yaw) - y * std::sin(yaw),
                   y * std::cos(yaw) + x * std::sin(yaw));
  }

  /**x-y plane vertex points*/
  Point2D tl() const{
    return Point2D(x, y) + this->rotate(0.5 * length, -0.5 * width, yaw);
  }

  Point2D tr() const{
    return Point2D(x, y) + this->rotate(0.5 * length, 0.5 * width, yaw);
  }

  Point2D bl() const{
    return Point2D(x, y) + this->rotate(-0.5 * length, -0.5 * width, yaw);
  }

  Point2D br() const{
    return Point2D(x, y) + this->rotate(-0.5 * length, 0.5 * width, yaw);
  }
};

}  // namespace bayes_track

#endif