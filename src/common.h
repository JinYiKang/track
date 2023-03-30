#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

struct Box
{
    double x, y, z;
    double width, height, length;
    double speed_x, speed_y, speed_z, speed;
    double yaw;
    uint64_t id;
};

#endif