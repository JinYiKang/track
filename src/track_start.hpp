#ifndef TRACK_START_H
#define TRACK_START_H
#include <chrono>
#include <map>
#include <vector>

#include "common.h"

namespace bayes_track {

struct PotenTraj {
  Box box;
  std::chrono::system_clock::time_point tp;
  int history_len;
  bool started;
  PotenTraj() : history_len(0), started(false) {}
};

class StartTrack {
 public:
  StartTrack()
      : min_speed_x_(-66.0),
        max_speed_x_(33.0),
        min_speed_y_(-33.0),
        max_speed_y_(33.0),
        speed_bias_(2.75),
        start_len_(3),
        preserve_time_(0.1) {}

  void set_start_len(int start_len) { start_len_ = start_len; }

  void feed_boxes(const std::vector<Box>& boxes);

  void get_starters(std::vector<Box>& starters) {
    for (auto& traj : branches_) {
      if (traj.started) {
        continue;
      }

      bool new_starter = false;
      if (traj.box.x < 80.0) {
        new_starter = traj.history_len >= start_len_ - 1;
      } else {
        new_starter = traj.history_len >= 2 * start_len_ - 1;
      }

      if (new_starter) {
        starters.push_back(traj.box);
        traj.started = true;
      }
    }
  }

  size_t get_banches_size() const { return branches_.size(); }

 private:
  double min_speed_x_, max_speed_x_;
  double min_speed_y_, max_speed_y_;
  double speed_bias_;
  int start_len_;
  double preserve_time_;
  std::vector<PotenTraj> branches_;
};

void StartTrack::feed_boxes(const std::vector<Box>& boxes) {
  if (boxes.empty()) {
    return;
  }

  auto tp_now = std::chrono::system_clock::now();
  std::vector<PotenTraj> new_branches;
  std::vector<Box> branches_pred;

  // prediction
  for (const PotenTraj& traj : branches_) {
    double delta_t =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - traj.tp)
            .count() *
        0.001;  // seconds
    Box box = traj.box;
    if (traj.history_len > 0) {
      box.x = box.x + box.speed_x * delta_t;
      box.y = box.y + box.speed_y * delta_t;
    }

    if (traj.history_len > 1) {
      double delta_t_2 = delta_t * delta_t;
      box.x = box.x + 0.5 * delta_t_2 * box.acc_x;
      box.y = box.y + 0.5 * delta_t_2 * box.acc_y;
    }
    branches_pred.push_back(box);
  }

  // generate the nearest branches
  std::vector<int> new_starters_flag(boxes.size(), -1);

  for (size_t i = 0; i < branches_.size(); ++i) {
    const PotenTraj& traj = branches_[i];
    const Box& box_pred = branches_pred[i];
    if (traj.started) {
      continue;
    }

    double delta_t =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - traj.tp)
            .count() *
        0.001;  // seconds

    double min_dist = 100000.0;
    bool update_branch = false;

    PotenTraj new_traj = traj;

    for (size_t j = 0; j < boxes.size(); ++j) {
      const Box& box = boxes[j];
      double x = box.x;
      double y = box.y;
      double z = box.z;

      double vx = (x - traj.box.x) / delta_t;
      double vy = (y - traj.box.y) / delta_t;

      if (traj.history_len == 0) {
        if (vx >= min_speed_x_ && vx < max_speed_x_ && vy >= min_speed_y_ &&
            vy < max_speed_y_) {
          double dist = std::sqrt((x - traj.box.x) * (x - traj.box.x) +
                                  (y - traj.box.y) * (y - traj.box.y));
          if (dist < min_dist) {
            min_dist = dist;
            new_traj.box.x = x, new_traj.box.y = y, new_traj.box.z = z;
            new_traj.box.speed_x = vx, new_traj.box.speed_y = vy;
            new_traj.box.width = std::max(box.width, new_traj.box.width);
            new_traj.box.height = std::max(box.height, new_traj.box.height);
            new_traj.box.length = std::max(box.length, new_traj.box.length);
          }
          update_branch = true;
          new_starters_flag[j] = 1;
        }
      } else {
        double speed_bias = speed_bias_;
        if (x > 80.0) {
          speed_bias *= 2.0;
        }
        double dist_bias = delta_t * speed_bias;
        if (std::abs(x - box_pred.x) < dist_bias &&
            std::abs(y - box_pred.y) < dist_bias) {
          double dist = std::sqrt((x - traj.box.x) * (x - traj.box.x) +
                                  (y - traj.box.y) * (y - traj.box.y));
          if (dist < min_dist) {
            min_dist = dist;
            new_traj.box.x = x, new_traj.box.y = y, new_traj.box.z = z;
            new_traj.box.acc_x = (vx - traj.box.speed_x) / delta_t;
            new_traj.box.acc_y = (vy - traj.box.speed_y) / delta_t;
            new_traj.box.speed_x = vx, new_traj.box.speed_y = vy;
            new_traj.box.width = std::max(box.width, new_traj.box.width);
            new_traj.box.height = std::max(box.height, new_traj.box.height);
            new_traj.box.length = std::max(box.length, new_traj.box.length);
          }
          update_branch = true;
          new_starters_flag[j] = 1;
        }
      }
    }

    if (update_branch) {
      new_traj.history_len++;
      new_traj.tp = tp_now;
      new_branches.push_back(new_traj);
    } else if (delta_t < preserve_time_) {
      new_branches.push_back(new_traj);
    }
  }

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (new_starters_flag[i] < 0) {
      const Box& box = boxes[i];
      PotenTraj new_traj;
      new_traj.tp = tp_now;
      new_traj.box.x = box.x;
      new_traj.box.y = box.y;
      new_traj.box.z = box.z;
      new_traj.box.width = box.width;
      new_traj.box.height = box.height;
      new_traj.box.length = box.length;
      new_branches.push_back(new_traj);
    }
  }

  std::swap(new_branches, branches_);
}

}  // namespace bayes_track

#endif