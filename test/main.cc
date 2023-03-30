#include "../src/tracker_manager.hpp"
#include <iostream>
#include <fstream>
#include <thread>

using namespace inno_track;
int main()
{
  iJIPDA_Tracker bayes_tracker;

  std::vector<Box> boxes(6);
  boxes[0].x = 0, boxes[0].y = 0, boxes[0].speed_x = 10, boxes[0].speed_y = 10;
  boxes[1].x = 0, boxes[1].y = 200, boxes[1].speed_x = 10, boxes[1].speed_y = -10;
  boxes[2].x = 0, boxes[2].y = 100, boxes[2].speed_x = 10, boxes[2].speed_y = 10;
  boxes[3].x = 0, boxes[3].y = 20, boxes[3].speed_x = 10, boxes[3].speed_y = 10;
  boxes[4].x = 0, boxes[4].y = -50, boxes[4].speed_x = 10, boxes[4].speed_y = 20;
  boxes[5].x = 0, boxes[5].y = 300, boxes[5].speed_x = 10, boxes[5].speed_y = -30;

  bayes_tracker.add_new_track(boxes[0]);
  bayes_tracker.add_new_track(boxes[1]);
  bayes_tracker.add_new_track(boxes[2]);
  bayes_tracker.add_new_track(boxes[3]);
  bayes_tracker.add_new_track(boxes[4]);
  bayes_tracker.add_new_track(boxes[5]);

  std::ofstream ofile("../traj.txt", std::ios::trunc);
  for (int i = 1; i < 100; ++i)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    double x = i * 1.0;
    boxes[0].x = x, boxes[1].x = x, boxes[2].x = x, boxes[3].x = x, boxes[4].x = x, boxes[5].x = x;
    boxes[0].y = 2 * x;
    boxes[1].y = -2 * x + 200.0;
    boxes[2].y = 100.0;
    boxes[3].y = 1.6 * x + 20.0;
    boxes[4].y = 3 * x - 50.0;
    boxes[5].y = -4 * x + 300.0;
    bayes_tracker.update_tracks(boxes);
    std::vector<Box> box_out;
    bayes_tracker.get_tracks(box_out);
    for (auto &box : box_out)
    {
      ofile << box.id << "," << box.x << "," << box.y << std::endl;
      std::cout << box.id << "," << box.x << "," << box.y << "," << box.speed_x << "," << box.speed_y << std::endl;
    }
  }

  return 0;
}