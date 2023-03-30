#ifndef INNO_TRACKER_H
#define INNO_TRACKER_H

#include <stdint.h>

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "common.h"
#include "kalman_filter.hpp"

struct BayesVariables
{
  double p_d;
  double p_g;
  Eigen::Vector3d existence;
};

class iJIPDA_Tracker
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  iJIPDA_Tracker();

  iJIPDA_Tracker(int max_levels, double clutter_spatial_density, double delete_dur_sec);

  void set_existence_transform_matrix(const Eigen::Matrix3d &transf_mat);

  void update_tracks(const std::vector<Box> &measures);

  void get_tracks(std::vector<Box> &boxes);

  void add_new_track(const Box &box);

private:
  std::map<uint64_t, KalmanFilter *> trackers_;

  std::map<uint64_t, BayesVariables> bayes_variables_;

  std::map<uint64_t, std::chrono::system_clock::time_point> trackers_tp_;

  Eigen::Matrix3d exist_transf_matrix_;

  double clutter_spatial_density_;

  int max_levels_;

  uint64_t track_id_;

  double delete_dur_sec_;

  void update_state_(
      const std::vector<Box> &measures,
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, double &sum_weighted_like);

  void select_vaild_measures_(
      const std::vector<Box> &measures,
      std::multimap<uint64_t, size_t> &vaild_measures,
      std::multimap<size_t, uint64_t> &intersect_tracks);

  double calc_modulated_clutter_spatial_density_(
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, size_t measure_id, std::set<uint64_t> &track_set,
      std::set<size_t> &measure_set, int level);

  double calc_R_(
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, size_t measure_id, std::set<uint64_t> &track_set,
      std::set<size_t> &measure_set, int level);

  void update_likelihood_(
      const std::vector<Box> &measures,
      const std::multimap<uint64_t, size_t> &vaild_measures,
      std::map<std::pair<uint64_t, size_t>, double> &likelihood);

  void predict_existence_(BayesVariables &var);

  void update_existence_(BayesVariables &var, double sum_weighted_like);

  void delete_inactive_tracks_(const std::multimap<uint64_t, size_t> &vaild_measures);

private:
  virtual std::pair<KalmanFilter *, BayesVariables>
  create_new_track_(const Box &box);

  virtual Eigen::VectorXd get_measure_state_(const Box &box);

  virtual Box create_box_out_(const Eigen::VectorXd &state);

  virtual bool inner_gate_(const KalmanFilter *const klm, const Box &box);
};

#endif