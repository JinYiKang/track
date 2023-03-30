#ifndef INNO_TRACKER_H
#define INNO_TRACKER_H

#include <stdint.h>

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

namespace inno_track
{

  struct Box
  {
    double x, y, z;
    double width, height, length;
    double speed_x, speed_y, speed_z, speed;
    double yaw;
    uint64_t id;
  };

  class KalmanFilter
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KalmanFilter()
    {
      state_dim_ = 4; // state:x, y, speed_x, speed_y

      measure_dim_ = 2; // measure state:x, y

      weight_sum_ = 0;

      state_pred_ = Eigen::VectorXd::Zero(state_dim_);

      cov_pred_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

      state_measure_pred_ = Eigen::VectorXd::Zero(measure_dim_);

      cov_measure_pred_ = Eigen::MatrixXd::Identity(measure_dim_, measure_dim_);

      state_corrected_ = Eigen::VectorXd::Zero(state_dim_);

      cov_corrected_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

      gain_ = Eigen::MatrixXd::Zero(state_dim_, measure_dim_);

      innovation_ = Eigen::VectorXd::Zero(measure_dim_);

      weighted_cov_measures_ = Eigen::MatrixXd::Zero(measure_dim_, measure_dim_);

      Q_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

      R_ = Eigen::MatrixXd::Identity(measure_dim_, measure_dim_);

      F_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);

      H_ = Eigen::MatrixXd::Zero(measure_dim_, state_dim_);

      F_ << 1.0, 0, 0.1, 0, 0, 1.0, 0, 0.1, 0, 0, 1.0, 0, 0, 0, 0, 1.0; // delta_t = 0.1s

      H_ << 1.0, 0, 0, 0, 0, 1.0, 0, 0;

      Q_ << 0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01;

      R_ << 0.25, 0, 0, 0.25;

      tp_ = std::chrono::system_clock::now();
    }

    virtual ~KalmanFilter(){}

    void add_weighted_measure(const Eigen::VectorXd &measure, double weight)
    {
      auto inno = measure - state_measure_pred_;
      innovation_ = innovation_ + weight * inno;
      weighted_cov_measures_ =
          weighted_cov_measures_ + weight * inno * inno.transpose();
      weight_sum_ += weight;
    }

    void bayes_update()
    {
      state_corrected_ = state_pred_ + gain_ * innovation_;
      cov_corrected_ =
          (1.0 - weight_sum_) * cov_pred_ + weight_sum_ * cov_corrected_ +
          gain_ *
              (weighted_cov_measures_ - innovation_ * innovation_.transpose()) *
              gain_.transpose();
      innovation_.fill(0.0);
      weighted_cov_measures_.fill(0.0);
      weight_sum_ = 0.0;
    }

    void set_state(const Eigen::VectorXd &state)
    {
      state_corrected_ = state;
      tp_ = std::chrono::system_clock::now();
    }

    const Eigen::VectorXd &get_state() const
    {
      return state_corrected_;
    }

    virtual void predict()
    {
      auto tp_now = std::chrono::system_clock::now();
      double delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_).count();
      F_(0, 2) = delta_t * 0.001;
      F_(1, 3) = delta_t * 0.001;
      tp_ = tp_now;

      state_pred_ = F_ * state_corrected_;
      cov_pred_ = F_ * cov_corrected_ * F_.transpose() + Q_;

      state_measure_pred_ = H_ * state_pred_;
      cov_measure_pred_ = H_ * cov_pred_ * H_.transpose() + R_;

      gain_ = cov_pred_ * H_.transpose() * cov_measure_pred_.inverse();

      cov_corrected_ = cov_pred_ - gain_ * H_ * cov_pred_;
    }

    Eigen::VectorXd get_state_measure_pred() const { return state_measure_pred_; }

    Eigen::MatrixXd get_cov_measure_pred() const { return cov_measure_pred_; }

  private:
    Eigen::VectorXd state_pred_;

    Eigen::MatrixXd cov_pred_;

    Eigen::VectorXd state_measure_pred_;

    Eigen::MatrixXd cov_measure_pred_;

    Eigen::VectorXd state_corrected_;

    Eigen::MatrixXd cov_corrected_;

    Eigen::MatrixXd gain_;

    Eigen::VectorXd innovation_;

    Eigen::MatrixXd weighted_cov_measures_;

    double weight_sum_;

    Eigen::MatrixXd Q_;

    Eigen::MatrixXd R_;

    Eigen::MatrixXd F_;

    Eigen::MatrixXd H_;

    int state_dim_;

    int measure_dim_;

    std::chrono::system_clock::time_point tp_;
  };

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

} // namespace inno_track

#endif