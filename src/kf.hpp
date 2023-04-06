#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H
#include <chrono>
#include <vector>

#include <Eigen/Dense>
namespace bayes_track {

class KalmanFilter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KalmanFilter() {
    state_dim_ = 4;  // state:x, y, speed_x, speed_y

    measure_dim_ = 2;  // measure state:x, y

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

    F_ << 1.0, 0, 0.1, 0, 0, 1.0, 0, 0.1, 0, 0, 1.0, 0, 0, 0, 0,
        1.0;  // delta_t = 0.1s

    H_ << 1.0, 0, 0, 0, 0, 1.0, 0, 0;

    Q_ << 0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01;

    R_ << 0.25, 0, 0, 0.25;

    tp_ = std::chrono::system_clock::now();
  }

  virtual ~KalmanFilter() {}

  void add_weighted_measure(const Eigen::VectorXd &measure, double weight) {
    auto inno = measure - state_measure_pred_;
    innovation_ = innovation_ + weight * inno;
    weighted_cov_measures_ =
        weighted_cov_measures_ + weight * inno * inno.transpose();
    weight_sum_ += weight;
  }

  void bayes_update() {
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

  void set_state(const Eigen::VectorXd &state) {
    state_corrected_ = state;
    tp_ = std::chrono::system_clock::now();
  }

  void set_Q(const Eigen::MatrixXd& Q){
    Q_ = Q;
  }

  void set_R(const Eigen::MatrixXd& R){
    R_ = R;
  }

  const Eigen::VectorXd &get_state() const { return state_corrected_; }

  virtual void predict() {
    auto tp_now = std::chrono::system_clock::now();
    double delta_t =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp_now - tp_)
            .count();
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
}  // namespace bayes_track
#endif