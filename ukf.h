#ifndef UKF_H
#define UKF_H

#include <Eigen>
#include <functional>
#include <vector>

// Coding according to:
// 1. https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
// 2. Paper:The Unscented Kalman Filter for Nonlinear Estimation

class UKF
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // each col as a sigma points
  Eigen::MatrixXd sigma_points_;

  Eigen::MatrixXd sigma_points_pred_;

  Eigen::MatrixXd sigma_points_measure_;

  Eigen::VectorXd weights_m_;

  Eigen::VectorXd weights_cov_;

  // std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> predict_cb_;

  // std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> measure_trans_cb_;

  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> predict_cb_;

  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> measure_trans_cb_;

  Eigen::MatrixXd process_noise_cov_;

  Eigen::MatrixXd measure_noise_cov_;

  Eigen::VectorXd stat_pred_;

  Eigen::MatrixXd stat_cov_pred_;

  Eigen::VectorXd stat_refine_;

  Eigen::MatrixXd stat_cov_refine_;

  Eigen::MatrixXd gain_matrix_;

  int stat_num_;

  int measure_stat_num_;

  // according to the paper weight set
  double lambda_;

  double alpha_;

  double beta_;

  double kappa_;

public:
  UKF(int state_num, int measure_state_num);

  void set_ut_params(double alpha, double beta, double kappa);

  // void set_predict_callback(
  //     const std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> &predict_cb);

  // void set_measure_transform_callback(
  //     const std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> &measure_trans_cb);

  void set_predict_callback(
      const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &predict_cb);

  void set_measure_transform_callback(
      const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &measure_trans_cb);

  void set_process_noise_cov(const Eigen::MatrixXd &noise_cov);

  void set_measure_noise_cov(const Eigen::MatrixXd &noise_cov);

  void predict(const Eigen::MatrixXd &control_matrix, const Eigen::VectorXd &control_vector);

  void predict();

  void correct(const Eigen::VectorXd &measure_stat);

  void get_predict_state(Eigen::VectorXd &stat);

  void get_correct_state(Eigen::VectorXd &stat);

private:
  void generate_sigma_points_();

  void generate_wieghts_();

  void calc_lambda_();

  void predict_sigma_points_();

  void transform_sigma_points_to_measure_();
};

#endif // ukf.h