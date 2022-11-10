#include "ukf.h"
#include <assert.h>
using namespace Eigen;

UKF::UKF(int state_num, int measure_state_num)
    : stat_num_(state_num), measure_stat_num_(measure_state_num),
      alpha_(0.001), beta_(2), kappa_(0)
{
  assert(state_num > 0 && measure_state_num > 0);
  stat_pred_ = MatrixXd::Zero(stat_num_, 1);
  stat_refine_ = MatrixXd::Zero(stat_num_, 1);
  stat_cov_pred_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  stat_cov_refine_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  process_noise_cov_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  measure_noise_cov_ = MatrixXd::Identity(measure_stat_num_, measure_stat_num_) * 0.001;
  gain_matrix_ = MatrixXd(stat_num_, measure_stat_num_);
  sigma_points_ = MatrixXd(stat_num_, 2 * stat_num_ + 1);
  sigma_points_pred_ = MatrixXd(stat_num_, 2 * stat_num_ + 1);
  sigma_points_measure_ = MatrixXd(measure_stat_num_, 2 * stat_num_ + 1);
  weights_m_ = VectorXd(2 * stat_num_ + 1);
  weights_cov_ = VectorXd(2 * stat_num_ + 1);
  calc_lambda_();
  generate_wieghts_();
}

void UKF::calc_lambda_()
{
  lambda_ = alpha_ * alpha_ * (stat_num_ + kappa_) - stat_num_;
}

void UKF::set_ut_params(double alpha, double beta, double kappa)
{
  alpha_ = alpha;
  beta_ = beta;
  kappa_ = kappa;
  calc_lambda_();
  generate_wieghts_();
}

// void UKF::set_predict_callback(
//     const std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> &predict_cb)
// {
//   predict_cb_ = predict_cb;
// }

// void UKF::set_measure_transform_callback(
//     const std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> &measure_trans_cb)
// {
//   measure_trans_cb_ = measure_trans_cb;
// }

void UKF::set_predict_callback(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &predict_cb)
{
  predict_cb_ = predict_cb;
}

void UKF::set_measure_transform_callback(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &measure_trans_cb)
{
  measure_trans_cb_ = measure_trans_cb;
}

void UKF::set_process_noise_cov(const Eigen::MatrixXd &noise_cov)
{
  process_noise_cov_ = noise_cov;
}

void UKF::set_measure_noise_cov(const Eigen::MatrixXd &noise_cov)
{
  measure_noise_cov_ = noise_cov;
}

void UKF::generate_sigma_points_()
{
  auto temp_matrix = (stat_num_ + lambda_) * stat_cov_refine_;
  MatrixXd cov_square_root = temp_matrix.llt().matrixL();

  sigma_points_.col(0) = stat_refine_;
  for (int i = 0; i < stat_num_; ++i)
  {
    sigma_points_.col(i + 1) = stat_refine_ + cov_square_root.col(i);
    sigma_points_.col(i + stat_num_ + 1) = stat_refine_ - cov_square_root.col(i);
  }
}

void UKF::generate_wieghts_()
{
  double weight_temp = 1.0 / (2.0 * (stat_num_ + lambda_));
  weights_m_ = VectorXd::Constant(2 * stat_num_ + 1, weight_temp);
  weights_cov_ = VectorXd::Constant(2 * stat_num_ + 1, weight_temp);
  weights_m_(0) = lambda_ / (stat_num_ + lambda_);
  weights_cov_(0) = weights_m_(0) + 1.0 - alpha_ * alpha_ + beta_;
}

void UKF::predict_sigma_points_()
{
  for (int i = 0; i < 2 * stat_num_ + 1; ++i)
  {
    sigma_points_pred_.col(i) = predict_cb_(sigma_points_.col(i));
  }
}

void UKF::transform_sigma_points_to_measure_()
{
  for (int i = 0; i < 2 * stat_num_ + 1; ++i)
  {
    sigma_points_measure_.col(i) = measure_trans_cb_(sigma_points_pred_.col(i));
  }
}

void UKF::predict(const Eigen::MatrixXd &control_matrix, const Eigen::VectorXd &control_vector)
{
  generate_sigma_points_();
  // predict_cb_(sigma_points_, sigma_points_pred_);
  // measure_trans_cb_(sigma_points_pred_, sigma_points_measure_);
  predict_sigma_points_();
  transform_sigma_points_to_measure_();

  assert(sigma_points_pred_.rows() == stat_num_ &&
         sigma_points_pred_.cols() == 2 * stat_num_ + 1);
  assert(sigma_points_measure_.rows() == measure_stat_num_ &&
         sigma_points_measure_.cols() == 2 * stat_num_ + 1);

  stat_pred_ = sigma_points_pred_ * weights_m_ + control_matrix * control_vector;
  sigma_points_pred_.colwise() -= stat_pred_;
  stat_cov_pred_ =
      sigma_points_pred_ * weights_cov_.asDiagonal() * sigma_points_pred_.transpose() +
      process_noise_cov_;

  stat_refine_ = stat_pred_;
  stat_cov_refine_ = stat_cov_pred_;
}

void UKF::predict()
{
  predict(MatrixXd::Zero(stat_num_, stat_num_), VectorXd::Zero(stat_num_));
}

void UKF::correct(const Eigen::VectorXd &measure_stat)
{
  VectorXd m_trans = sigma_points_measure_ * weights_m_;
  sigma_points_measure_.colwise() -= m_trans;
  MatrixXd cov_trans =
      sigma_points_measure_ * weights_cov_.asDiagonal() * sigma_points_measure_.transpose() +
      measure_noise_cov_;
  MatrixXd cross_corr =
      sigma_points_pred_ * weights_cov_.asDiagonal() * sigma_points_measure_.transpose();
  // kalman gain
  gain_matrix_ = cross_corr * cov_trans.inverse();
  stat_refine_ = stat_pred_ + gain_matrix_ * (measure_stat - m_trans);
  stat_cov_refine_ = stat_cov_pred_ - gain_matrix_ * cov_trans * gain_matrix_.transpose();
}

void UKF::get_predict_state(Eigen::VectorXd &stat)
{
  stat = stat_pred_;
}

void UKF::get_correct_state(Eigen::VectorXd &stat)
{
  stat = stat_refine_;
}