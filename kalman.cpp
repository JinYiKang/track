#include "kalman.h"
#include <assert.h>

using namespace Eigen;

Kalman::Kalman(int state_num, int measure_state_num)
    : stat_num_(state_num), measure_stat_num_(measure_state_num)
{
  assert(state_num > 0 && measure_state_num > 0);
  stat_pred_ = MatrixXd::Zero(stat_num_, 1);
  stat_refine_ = MatrixXd::Zero(stat_num_, 1);
  stat_cov_pred_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  stat_cov_refine_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  predict_matrix_ = MatrixXd(stat_num_, stat_num_);
  external_noise_cov_ = MatrixXd::Identity(stat_num_, stat_num_) * 0.001;
  measure_noise_cov_ = MatrixXd::Identity(measure_stat_num_, measure_stat_num_) * 0.001;
  measure_trans_ = MatrixXd(measure_stat_num_, stat_num_);
  gain_matrix_ = MatrixXd(stat_num_, measure_stat_num_);
}

void Kalman::set_predict_matrix(const Eigen::MatrixXd &predict_matrix)
{
  predict_matrix_ = predict_matrix;
}

void Kalman::set_external_noise_cov(const Eigen::MatrixXd &noise_cov)
{
  external_noise_cov_ = noise_cov;
}

void Kalman::set_measure_noise_cov(const Eigen::MatrixXd &noise_cov)
{
  measure_noise_cov_ = noise_cov;
}

void Kalman::set_measure_transform_matrix(const Eigen::MatrixXd &measure_trans)
{
  measure_trans_ = measure_trans;
}

void Kalman::calc_kalman_gain_(const Eigen::MatrixXd &noise_cov)
{
  auto calc_temp = stat_cov_pred_ * measure_trans_.transpose();
  auto calc_temp_2 = measure_trans_ * calc_temp + noise_cov;
  gain_matrix_ = calc_temp * calc_temp_2.inverse();
}

void Kalman::predict()
{
  predict(MatrixXd::Zero(stat_num_, stat_num_), VectorXd::Zero(stat_num_));
}

void Kalman::predict(const Eigen::MatrixXd &control_matrix, const Eigen::VectorXd &control_vector)
{
  stat_pred_ = predict_matrix_ * stat_refine_ + control_matrix * control_vector;
  stat_cov_pred_ =
      predict_matrix_ * stat_cov_refine_ * predict_matrix_.transpose() + external_noise_cov_;
  stat_refine_ = stat_pred_;
  stat_cov_refine_ = stat_cov_pred_;
}

void Kalman::correct(const Eigen::VectorXd &measure_stat)
{
  calc_kalman_gain_(measure_noise_cov_);
  stat_refine_ =
      stat_pred_ + gain_matrix_ * (measure_stat - measure_trans_ * stat_pred_);
  stat_cov_refine_ = stat_cov_pred_ - gain_matrix_ * measure_trans_ * stat_cov_pred_;
}

void Kalman::get_predict_state(Eigen::VectorXd &stat)
{
  stat = stat_pred_;
}

void Kalman::get_correct_state(Eigen::VectorXd &stat)
{
  stat = stat_refine_;
}