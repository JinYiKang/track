#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen>
//https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

class Kalman
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::MatrixXd predict_matrix_;

  Eigen::MatrixXd external_noise_cov_;

  Eigen::MatrixXd measure_trans_;

  Eigen::VectorXd stat_pred_;

  Eigen::MatrixXd stat_cov_pred_;

  Eigen::VectorXd stat_refine_;

  Eigen::MatrixXd stat_cov_refine_;

  Eigen::MatrixXd gain_matrix_;

  int stat_num_;

  int measure_stat_num_;

public:
  Kalman(int state_num, int measure_state_num);

  void set_predict_matrix(const Eigen::MatrixXd &predict_matrix);

  void set_external_noise(const Eigen::MatrixXd &noise_cov);

  void set_measure_transform_matrix(const Eigen::MatrixXd &measure_trans);

  void predict(const Eigen::MatrixXd &control_matrix, const Eigen::VectorXd &control_vector);

  void predict();

  void correct(const Eigen::VectorXd &measure_stat, const Eigen::MatrixXd &measure_noise_cov);

  void get_predict_state(Eigen::VectorXd &stat);

  void get_correct_state(Eigen::VectorXd &stat);

private:
  void calc_kalman_gain_(const Eigen::MatrixXd &noise_cov);
};

#endif //end