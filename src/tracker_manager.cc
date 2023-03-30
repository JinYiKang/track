#include "tracker_manager.hpp"

namespace inno_track
{

  iJIPDA_Tracker::iJIPDA_Tracker() : clutter_spatial_density_(1e-6), max_levels_(2), track_id_(0), delete_dur_sec_(1.0)
  {
    exist_transf_matrix_ << 0.7, 0.2, 0.1,
        0.1, 0.7, 0.2,
        0.1, 0.2, 0.7;
  }

  iJIPDA_Tracker::iJIPDA_Tracker(int max_levels,
                                 double clutter_spatial_density, double delete_dur_sec)
  {
    max_levels_ = max_levels;
    clutter_spatial_density_ = clutter_spatial_density;
    delete_dur_sec_ = delete_dur_sec;
    track_id_ = 0;

    exist_transf_matrix_ << 0.7, 0.2, 0.1,
        0.1, 0.7, 0.2,
        0.1, 0.2, 0.7;
  }

  void iJIPDA_Tracker::set_existence_transform_matrix(
      const Eigen::Matrix3d &transf_mat)
  {
    exist_transf_matrix_ = transf_mat;
  }

  void iJIPDA_Tracker::add_new_track(const Box &box)
  {
    auto new_track = create_new_track_(box);
    assert(new_track.first);

    track_id_++;
    trackers_.insert(std::make_pair(track_id_, new_track.first));
    bayes_variables_.insert(std::make_pair(track_id_, new_track.second));

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    trackers_tp_.insert(std::make_pair(track_id_, tp));
  }

  void iJIPDA_Tracker::update_tracks(const std::vector<Box> &measures)
  {
    std::multimap<uint64_t, size_t> vaild_measures;
    std::multimap<size_t, uint64_t> intersect_tracks;
    std::map<std::pair<uint64_t, size_t>, double> likelihood;

    for (auto iter = trackers_.begin(); iter != trackers_.end(); ++iter)
    {
      iter->second->predict();
    }

    select_vaild_measures_(measures, vaild_measures, intersect_tracks);

    update_likelihood_(measures, vaild_measures, likelihood);

    for (auto iter = bayes_variables_.begin(); iter != bayes_variables_.end();
         ++iter)
    {
      predict_existence_(iter->second);
    }

    for (auto iter = bayes_variables_.begin(); iter != bayes_variables_.end();
         ++iter)
    {
      double sum_weighted_like = 0.0;
      update_state_(measures, vaild_measures, intersect_tracks, likelihood,
                    iter->first, sum_weighted_like);
      update_existence_(iter->second, sum_weighted_like);
    }

    delete_inactive_tracks_(vaild_measures);
  }

  void iJIPDA_Tracker::delete_inactive_tracks_(const std::multimap<uint64_t, size_t> &vaild_measures)
  {
    auto tp = std::chrono::system_clock::now();
    for (auto iter = trackers_tp_.begin(); iter != trackers_tp_.end();)
    {
      if (vaild_measures.count(iter->first) > 0)
      {
        iter->second = tp;
      }
      else
      {
        double delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(tp - iter->second).count();
        if (delta_t * 0.001 > delete_dur_sec_)
        {

          delete trackers_[iter->first];
          trackers_.erase(iter->first);
          bayes_variables_.erase(iter->first);
          iter = trackers_tp_.erase(iter);
          continue;
        }
      }
      ++iter;
    }
  }

  void iJIPDA_Tracker::get_tracks(std::vector<Box> &boxes)
  {
    for (auto iter = trackers_.begin(); iter != trackers_.end(); ++iter)
    {
      Box box = create_box_out_(iter->second->get_state());
      box.id = iter->first;
      boxes.push_back(box);
    }
  }

  void iJIPDA_Tracker::select_vaild_measures_(
      const std::vector<Box> &measures,
      std::multimap<uint64_t, size_t> &vaild_measures,
      std::multimap<size_t, uint64_t> &intersect_tracks)
  {
    for (size_t i = 0; i < measures.size(); ++i)
    {
      const Box &box = measures[i];
      for (auto iter = trackers_.begin(); iter != trackers_.end(); ++iter)
      {
        uint64_t track_id = iter->first;
        KalmanFilter *klm = iter->second;

        if (inner_gate_(klm, box))
        {
          vaild_measures.insert(std::make_pair(track_id, i));
          intersect_tracks.insert(std::make_pair(i, track_id));
        }
      }
    }
  }

  double iJIPDA_Tracker::calc_modulated_clutter_spatial_density_(
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, size_t measure_id, std::set<uint64_t> &track_set,
      std::set<size_t> &measure_set, int level)
  {
    // return clutter spatial density if reach lmax level
    if (max_levels_ > 0 && level > max_levels_)
    {
      return clutter_spatial_density_;
    }

    double sum_R = 0.0;

    // get set of indices of tracks that share a same measure.
    auto shared_tracks_id = intersect_tracks.equal_range(measure_id);

    // update selected measure for next level
    measure_set.insert(measure_id);
    // update selected track for next level
    track_set.insert(track_id);

    for (auto iter = shared_tracks_id.first; iter != shared_tracks_id.second;
         ++iter)
    {
      // a track can only be selected once in a branch
      if (track_set.count(iter->second) == 0)
      {
        sum_R += calc_R_(vaild_measures, intersect_tracks, likelihood,
                         iter->second, measure_id, track_set, measure_set, level);
      }
    }

    // restore selected measure id for lower level
    measure_set.erase(measure_id);
    // restore selected tracks id for lower level
    track_set.erase(track_id);

    return clutter_spatial_density_ * (1.0 + sum_R);
  }

  double iJIPDA_Tracker::calc_R_(
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, size_t measure_id, std::set<uint64_t> &track_set,
      std::set<size_t> &measure_set, int level)
  {
    auto bayes_var_iter = bayes_variables_.find(track_id);

    const BayesVariables &bayes_var = bayes_var_iter->second;
    double p_d = bayes_var.p_d;
    double p_g = bayes_var.p_g;
    double exist_prob = bayes_var.existence(0);

    auto p_like_iter = likelihood.find(std::make_pair(track_id, measure_id));

    double p_like = p_like_iter->second;

    double R = p_d * exist_prob * p_like / clutter_spatial_density_;
    double temp = 1.0 - p_d * p_g * p_like;

    // get set of indices of measures that inner a track gate.
    auto vaild_measures_id = vaild_measures.equal_range(track_id);

    for (auto iter = vaild_measures_id.first; iter != vaild_measures_id.second;
         ++iter)
    {
      // a measure can only be selected once in a branch
      if (measure_set.count(iter->second) == 0)
      {
        auto p_like_temp_iter =
            likelihood.find(std::make_pair(track_id, iter->second));
        double p_like_temp = p_like_temp_iter->second;

        temp += p_d * exist_prob * p_like_temp /
                calc_modulated_clutter_spatial_density_(
                    vaild_measures, intersect_tracks, likelihood, track_id,
                    iter->second, track_set, measure_set, level + 1);
      }
    }

    return R / temp;
  }

  void iJIPDA_Tracker::update_state_(
      const std::vector<Box> &measures,
      const std::multimap<uint64_t, size_t> &vaild_measures,
      const std::multimap<size_t, uint64_t> &intersect_tracks,
      const std::map<std::pair<uint64_t, size_t>, double> &likelihood,
      uint64_t track_id, double &sum_weighted_like)
  {
    // get kalman filter
    auto klm_iter = trackers_.find(track_id);
    KalmanFilter *klm = klm_iter->second;

    auto bayes_var_iter = bayes_variables_.find(track_id);
    const BayesVariables &bayes_var = bayes_var_iter->second;

    std::set<uint64_t> track_set;
    std::set<size_t> measure_set;
    std::map<size_t, double> measures_weighted_like;

    // get set of indices of measures that inner a track gate.
    auto vaild_measures_id = vaild_measures.equal_range(track_id);

    sum_weighted_like = 0.0;

    for (auto iter = vaild_measures_id.first; iter != vaild_measures_id.second;
         ++iter)
    {
      double p_MCSD = calc_modulated_clutter_spatial_density_(
          vaild_measures, intersect_tracks, likelihood, track_id, iter->second,
          track_set, measure_set, 0);

      auto p_like_iter = likelihood.find(std::make_pair(track_id, iter->second));
      auto p_like_weighted = p_like_iter->second / p_MCSD;

      measures_weighted_like.insert(
          std::make_pair(iter->second, p_like_weighted));

      sum_weighted_like += p_like_weighted;
    }

    double sum_association_probabilities =
        1.0 - bayes_var.p_g * bayes_var.p_d + bayes_var.p_d * sum_weighted_like;

    for (auto iter = measures_weighted_like.begin();
         iter != measures_weighted_like.end(); ++iter)
    {
      auto measure_state = get_measure_state_(measures[iter->first]);

      // calculate association probabilities
      double association_probabilities =
          bayes_var.p_d * iter->second / sum_association_probabilities;

      // add weighted measure to kalman filter for correct
      klm->add_weighted_measure(measure_state, association_probabilities);
    }

    // update kalman state
    klm->bayes_update();
  }

  void iJIPDA_Tracker::update_likelihood_(
      const std::vector<Box> &measures,
      const std::multimap<uint64_t, size_t> &vaild_measures,
      std::map<std::pair<uint64_t, size_t>, double> &likelihood)
  {
    for (auto iter = vaild_measures.begin(); iter != vaild_measures.end();
         ++iter)
    {
      uint64_t track_id = iter->first;
      size_t measure_id = iter->second;

      auto klm_iter = trackers_.find(track_id);
      KalmanFilter *klm = klm_iter->second;

      auto measure_state = get_measure_state_(measures[measure_id]);
      auto measure_state_pred = klm->get_state_measure_pred();
      auto inno = measure_state - measure_state_pred;
      auto inno_cov = klm->get_cov_measure_pred();

      double p_like =
          std::exp(-0.5 * inno.transpose() * inno_cov.inverse() * inno) /
          (2.0 * M_PI * std::sqrt(inno_cov.determinant()));

      auto key = std::make_pair(track_id, measure_id);
      likelihood.insert(std::make_pair(key, p_like));
    }
  }

  void iJIPDA_Tracker::update_existence_(BayesVariables &var,
                                         double sum_weighted_like)
  {
    double p_d = var.p_d;
    double p_g = var.p_g;
    Eigen::Vector3d &exist = var.existence;

    double delta = p_d * (p_g - sum_weighted_like);
    exist(1) = exist(1) / (1.0 - delta * exist(0));
    exist(0) = (1.0 - delta) * exist(0) / (1.0 - delta * exist(0));
    exist(2) = 1.0 - exist(0) - exist(1);
  }

  void iJIPDA_Tracker::predict_existence_(BayesVariables &var)
  {
    Eigen::Vector3d &exist = var.existence;
    exist = exist_transf_matrix_.transpose() * exist;
  }

  /**
   * @brief some inherited functions.
   *
   */
  Eigen::VectorXd iJIPDA_Tracker::get_measure_state_(const Box &box)
  {
    Eigen::Vector2d state;
    state << box.x, box.y;
    return state;
  }

  std::pair<KalmanFilter *, BayesVariables>
  iJIPDA_Tracker::create_new_track_(const Box &box)
  {
    KalmanFilter *klm = new KalmanFilter();
    Eigen::Vector4d state;
    state << box.x, box.y, box.speed_x, box.speed_y;
    klm->set_state(state);

    BayesVariables var;
    var.p_d = 0.9;
    var.p_g = 0.989;
    var.existence << 0.9, 0, 0.1;

    return std::make_pair(klm, var);
  }

  Box iJIPDA_Tracker::create_box_out_(const Eigen::VectorXd &state)
  {
    Box box;
    box.x = state(0);
    box.y = state(1);
    box.speed_x = state(2);
    box.speed_y = state(3);
    return box;
  }

  bool iJIPDA_Tracker::inner_gate_(const KalmanFilter *const klm,
                                   const Box &box)
  {
    auto measure_state = get_measure_state_(box);
    auto measure_state_pred = klm->get_state_measure_pred();
    auto inno = measure_state - measure_state_pred;
    auto inno_cov = klm->get_cov_measure_pred();
    double gate_thresh = inno.transpose() * inno_cov.inverse() * inno;
    return gate_thresh <= 9.0;
  }
} // namespace inno_track