#include "kalman.h"
#include "ukf.h"
#include <opencv2/opencv.hpp>
#include <Eigen>
#include <functional>
using namespace Eigen;
using namespace cv;

Mat img(500, 1000, CV_8UC3);
Point mouse_pos;
const int state_num = 2;
const int measure_state_num = 1;

// static void onMouse(int event, int x, int y, int flags, void *param)
// {
//     if (event == EVENT_MOUSEMOVE)
//     {
//         x = std::max(0, x);
//         x = std::min(x, img.cols);
//         y = std::max(0, y);
//         y = std::min(y, img.rows);
//         mouse_pos.x = x;
//         mouse_pos.y = y;
//     }
// }

double calc_func(double x)
{
    return 100.0 * std::sin(x * 0.1) + 250.0;
}

VectorXd precidt_operator(const Eigen::VectorXd &sigma_point)
{
    double x = sigma_point(0);
    double y = sigma_point(1);
    VectorXd sigma_new(state_num);
    sigma_new(0) = x + 1;
    sigma_new(1) = calc_func(x);
    return sigma_new;
}

VectorXd measure_operator(const Eigen::VectorXd &sigma_point)
{
    double y = sigma_point(1);
    VectorXd sigma_new(measure_state_num);
    sigma_new(0) = y;
    return sigma_new;
}

int main()
{
    // Kalman KF(4, 2);
    UKF ukf(state_num, measure_state_num);
    ukf.set_predict_callback(
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)>(precidt_operator));
    ukf.set_measure_transform_callback(
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)>(measure_operator));
    auto measure_state = VectorXd(measure_state_num);
    auto state_pred = VectorXd(state_num);

    namedWindow("test");
    // setMouseCallback("mouse_move", onMouse, 0);
    Point last_pt(0, 250);
    Point last_pt_gt(0, 250);
    for (int i = 0; i < 1000; ++i)
    {
        ukf.predict();
        double x = i*1.0 + rand() * 1.0 / RAND_MAX;
        measure_state(0) = calc_func(x) + (rand() * 50.0 / RAND_MAX - 25.0);
        ukf.correct(measure_state);
        ukf.get_predict_state(state_pred);

        Point cur_pt(i, state_pred(1));
        Point cur_pt_gt(i, calc_func(i));
        line(img, last_pt, cur_pt, Scalar(0, 255, 0));
        // line(img, last_pt_gt, cur_pt_gt, Scalar(255, 255, 255));
        circle(img, Point(x, measure_state(0)), 2, Scalar(0, 0, 255));

        // std::cout << "predict state:" << state_pred << std::endl;
        last_pt = cur_pt;
        last_pt_gt = cur_pt_gt;
        imshow("test", img);
        waitKey(10);
    }
    waitKey(0);

    return 0;
}