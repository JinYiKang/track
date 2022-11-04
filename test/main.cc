#include "kalman.h"
#include <opencv2/opencv.hpp>
#include <Eigen>
using namespace Eigen;
using namespace cv;

Mat img(1000, 1000, CV_8UC3);
Point mouse_pos;

static void onMouse(int event, int x, int y, int flags, void *param)
{
    if (event == EVENT_MOUSEMOVE)
    {
        x = std::max(0, x);
        x = std::min(x, img.cols);
        y = std::max(0, y);
        y = std::min(y, img.rows);
        mouse_pos.x = x;
        mouse_pos.y = y;
    }
}

int main()
{
    Kalman KF(4, 2);
    auto pred_mat = MatrixXd(4, 4);
    pred_mat << 1, 0, 0.015, 0,
        0, 1, 0, 0.015,
        0, 0, 1, 0,
        0, 0, 0, 1;
    auto measure_mat = MatrixXd(2, 4);
    measure_mat << 1, 0, 0, 0,
        0, 1, 0, 0;

    auto measure_noise_cov = MatrixXd::Identity(2, 2) * 0.001;
    auto measure_state = VectorXd(2);
    auto state_pred = VectorXd(4);

    KF.set_predict_matrix(pred_mat);
    KF.set_measure_transform_matrix(measure_mat);
    KF.set_external_noise(MatrixXd::Identity(4, 4) * 0.001);

    namedWindow("mouse_move");
    setMouseCallback("mouse_move", onMouse, 0);
    while (1)
    {
        KF.predict();
        measure_state(0) = mouse_pos.x;
        measure_state(1) = mouse_pos.y;
        KF.correct(measure_state, measure_noise_cov);
        KF.get_predict_state(state_pred);
        circle(img, Point(state_pred(0), state_pred(1)), 2, Scalar(0, 255, 0), 2);
        std::cout << "predict state:" << state_pred(0) << " , " << state_pred(1)
                  << " , " << state_pred(2) << " , " << state_pred(3) << std::endl;

        imshow("mouse_move", img);
        char c = (char)waitKey(10);
        if (c == 's')
            break;
        else if (c == 'c')
        {
            img.setTo(0);
        }
    }
    
    return 0;
}