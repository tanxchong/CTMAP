#ifndef MFA_HEADER
#define MFA_HEADER

#include "Param.h"
#include "cost.h"
#include "highgui.h"
#include "cvaux.h"
#include "cvwimage.h"
#include "cxcore.h"
#include "ml.h"
#include "cv.h"

void Exp_and_norm(double * msg, double * data,  int width, int height);
void convolu_MFA(double * msg, int width, int height, double w_l, double truncation);
void compute_KL(double * Q, double * cost, double weight, double truncation, int width, int height, cv::Mat & im, std::vector<cv::Mat> & invert_mats, 
	std::vector<cv::Mat> & constant_val, int F_win_l,double eps);
void compute_Energy(double * Q, double * cost, double weight, double truncation, int width, int height, cv::Mat & im, std::vector<cv::Mat> & invert_mats, 
	std::vector<cv::Mat> & constant_val, int F_win_l,double eps);
void convolu_add(double * msg, int width, int height, double w_l, double truncation,double add_term);

#endif
