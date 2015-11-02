#ifndef data_cost
#define data_cost
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "highgui.h"
#include "cvaux.h"
#include "cvwimage.h"
#include "cxcore.h"
#include "ml.h"
#include "cv.h"
#include "opencv2/opencv.hpp"






double * cal_cost(cv::Mat & imL,cv::Mat & imR, double tau1,double tau2,double alph, int r);
double * cal_cost_robust(cv::Mat & imL,cv::Mat & imR, double tau1,double tau2 ,int r);
double * cal_cost_ad_census(cv::Mat & imL,cv::Mat & imR, double T, double lambda_color, double lambda_census,int r);


void test_vol(cv::Mat & im, double * datacost,int level);
void cal_constant_val(cv::Mat & I, int r,double eps, std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert);
void Guidedfilter_color(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I);
void Guidedfilter_sum(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert_mats);
void Guidedfilter_median(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert_mats);
void V_vol_regulation_pottos(double * V, double *cost, int width, int height, int level, double p1, double p2);
void V_vol_regulation_send(double * V, double *cost, int width, int height, int level, float weight, float trunc);
void V_vol_regulation_linear(double * V, double *cost, int width, int height, int level, float weight, float trunc);
void Sum_up_cost(double * V, double *cost, int width, int height, int level);
void disp_out(cv::Mat & im, double * datacost,int level,cv::Mat & out_im);
#endif
