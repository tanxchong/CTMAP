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
#include "Param.h"

#define levelK Disp_K			//	level of disparity searching
#define lambda_color 10
#define lambda_census 30
#define census_color 3000





double get_Ele(cv::Mat & res, int x, int y, int channel);
int * get_arm(cv::Mat & res,int T,int L);
double * row_cost(cv::Mat & imL,cv::Mat &imR,int dis, double T,int * census_cost);



double * edh_cost(double * row_cost, int width, int height,int disp, int * arm_L, int *arm_R);
double * ed_cost(double * edh_cost,int width,int height);




double * cost_L(cv::Mat & imL,cv::Mat & imR, int L, double T /*truncation*/,int * arm_L, int *arm_R,int *census_cost);
double * cost(cv::Mat & imL, cv::Mat & imR, int L1,int L2,int color_diff =30, int win_max = 17,int datacost_min =60);
void output_cost(uchar * output ,double * cost, int width, int height, int L);
void output_cost_unscaled(uchar * output ,double * cost, int width, int height, int L);
int * census_cost(cv::Mat & imL,cv::Mat &imR,cv::Mat &imL_,cv::Mat &imR_,int W, int H,int dis_level,int T);






void calac_weight(cv::Mat & res, int x, int y, double * weight, int u,int d,int l,int r,double rc,double rp);
double color_cost(cv::Mat & imL,cv::Mat & imR, int x, int y, int disp,double T);
double * cost_L(cv::Mat & imL, cv::Mat & imR, int L, int win,double rc,double rp,double T,int *census_cost);
double * calc_cost(cv::Mat & imL, cv::Mat & imR, int L, int win, double rc, double rp, double T);



void view_data(const char * name);
void save_data(const char * name,const char * data,int longth);

#endif
