#include"cost.h"
#include"Param.h"
#include <fstream>
#include <iostream>
#include "MFA.h"

std::vector<std::string> name_pool;

double cov_pool[Disp_K][Disp_K];

char char_buffer[50];


#define weight_l 0.0005		//0.0004				//	2.0
#define trun_l weight_l*5		//0.002		

void Exp_and_norm(double * msg, double * data,  int width, int height);

void cal_constant_val(cv::Mat & I, int r, std::vector<cv::Mat> & Var_I);
void Guidedfilter_color(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I);
void V_vol_regulation_linear(double * V, double *cost, int width, int height, int level, float weight, float trunc);
void V_vol_regulation_pottos(double * V, double *cost, int width, int height, int level, double p1, double p2);
void Sum_up_cost(double * V, double *cost, int width, int height, int level);
double * cost_cross(cv::Mat & imL, cv::Mat & imR, int L1,int L2 ,int color_diff , int win_max ,int datacost_min );
void stereo_level(double * cost, cv::Mat & im, int F_win_l, double eps, double param_1, double param_2, double * V,cv::Mat & out_im);
void cross_check(cv::Mat & imL, cv::Mat & imR,cv::Mat & Occ_map);
void fill_in_stick(cv::Mat & Disp_map,cv::Mat & Occ_map);
void stereo_pyra(double * dd_cost,cv::Mat & im, int F_win_l, double eps, double param_1, double param_2,int cur_l,double *& old_V,cv::Mat & out_im);
void WMF(cv::Mat & imL, cv::Mat & Disp_map, cv::Mat & Occ_map,double gamma_c,double gamma_d,double r_median);



void build_pyra(double * cost, cv::Mat & im, cv::Mat & new_im, double *& new_cost);

void save_data(const char * name,const char * data,int longth){
	fstream file;
	file.open(name ,ios_base::out|std::ios::binary|ios_base::in|ios_base::trunc );//¸½¼Ó£ºios_base::app
	file.write(data ,longth);
	file.close();
}





void compare_res(cv::Mat & res,cv::Mat & gro){
	cv::Mat c_res(res.rows,res.cols,CV_8UC3,cv::Scalar::all(0));




	for(int j=0;j<res.rows;j++){
		for(int i=0;i<res.cols;i++){

				c_res.data[3*(j*res.cols+i)] = res.data[j*res.cols+i];
				c_res.data[3*(j*res.cols+i)+1] = res.data[j*res.cols+i];
				c_res.data[3*(j*res.cols+i)+2] = res.data[j*res.cols+i];


		}
	}



	for(int j=0;j<res.rows;j++){
		for(int i=0;i<res.cols;i++){
			if(gro.data[j*res.cols+i]==0)
				continue;
			if(abs(res.data[j*res.cols+i]-gro.data[j*res.cols+i])>scale_K){

				c_res.data[3*(j*res.cols+i)] = 0;
				c_res.data[3*(j*res.cols+i)+1] = 0;
				c_res.data[3*(j*res.cols+i)+2] = 255;

			}
		}
	}

    cv::imwrite("/Users/xiaotan/Data/CTMAP/comp_res.png",c_res);

	
}






int main(){


	name_pool.push_back("teddy");


    for(int iii = 0;iii<name_pool.size();iii++){

    std::string directory = "/Users/xiaotan/Data/CTMAP/";

	std::string imL_name = directory+name_pool[iii]+"/imL.png";
	std::string imR_name = directory+name_pool[iii]+"/imR.png";



	cv::Mat imL=cv::imread(imL_name.c_str());
	cv::Mat imR=cv::imread(imR_name.c_str());



	cv::Mat imL_;
	cv::Mat imR_;
	cv::flip(imR,imL_,1);
	cv::flip(imL,imR_,1);







	int height = imL.rows;
	int width = imL.cols;

		
double * cost = cal_cost(imL,imR,15.0/255,4.0/255,0.86,1);				//15,4.0 0.86 1
///////////////////////////////////smooth before using for guiding////////////////////////////////
/*
	cv::medianBlur(imL,imL,3);
	cv::medianBlur(imR,imR,3);
	cv::medianBlur(imL_,imL_,3);
	cv::medianBlur(imR_,imR_,3);
*/

	imL.convertTo(imL,CV_64FC3);
	imR.convertTo(imR,CV_64FC3);

	imL = imL.mul(cv::Mat(height,width,CV_64FC3,cv::Scalar::all(1.0/255)));
	imR = imR.mul(cv::Mat(height,width,CV_64FC3,cv::Scalar::all(1.0/255)));


	imL_.convertTo(imL_,CV_64FC3);
	imR_.convertTo(imR_,CV_64FC3);

	imL_ = imL_.mul(cv::Mat(height,width,CV_64FC3,cv::Scalar::all(1.0/255)));
	imR_ = imR_.mul(cv::Mat(height,width,CV_64FC3,cv::Scalar::all(1.0/255)));


	double delta_step = 1.5;
	double delta_min = 0.00001;




	double * old_V = NULL;
	double * old_V_ = NULL;

	cv::Mat disp_L;
	cv::Mat disp_R;



	stereo_pyra( cost,imL, F_win, Eps, 0.00002, 0.00002*5, 0,old_V,disp_L);
	Exp_and_norm(old_V,cost,width,height);



	cv::Mat out_im;
    disp_out(imL,old_V,Disp_K,out_im);
    cv::imwrite("/Users/xiaotan/Data/CTMAP/teddy/result.png",out_im);



	free(old_V);
	free(old_V_);

		free(cost);
	}

return 0;



}



