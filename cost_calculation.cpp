#include"cost.h"
#include"Param.h"
int * census_cost(int width, int height, cv::Mat &imL,cv::Mat& imR,int W, int H,int dis_level,int T){

	int * census_cost = (int *)malloc(sizeof(int)*width*height*dis_level);

	int * census_L = (int *)malloc(sizeof(int)*width*height*W*H);
	int * census_R = (int *)malloc(sizeof(int)*width*height*W*H);
	int j_step = W*H*width;
	int i_step = W*H;


	int ww = (W-1)/2;
	int hh = (H-1)/2;
/////////////////////////////////////////////////////////////////初始化censusL 以及 censusR//////////////////////////////////////////////////////////
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){


			for(int w=-1*ww;w<=ww;w++){
				for(int h=-1*hh;h<=hh;h++){
					if(w+i<0||w+i>=width||h+j<0||h+j>=height)
						census_L[j*j_step+i*i_step+W*(hh+h)+w+ww]=0;
					else{


						if(((double*)imL.data)[(h+j)*width+w+i]-((double*)imL.data)[j*width+i]<0)
							census_L[j*j_step+i*i_step+W*(h+hh)+w+ww]=-1;
						else
							census_L[j*j_step+i*i_step+W*(h+hh)+w+ww]=1;


					}
				}
			}
		}
	}











	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){



			for(int w=-1*ww;w<=ww;w++){
				for(int h=-1*hh;h<=hh;h++){
					if(w+i<0||w+i>=width||h+j<0||h+j>=height)
						census_R[j*j_step+i*i_step+W*(h+hh)+w+ww]=0;
					else{


						if(((double*)imR.data)[(h+j)*width+w+i]-((double*)imR.data)[j*width+i]<0)
							census_R[j*j_step+i*i_step+W*(h+hh)+w+ww]=-1;
						else
							census_R[j*j_step+i*i_step+W*(h+hh)+w+ww]=1;

					}
				}
			}
		}
	}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			for(int l=0;l<=i&&l<dis_level;l++){
				int count_pixels = 0;
				int count_val = 0;
				for(int w = -1*ww;w<=ww;w++){
					for(int h = -1*hh;h<=hh;h++){
						if(census_L[j*j_step+i*i_step+W*(h+hh)+w+ww]!=0&&census_R[j*j_step+(i-l)*i_step+W*(h+hh)+w+ww]!=0){	//两个均不为0
							count_pixels++;
							if(census_L[j*j_step+i*i_step+W*(h+hh)+w+ww]!=census_R[j*j_step+(i-l)*i_step+W*(h+hh)+w+ww])
								count_val++;

						}
					}
				}
				//census_cost[(j*width+i)*dis_level + l] = int(1.0*i_step/count_pixels*count_val);
				census_cost[(j*width+i)*dis_level + l] = int(count_val);
			}
		}
	}


	for(int i=0;i<dis_level;i++){
		for(int l=i+1;l<dis_level;l++){
			for(int j=0;j<height;j++){
				census_cost[(j*width+i)*dis_level + l] = T;
			}
		}
	}

	


	return census_cost;
}




void box_filter(cv::Mat & src, cv::Mat & dst_,int r);
void grad_mat(cv::Mat & input, cv::Mat & output){
	int width = input.cols;
	int height = input.rows;
	output.create(height,width,CV_64F);
	for(int x=1;x<width-1;x++){
		for(int y=0;y<height;y++){

			
			double x_1 = ((double*)input.data)[y*width+x+1]/255.0;
			double x_2 = ((double*)input.data)[y*width+x-1]/255.0;

			((double*)output.data)[y*width+x] = (x_1-x_2)/2.0;

		}
	}
	for(int y=0;y<height;y++){
		((double*)output.data)[y*width] = (((double*)input.data)[y*width+1]-((double*)input.data)[y*width])/255.0;
		((double*)output.data)[y*width+width-1] = (((double*)input.data)[y*width+width-1]-((double*)input.data)[y*width+width-2])/255.0;
	}


}




double * cal_cost(cv::Mat & imL,cv::Mat & imR, double tau1,double tau2 ,double alph,int r){


	
	int width = imL.cols;
	int height = imL.rows;
	double * cost_v = (double*) malloc(sizeof(double)*width*height*Disp_K);
	memset(cost_v,0,sizeof(double)*width*height*Disp_K);

	cv::Mat imL_grad;
	cv::Mat imR_grad;







	cv::Mat imL_gray_F(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat imR_gray_F(height,width,CV_64F,cv::Scalar::all(0));



	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			((double*)imL_gray_F.data)[y*width+x] =  0.29900 * (imL.data)[(y*width+x)*3+2]
			+ 0.58700 * imL.data[(y*width+x)*3+1]
			+ 0.11400 * imL.data[(y*width+x)*3];
			((double*)imR_gray_F.data)[y*width+x] =  0.29900 * (imR.data)[(y*width+x)*3+2]
			+ 0.58700 * imR.data[(y*width+x)*3+1]
			+ 0.11400 * imR.data[(y*width+x)*3];
		}
	}













	grad_mat(imL_gray_F,imL_grad);
	grad_mat(imR_gray_F,imR_grad);
	
	
	double * imL_grad_ptr = (double*) imL_grad.data;
	double * imR_grad_ptr = (double*) imR_grad.data;




	double alph_1 = (1-alph);

	for(int j=0;j<height;j++){
		int step_1 = width*j;
		for(int i=0;i<width;i++){
			int step_2 = (step_1+i)*Disp_K; 
			double r1 = imL.data[(imL.cols*j+i)*3];
			double g1 = imL.data[(imL.cols*j+i)*3+1];
			double b1 = imL.data[(imL.cols*j+i)*3+2];
			double grad_1 =  imL_grad_ptr[(imL_grad.cols*j+i)];


			for(int l=0;l<Disp_K&&(i-l-1)>=0;l++){
				double r2 = imR.data[(imR.cols*j+i-l)*3];
				double g2 = imR.data[(imR.cols*j+i-l)*3+1];
				double b2 = imR.data[(imR.cols*j+i-l)*3+2];
				double grad_2 = imR_grad_ptr[(imR_grad.cols*j+i-l)];

				double diff_intensity = (abs(r1-r2)+abs(b1-b2)+abs(g1-g2))/3.0/255;


				double diff_devi = abs(grad_1-grad_2);


				cost_v[l*width*height+j*width+i] = alph_1*min(tau1,diff_intensity)+alph*min(tau2,diff_devi);

			}

			for(int l = i;l<Disp_K;l++){
				double diff_intensity = (abs(r1-Threshold_B)+abs(b1-Threshold_B)+abs(g1-Threshold_B))/3.0/255;


				double diff_devi = abs(grad_1-Threshold_B);


				cost_v[l*width*height+j*width+i] = (alph_1*min(tau1,diff_intensity)+alph*min(tau2,diff_devi));
			}


		}
	}

	if(r==0){
		for(int y=0;y<height;y++){
		for(int x=0;x<Disp_K-1;x++){
			float avg=0;
			float best = 10000;
			for(int level=0;level<x+1;level++){
				avg+=cost_v[width*height*level+width*y+x];
				if(best>cost_v[width*height*level+width*y+x])
					best = cost_v[width*height*level+width*y+x];
			}
			avg/=x+1;
			for(int level=0;level<x+1;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
			for(int level=x+1;level<Disp_K;level++){
				cost_v[width*height*level+width*y+x]=avg;			//imR
			}

		}
		for(int x=Disp_K-1;x<width;x++){
			float avg=0;
			for(int level=0;level<Disp_K;level++){
				avg+=cost_v[width*height*level+width*y+x];
			}
			avg/=Disp_K;
			for(int level=0;level<Disp_K;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
		}
	}




	return cost_v;
	}

	cv::Mat N = cv::Mat::ones(height,width,CV_64F);

	box_filter(N,N,r);

	for(int l=0;l<Disp_K;l++){
		cv::Mat tt_p(height,width,CV_64F,cost_v+l*height*width);
		box_filter(tt_p,tt_p,r);
		tt_p = tt_p/N;
		memcpy(cost_v+l*height*width,tt_p.data,sizeof(double)*width*height);

	}



	for(int y=0;y<height;y++){
		for(int x=0;x<Disp_K-1;x++){
			float avg=0;
			float best = 10000;
			for(int level=0;level<x+1;level++){
				avg+=cost_v[width*height*level+width*y+x];
				if(best>cost_v[width*height*level+width*y+x])
					best = cost_v[width*height*level+width*y+x];
			}
			avg/=x+1;
			for(int level=0;level<x+1;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
			for(int level=x+1;level<Disp_K;level++){
				cost_v[width*height*level+width*y+x]=avg;			//imR
			}

		}
		for(int x=Disp_K-1;x<width;x++){
			float avg=0;
			for(int level=0;level<Disp_K;level++){
				avg+=cost_v[width*height*level+width*y+x];
			}
			avg/=Disp_K;
			for(int level=0;level<Disp_K;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
		}
	}




	return cost_v;
}



double * cal_cost_robust(cv::Mat & imL,cv::Mat & imR, double tau1,double tau2 ,int r){


	
	int width = imL.cols;
	int height = imL.rows;
	double * cost_v = (double*) malloc(sizeof(double)*width*height*Disp_K);
	memset(cost_v,0,sizeof(double)*width*height*Disp_K);

	cv::Mat imL_grad;
	cv::Mat imR_grad;







	cv::Mat imL_gray_F(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat imR_gray_F(height,width,CV_64F,cv::Scalar::all(0));



	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			((double*)imL_gray_F.data)[y*width+x] =  0.29900 * imL.data[(y*width+x)*3+2]
			+ 0.58700 * imL.data[(y*width+x)*3+1]
			+ 0.11400 * imL.data[(y*width+x)*3];
			((double*)imR_gray_F.data)[y*width+x] =  0.29900 * imR.data[(y*width+x)*3+2]
			+ 0.58700 * imR.data[(y*width+x)*3+1]
			+ 0.11400 * imR.data[(y*width+x)*3];
		}
	}













	grad_mat(imL_gray_F,imL_grad);
	grad_mat(imR_gray_F,imR_grad);
	
	
	double * imL_grad_ptr = (double*) imL_grad.data;
	double * imR_grad_ptr = (double*) imR_grad.data;





	for(int j=0;j<height;j++){
		int step_1 = width*j;
		for(int i=0;i<width;i++){
			int step_2 = (step_1+i)*Disp_K; 
			double r1 = imL.data[(imL.cols*j+i)*3];
			double g1 = imL.data[(imL.cols*j+i)*3+1];
			double b1 = imL.data[(imL.cols*j+i)*3+2];
			double grad_1 =  imL_grad_ptr[(imL_grad.cols*j+i)];


			for(int l=0;l<Disp_K&&(i-l-1)>=0;l++){
				double r2 = imR.data[(imR.cols*j+i-l)*3];
				double g2 = imR.data[(imR.cols*j+i-l)*3+1];
				double b2 = imR.data[(imR.cols*j+i-l)*3+2];
				double grad_2 = imR_grad_ptr[(imR_grad.cols*j+i-l)];

				double diff_intensity = (abs(r1-r2)+abs(b1-b2)+abs(g1-g2))/3.0;


				double diff_devi = abs(grad_1-grad_2)*255.0;


				cost_v[l*width*height+j*width+i] = regulation_data_weight*(2-std::exp(-1.0*diff_intensity/tau1)-std::exp(-1.0*diff_devi/tau2));

			}

			for(int l = i;l<Disp_K;l++){

				double diff_intensity = (abs(r1-Threshold_B)+abs(b1-Threshold_B)+abs(g1-Threshold_B))/3.0;


				double diff_devi = abs(grad_1-Threshold_B)*255.0;


				cost_v[l*width*height+j*width+i] = regulation_data_weight*(2-std::exp(-1.0*diff_intensity/tau1)-std::exp(-1.0*diff_devi/tau2));
			}


		}
	}

	if(r==0){
		for(int y=0;y<height;y++){
		for(int x=0;x<Disp_K-1;x++){
			float avg=0;
			float best = 10000;
			for(int level=0;level<x+1;level++){
				avg+=cost_v[width*height*level+width*y+x];
				if(best>cost_v[width*height*level+width*y+x])
					best = cost_v[width*height*level+width*y+x];
			}
			avg/=x+1;
			for(int level=0;level<x+1;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
			for(int level=x+1;level<Disp_K;level++){
				cost_v[width*height*level+width*y+x]=avg;			//imR
			}

		}
		for(int x=Disp_K-1;x<width;x++){
			float avg=0;
			for(int level=0;level<Disp_K;level++){
				avg+=cost_v[width*height*level+width*y+x];
			}
			avg/=Disp_K;
			for(int level=0;level<Disp_K;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
		}
	}




	return cost_v;
	}

	cv::Mat N = cv::Mat::ones(height,width,CV_64F);

	box_filter(N,N,r);

	for(int l=0;l<Disp_K;l++){
		cv::Mat tt_p(height,width,CV_64F,cost_v+l*height*width);
		box_filter(tt_p,tt_p,r);
		tt_p = tt_p/N;
		memcpy(cost_v+l*height*width,tt_p.data,sizeof(double)*width*height);

	}



	for(int y=0;y<height;y++){
		for(int x=0;x<Disp_K-1;x++){
			float avg=0;
			float best = 10000;
			for(int level=0;level<x+1;level++){
				avg+=cost_v[width*height*level+width*y+x];
				if(best>cost_v[width*height*level+width*y+x])
					best = cost_v[width*height*level+width*y+x];
			}
			avg/=x+1;
			for(int level=0;level<x+1;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
			for(int level=x+1;level<Disp_K;level++){
				cost_v[width*height*level+width*y+x]=avg;			//imR
			}

		}
		for(int x=Disp_K-1;x<width;x++){
			float avg=0;
			for(int level=0;level<Disp_K;level++){
				avg+=cost_v[width*height*level+width*y+x];
			}
			avg/=Disp_K;
			for(int level=0;level<Disp_K;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
		}
	}




	return cost_v;
}



void test_vol(cv::Mat & im, double * datacost,int level){
	cv::Mat disp(im.rows,im.cols,CV_8UC1,cv::Scalar::all(0));
	int width = im.cols;
	int height = im.rows;

	for(int y=0;y<im.rows;y++){
		for(int x=0;x<im.cols;x++){
			double best_cost=1E10;
			int best_val=0;
			for(int i=0;i<level;i++){
				if(datacost[i*width*height+y*width+x]<best_cost){
					best_cost=datacost[i*width*height+y*width+x];
					best_val=i*scale_K;
			
				}
			}
			disp.data[y*im.cols+x]=best_val;
		}
	}
	cv::imwrite("D:/teddy.png",disp);
	cv::namedWindow("test",0);
	cv::imshow("test",disp);
	cv::waitKey();
}

void disp_out(cv::Mat & im, double * datacost,int level,cv::Mat & out_im){
	out_im.create(im.rows,im.cols,CV_8UC1);
	int width = im.cols;
	int height = im.rows;




	for(int y=0;y<im.rows;y++){
		for(int x=0;x<im.cols;x++){
			double best_cost=datacost[y*width+x];
			int best_val=0;
			for(int i=0;i<level;i++){
				if(datacost[i*width*height+y*width+x]>best_cost){
					best_cost=datacost[i*width*height+y*width+x];
					best_val=i*scale_K;
			
				}
			}
			out_im.data[y*im.cols+x]=best_val;
		}
	}
}
double * cal_cost_ad_census(cv::Mat & imL,cv::Mat & imR, double T, double lambda_color, double lambda_census,int r){


	
	int width = imL.cols;
	int height = imL.rows;
	double * cost_v = (double*) malloc(sizeof(double)*width*height*Disp_K);
	memset(cost_v,0,sizeof(double)*width*height*Disp_K);

	cv::Mat imL_grad;
	cv::Mat imR_grad;







	cv::Mat imL_gray_F(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat imR_gray_F(height,width,CV_64F,cv::Scalar::all(0));



	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			((double*)imL_gray_F.data)[y*width+x] =  0.29900 * imL.data[(y*width+x)*3+2]
			+ 0.58700 * imL.data[(y*width+x)*3+1]
			+ 0.11400 * imL.data[(y*width+x)*3];
			((double*)imR_gray_F.data)[y*width+x] =  0.29900 * imR.data[(y*width+x)*3+2]
			+ 0.58700 * imR.data[(y*width+x)*3+1]
			+ 0.11400 * imR.data[(y*width+x)*3];
		}
	}













	grad_mat(imL_gray_F,imL_grad);
	grad_mat(imR_gray_F,imR_grad);
	
	
	double * imL_grad_ptr = (double*) imL_grad.data;
	double * imR_grad_ptr = (double*) imR_grad.data;


	int * census_c = census_cost(width, height, imL_gray_F,imR_gray_F,9,7,Disp_K,40);




	for(int j=0;j<height;j++){
		int step_1 = width*j;
		for(int i=0;i<width;i++){
			int step_2 = (step_1+i)*Disp_K; 
			double r1 = imL.data[(imL.cols*j+i)*3];
			double g1 = imL.data[(imL.cols*j+i)*3+1];
			double b1 = imL.data[(imL.cols*j+i)*3+2];
			double grad_1 =  imL_grad_ptr[(imL_grad.cols*j+i)];


			for(int l=0;l<Disp_K&&(i-l-1)>=0;l++){
				double r2 = imR.data[(imR.cols*j+i-l)*3];
				double g2 = imR.data[(imR.cols*j+i-l)*3+1];
				double b2 = imR.data[(imR.cols*j+i-l)*3+2];
				double grad_2 = imR_grad_ptr[(imR_grad.cols*j+i-l)];

				double cost_color = min((abs(r1-r2)+abs(b1-b2)+abs(g1-g2))/3.0,T);


				float cost_census_p = float(census_c[(imL.cols*j+i)*Disp_K+l]);


				cost_v[l*width*height+j*width+i] = (2.0-exp(-1.0*cost_color/lambda_color)-exp(-1.0*cost_census_p/lambda_census))*0.02;

			}

			for(int l = i;l<Disp_K;l++){
				double cost_color = min((abs(r1-Threshold_B)+abs(b1-Threshold_B)+abs(g1-Threshold_B))/3.0,T);


				double cost_census_p = 20;


				cost_v[l*width*height+j*width+i] = (2.0-exp(-1.0*cost_color/lambda_color)-exp(-1.0*cost_census_p/lambda_census))*0.02;
			}


		}
	}
	cv::Mat N = cv::Mat::ones(height,width,CV_64F);

	box_filter(N,N,r);

	for(int l=0;l<Disp_K;l++){
		cv::Mat tt_p(height,width,CV_64F,cost_v+l*height*width);
		box_filter(tt_p,tt_p,r);
		tt_p = tt_p/N;
		memcpy(cost_v+l*height*width,tt_p.data,sizeof(double)*width*height);

	}

	for(int y=0;y<height;y++){
		for(int x=0;x<Disp_K-1;x++){
			float avg=0;
			float best = 10000;
			for(int level=0;level<x+1;level++){
				avg+=cost_v[width*height*level+width*y+x];
				if(best>cost_v[width*height*level+width*y+x])
					best = cost_v[width*height*level+width*y+x];
			}
			avg/=x+1;
			for(int level=0;level<x+1;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
			for(int level=x+1;level<Disp_K;level++){
				cost_v[width*height*level+width*y+x]=avg;			//imR
			}

		}
		for(int x=Disp_K-1;x<width;x++){
			float avg=0;
			for(int level=0;level<Disp_K;level++){
				avg+=cost_v[width*height*level+width*y+x];
			}
			avg/=Disp_K;
			for(int level=0;level<Disp_K;level++){
				if(cost_v[width*height*level+width*y+x]>avg){
					cost_v[width*height*level+width*y+x]=avg;
				}
			}
		}
	}






	free(census_c);
	return cost_v;
}

