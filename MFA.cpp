#include "MFA.h"


void convolu_MFA(double * msg, int width, int height, double w_l, double truncation){

	double conv_pool[Disp_K][Disp_K];
	for(int ii=0;ii<Disp_K;ii++){
		for(int jj=0;jj<Disp_K;jj++){
			conv_pool[ii][jj] = abs(ii-jj)*w_l;
		}
	}

	for(int ii=0;ii<Disp_K;ii++){
		for(int jj=0;jj<Disp_K;jj++){
			if(conv_pool[ii][jj]>truncation)
			conv_pool[ii][jj] = truncation;
		}
	}

	double * out_Q = (double*)malloc(sizeof(double)*Disp_K*width*height);
	memset(out_Q,0,sizeof(double)*Disp_K*width*height);

	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				for(int r=0;r<Disp_K;r++){
					out_Q[l*width*height+y*width+x] += 
						conv_pool[r][l]*msg[r*width*height+y*width+x];
				}
			}
		}
	}

	memcpy(msg,out_Q,sizeof(double)*Disp_K*width*height);
	free(out_Q);

}

void convolu_add(double * msg, int width, int height, double w_l, double truncation,double add_term){		//max(-V(xi,xj)Q(xj))

	double conv_pool[Disp_K][Disp_K];
	for(int ii=0;ii<Disp_K;ii++){
		for(int jj=0;jj<Disp_K;jj++){
			conv_pool[ii][jj] = abs(ii-jj)*w_l+add_term;
		}
	}
	for(int ii=0;ii<Disp_K;ii++){
		for(int jj=0;jj<Disp_K;jj++){
			if(conv_pool[ii][jj]>truncation+add_term)
			conv_pool[ii][jj] = truncation+add_term;
		}
	}

	double * out_Q = (double*)malloc(sizeof(double)*Disp_K*width*height);
	memset(out_Q,0,sizeof(double)*Disp_K*width*height);

	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				out_Q[l*width*height+y*width+x] = 1000000;
				for(int r=0;r<Disp_K;r++){
					double this_v = msg[r*width*height+y*width+x]*conv_pool[r][l];
					if(out_Q[l*width*height+y*width+x]>this_v){
						out_Q[l*width*height+y*width+x] = this_v;
					}
				}
			}
		}
	}

	memcpy(msg,out_Q,sizeof(double)*Disp_K*width*height);
	free(out_Q);

}

void Exp_and_norm(double * msg, double * data,  int width, int height){
	double * out_Q = (double*)malloc(sizeof(double)*Disp_K*width*height);
	memset(out_Q,0,sizeof(double)*Disp_K*width*height);

	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			double min_v = 10E20;
			int best_v = 0;
				//find the min value; prevent explosion
			for(int r=0;r<Disp_K;r++){
				if(min_v>(msg[r*width*height+y*width+x]+
					data[r*width*height+y*width+x])){
					min_v = msg[r*width*height+y*width+x]+
					data[r*width*height+y*width+x];
					best_v = r;
				}
			}	
				
			//normlize the msg; prevent explosion
			for(int r=0;r<Disp_K;r++){
				out_Q[r*width*height+y*width+x] = msg[r*width*height+y*width+x]+
					data[r*width*height+y*width+x] - min_v;
			}

			for(int r=0;r<Disp_K;r++){
				out_Q[r*width*height+y*width+x] = 0;
			}
			out_Q[best_v*width*height+y*width+x] =1;
		}
	}



	memcpy(msg,out_Q,sizeof(double)*Disp_K*width*height);
	free(out_Q);
}

void compute_KL(double * Q, double * cost, double weight, double truncation, int width, int height, cv::Mat & im, std::vector<cv::Mat> & invert_mats, 
	std::vector<cv::Mat> & constant_val, int F_win_l,double eps){

	
	



	double E_uQ = 0;
	double E_uD = 0;
	double E_uP = 0;










	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				if(Q[l*width*height+y*width+x]>10E-30)
				E_uQ += Q[l*width*height+y*width+x]*log(Q[l*width*height+y*width+x]);
			}
		}
	}


	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				if(Q[l*width*height+y*width+x]>10E-30)
				E_uD += Q[l*width*height+y*width+x]*cost[l*width*height+y*width+x];
			}
		}
	}

	double * comp_msg = (double *) malloc(sizeof(double)*width*height*Disp_K);
	memcpy(comp_msg,Q,sizeof(double)*width*height*Disp_K);
	convolu_MFA(comp_msg,width,height,weight,truncation);



	for(int ll = 0;ll<Disp_K;ll++){				
			cv::Mat filtering_l(height,width,CV_64F,comp_msg+ll*height*width);
			Guidedfilter_sum(im, filtering_l,filtering_l, F_win_l, eps*eps,constant_val,invert_mats);
		}


	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				E_uP += Q[l*width*height+y*width+x]*comp_msg[l*width*height+y*width+x];
			}
		}
	}
	E_uP /=2.0;

	double KL = E_uQ + E_uP + E_uD;

	std::cout<<KL<<endl;

}

void compute_Energy(double * Q, double * cost, double weight, double truncation, int width, int height, cv::Mat & im, std::vector<cv::Mat> & invert_mats, 
	std::vector<cv::Mat> & constant_val, int F_win_l,double eps){

	
	int * disp_map = (int*)malloc(sizeof(int)*width*height);


	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			double best_val = Q[y*width+x];
			disp_map[y*width+x] = 0;
			for(int l=0;l<Disp_K;l++){
				if(Q[l*width*height+y*width+x]<best_val){
					disp_map[y*width+x] = l;
					best_val = Q[l*width*height+y*width+x];
				}
			}
		}
	}


	double E_uD = 0;
	double E_P = 0;

	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			int this_d = disp_map[y*width+x];
			E_uD += cost[this_d*width*height+y*width+x];
		}
	}

	double * comp_msg = (double *) malloc(sizeof(double)*width*height*Disp_K);
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			int this_d = disp_map[y*width+x];
			for(int l=0;l<Disp_K;l++){
				comp_msg[l*width*height+y*width+x] = min(weight*abs(l-this_d),truncation);
			}
		}
	}

	for(int ll = 0;ll<Disp_K;ll++){				
			cv::Mat filtering_l(height,width,CV_64F,comp_msg+ll*height*width);
			Guidedfilter_sum(im, filtering_l,filtering_l, F_win_l, eps*eps,constant_val,invert_mats);
		}


	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			int this_d = disp_map[y*width+x];
				E_P += comp_msg[this_d*width*height+y*width+x];
		}
	}
	E_P /=2.0;

	double EEE = E_P + E_uD;

	std::cout<<EEE<<endl;
	free(disp_map);

}

