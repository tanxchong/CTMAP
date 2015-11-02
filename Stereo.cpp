#include"cost.h"
#include"Param.h"
#include "MFA.h"




struct st_Sparce_hist{
	int disp;
	double weight;
};
bool cmp( const st_Sparce_hist* a,  const st_Sparce_hist* b)  
{ 
	if(a->disp<b->disp)
      return true;
	return false;
} 
void initial_Q( double * out_Q, int width, int height,double interlevel){
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<Disp_K;l++){
				out_Q[l*width*height+y*width+x] /= interlevel;
			}
		}
	}
}



void convolu(double * msg, double * data,  int width, int height, double w_l, double truncation){

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
	for(int ii=0;ii<Disp_K;ii++){
		for(int jj=0;jj<Disp_K;jj++){
			conv_pool[ii][jj] = std::exp(-1.0*conv_pool[ii][jj]);
		}
	}

	double * out_Q = (double*)malloc(sizeof(double)*Disp_K*width*height);
	memset(out_Q,0,sizeof(double)*Disp_K*width*height);

	for(int jj=0;jj<height;jj++){
		for(int ii=0;ii<width;ii++){
			double min_v = 100000.0;
			for(int ll=0;ll<Disp_K;ll++){
				if(msg[ll*width*height+jj*width+ii]<min_v)
					min_v = msg[ll*width*height+jj*width+ii];
			}
			for(int ll=0;ll<Disp_K;ll++){
				msg[ll*width*height+jj*width+ii] = msg[ll*width*height+jj*width+ii] - min_v-exp_tran;
				if(msg[ll*width*height+jj*width+ii] > exp_tran)
					msg[ll*width*height+jj*width+ii] = exp_tran;
			}
		}
	}
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			double z = 0;
			for(int l=0;l<Disp_K;l++){
				for(int r=0;r<Disp_K;r++){
					out_Q[l*width*height+y*width+x] += 
						conv_pool[r][l]*std::exp(-1.0*min(msg[r*width*height+y*width+x],exp_tran)-data[r*width*height+y*width+x]);
				}
				z+=out_Q[l*width*height+y*width+x];
			}
			//z = z*1.0/Disp_K;


			for(int l=0;l<Disp_K;l++){
				out_Q[l*width*height+y*width+x] = -1.0*log(out_Q[l*width*height+y*width+x]/z);
			}
		}
	}
	memcpy(msg,out_Q,sizeof(double)*Disp_K*width*height);
	free(out_Q);

}





void build_pyra(double * cost, cv::Mat & im, cv::Mat & new_im, double *& new_cost){
	int old_width = im.cols;
	int old_height = im.rows;
	int new_width = (int)ceil(old_width/2.0);
	int new_height = (int)ceil(old_height/2.0);


	new_cost = (double *)malloc(sizeof(double)*new_width*new_height*Disp_K);
	new_im.create(new_height,new_width,CV_64FC3);
	memset(new_im.data,0,sizeof(double)*3*new_height*new_width);
	memset(new_cost,0,sizeof(double)*new_width*new_height*Disp_K);
	

	for (int y = 0; y < old_height; y++) {
		for (int x = 0; x < old_width; x++) {
			((double*)new_im.data)[((y/2)*new_width+x/2)*3]	+= ((double*)im.data)[((y)*old_width+x)*3];
			((double*)new_im.data)[((y/2)*new_width+x/2)*3+1]	+= ((double*)im.data)[((y)*old_width+x)*3+1];
			((double*)new_im.data)[((y/2)*new_width+x/2)*3+2]	+= ((double*)im.data)[((y)*old_width+x)*3+2];
			for (int d = 0; d < Disp_K; d++) {
			  new_cost[d*new_width*new_height+((y/2)*new_width+x/2)] += cost[d*old_height*old_width+(y*old_width+x)];
			}
		}
	}

	for(int y=0;y<old_height/2;y++){
		for(int x=0;x<old_width/2;x++){
			((double*)new_im.data)[((y)*new_width+x)*3]	/= 4;
			((double*)new_im.data)[((y)*new_width+x)*3+1]/= 4;
			((double*)new_im.data)[((y)*new_width+x)*3+2]/= 4;
			
		}
	}
	if(old_width/2<new_width){
		for(int y=0;y<old_height/2;y++){
			((double*)new_im.data)[((y)*new_width+new_width-1)*3]	/= 2;
			((double*)new_im.data)[((y)*new_width+new_width-1)*3+1]	/= 2;
			((double*)new_im.data)[((y)*new_width+new_width-1)*3+2]	/= 2;
		}
	}

	if(old_height/2<new_height){
		for(int x=0;x<old_width/2;x++){
			((double*)new_im.data)[((new_height-1)*new_width+x)*3]	/= 2;
			((double*)new_im.data)[((new_height-1)*new_width+x)*3+1]	/= 2;
			((double*)new_im.data)[((new_height-1)*new_width+x)*3+2]	/= 2;
		}
	}
}



void stereo_level(double * cost, cv::Mat & im, int F_win_l, double eps, double param_1, double param_2, double * V,cv::Mat & out_im){
	int width = im.cols;
	int height = im.rows;



	std::vector<cv::Mat> constant_val(10);

	std::vector<cv::Mat> invert_mats(width*height);
	cal_constant_val(im,F_win_l,eps*eps,constant_val,invert_mats);




for(int iter = 0;iter<Iter;iter++){
    Exp_and_norm(V,cost,width,height);
		
							//minimum convolution
///////////////////////////////////////////////////do filtering////////////////////////////////////////////////	

		convolu_MFA(V,width,height,param_1,param_2);
		
		
		for(int ll = 0;ll<Disp_K;ll++){				
			cv::Mat filtering_l(height,width,CV_64F,V+ll*height*width);
			Guidedfilter_sum(im, filtering_l,filtering_l, F_win_l, eps*eps,constant_val,invert_mats);
		}
	}
	double * sum_vol = (double *) malloc(sizeof(double)*width*height*Disp_K);
	memcpy(sum_vol,V,sizeof(double)*width*height*Disp_K);
	free(sum_vol);
}











void stereo_pyra(double * dd_cost,cv::Mat & im, int F_win_l, double eps, double param_1, double param_2,int cur_l,double *& old_V,cv::Mat & out_im){

	int width = im.cols;
	int height =im.rows;



	double * new_V = (double*)malloc(sizeof(double)*width*height*Disp_K);
	memset(new_V,0,sizeof(double)*width*height*Disp_K);


	if(cur_l!=LEVELS){
		cv::Mat new_im;
		double * new_cost; 
		build_pyra(dd_cost,im,new_im,new_cost);
		stereo_pyra(new_cost,new_im,F_win_l,eps, param_1*inter_level_U,param_2*inter_level_U,cur_l+1,old_V,out_im);

		// updata_V
		for(int x=0;x<width;x++){
			for(int y=0;y<height;y++){
				for(int d = 0;d<Disp_K;d++){
					new_V[(width*height)*d+(width*y+x)] = old_V[((int)ceil(width/2.0)*(int)ceil(height/2.0))*d+int(((int)ceil(width/2.0)*int(y/2)+x/2))];
				}
			}
		}


		free(new_cost);
		stereo_level(dd_cost,im,F_win_l,eps,param_1,param_2,new_V,out_im);

		if(cur_l!=0){
		//////////////////////
		double * sum_vol = (double *) malloc(sizeof(double)*width*height*Disp_K);
		memcpy(sum_vol,new_V,sizeof(double)*width*height*Disp_K);
		initial_Q( new_V,  width,  height, 4);															//几个相乘取其中一个
		free(sum_vol);


///////////////////////////
		}

		free(old_V);
		old_V = new_V;

		return;

	}
	else{
		stereo_level(dd_cost,im,F_win_l,eps,param_1,param_2,new_V,out_im);
		if(cur_l!=0){
		double * sum_vol = (double *) malloc(sizeof(double)*width*height*Disp_K);
		memcpy(sum_vol,new_V,sizeof(double)*width*height*Disp_K);
		initial_Q( new_V,  width,  height, 4);															//几个相乘取其中一个
		free(sum_vol);
		}
		free(old_V);
		old_V = new_V;
		return;
	}
}
void cross_check(cv::Mat & imL, cv::Mat & imR,cv::Mat & Occ_map){
	int width = imL.cols;
	int height = imL.rows;
	cv::flip(imR,imR,1);
	Occ_map.create(height,width,CV_8UC1);
	memset(Occ_map.data,0,height*width);
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			if(i-imL.data[j*width+i]/scale_K<0){
				Occ_map.data[j*width+i] = 255;
				continue;
			}
			if(abs(1.0*imR.data[j*width+i-imL.data[j*width+i]/scale_K]-imL.data[j*width+i])>=scale_K)
				Occ_map.data[j*width+i] = 255;
		}
	}


};




void fill_in_stick(cv::Mat & Disp_map,cv::Mat & Occ_map){
	int width = Disp_map.cols;
	int height = Disp_map.rows;

	
	cv::Mat fill_left(height,width,CV_8UC1,cv::Scalar::all(0));
	cv::Mat fill_right(height,width,CV_8UC1,cv::Scalar::all(0));

	Disp_map.copyTo(fill_left);
	Disp_map.copyTo(fill_right);


	cv::Mat cur_fill(height,1,CV_8UC1,cv::Scalar::all(Disp_K));
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			if(Occ_map.data[j*width+i]==255)
				fill_left.data[j*width+i] = cur_fill.data[j];
			else
				cur_fill.data[j] = fill_left.data[j*width+i];
		}
	}

	cur_fill.setTo(cv::Scalar::all(Disp_K));
	for(int i=width-1;i>-1;i--){
		for(int j=0;j<height;j++){
			if(Occ_map.data[j*width+i]==255)
				fill_right.data[j*width+i] = cur_fill.data[j];
			else
				cur_fill.data[j] = fill_right.data[j*width+i];
		}
	}

	cv::Mat disp_view(height,width,CV_8UC1,cv::Scalar::all(0));

	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			Disp_map.data[j*width+i] = min(fill_left.data[j*width+i],fill_right.data[j*width+i]);
			disp_view.data[j*width+i] = scale_K * Disp_map.data[j*width+i];
		}
	}


	cv::namedWindow("disp",0);
	cv::imshow("disp",disp_view);
	cv::waitKey();


}


cv::Mat filtermask(cv::Mat & imL, int x, int y,int radius, double gamma_c,double gamma_p,int l, int r, int t, int b){
	cv::Mat mask;





/////////////////////////////color_diff////////////////////////////////////
	cv::Mat center_pix(b-t+1,r-l+1,CV_64FC3,cv::Scalar(((double*)imL.data)[3*(y*imL.cols+x)],((double*)imL.data)[3*(y*imL.cols+x)+1],((double*)imL.data)[3*(y*imL.cols+x)+2]));
	cv::Mat sub_mat = imL(cv::Rect(l,t,r-l+1,b-t+1));
	cv::Mat patch_win_color;
	sub_mat.copyTo(patch_win_color);





	cv::Mat color_diff_patch = center_pix - patch_win_color;








	color_diff_patch = color_diff_patch.mul(color_diff_patch);
	std::vector<cv::Mat> splited_c(3);

	cv::split(color_diff_patch,splited_c);

	color_diff_patch = splited_c[0]+splited_c[1]+splited_c[2];




	cv::pow(color_diff_patch,0.5,color_diff_patch);






////////////////////////////////////////////////////////////end_color_diff////////////////////////////
////////////////////////////////////////////////////////////sptial_diff/////////////////////////////////


	cv::Mat cor_x;
	cv::Mat cor_y;
	cor_x.create(b-t+1,r-l+1,CV_64F);
	cor_y.create(b-t+1,r-l+1,CV_64F);

	for(int c_x = l;c_x<=r;c_x++){
		cv::Mat sub_mat = cor_x(cv::Rect(c_x-l,0,1,b-t+1));
		sub_mat.setTo((cv::Scalar((c_x-x)*(c_x-x))));
	}

	for(int c_y = t;c_y<=b;c_y++){
		cv::Mat sub_mat = cor_y(cv::Rect(0,c_y-t,r-l+1,1));
		sub_mat.setTo((cv::Scalar((c_y-y)*(c_y-y))));
	}


	cv::Mat s_diff = cor_x+cor_y;
	cv::pow(s_diff,0.5,s_diff);




///////////////////////////////////////////////////////////cal_weight////////////////////////////////////


	s_diff.convertTo(s_diff,CV_64F,-1.0/(gamma_p*gamma_p));
	cv::exp(s_diff,s_diff);


	color_diff_patch.convertTo(color_diff_patch,CV_64F,-1.0/(gamma_c*gamma_c));
	cv::exp(color_diff_patch,color_diff_patch);


	mask = s_diff.mul(color_diff_patch);


	return mask;









}


void WMF(cv::Mat & imL, cv::Mat & Disp_map, cv::Mat & Occ_map,double gamma_c,double gamma_d,double winsize){
	int width = imL.cols;
	int height = imL.rows;


	cv::Mat imL_F;
	
	cv::Mat smoothed_imL;
	cv::medianBlur(imL,smoothed_imL,3);




	smoothed_imL.convertTo(imL_F,CV_64FC3,1.0/255.0);

	cv::Mat Disp_out;
	Disp_map.convertTo(Disp_out,CV_8UC1,scale_K);


	for(int x =0;x<width;x++){
		for(int y=0;y<height;y++){
			if(Occ_map.data[y*width+x]==255||0){
				int index_[Disp_K];
				memset(index_,0,sizeof(int)*Disp_K);
				int radius = winsize/2;
				int l = max(0,x-radius);
				int r = min(imL.cols-1,x+radius);
				int t = max(0,y-radius);
				int b = min(y+radius,imL.rows-1);
				cv::Mat weight = filtermask(imL_F,x,y,radius,gamma_c,gamma_d,l,r,t,b);


				std::vector<st_Sparce_hist *> weighted_disp_hist;
				
				
				cv::Mat disp_patch = Disp_map(cv::Rect(l,t,r-l+1,b-t+1));



				double total_w = 0;

				for(int ii = 0;ii<disp_patch.cols;ii++){
					for(int jj=0;jj<disp_patch.rows;jj++){
						int disp_ii = disp_patch.data[jj*disp_patch.step1()+ii];


		



						if(index_[disp_ii]){
							weighted_disp_hist[index_[disp_ii]-1]->weight += ((double*)weight.data)[jj*disp_patch.cols+ii];
							total_w += ((double*)weight.data)[jj*disp_patch.cols+ii];
						}
						else{
							st_Sparce_hist * aa = new st_Sparce_hist;
							aa->disp = disp_ii;
							aa->weight = ((double*)weight.data)[jj*disp_patch.cols+ii];
							total_w += aa->weight;
							weighted_disp_hist.push_back(aa);
							index_[disp_ii] = weighted_disp_hist.size();
						}
					}
				}


				std::sort(weighted_disp_hist.begin(),weighted_disp_hist.end(),cmp);


				double count_w = 0;
				for(int k=0;k<weighted_disp_hist.size();k++){
					count_w += weighted_disp_hist[k]->weight;
					if(count_w>=total_w/2.0){
						Disp_out.data [y*width+x] = weighted_disp_hist[k]->disp*scale_K;
						break;
					}
				}

				for(int k=0;k<weighted_disp_hist.size();k++){
					free(weighted_disp_hist[k]);
				}


			}

		}


	}





	cv::namedWindow("111",0);
	cv::imwrite("d:/SGCF/teddy/teddy_final_CVW.png",Disp_out);
	cv::imshow("111",Disp_out);
	cv::waitKey();




	

	return;
	
















	std::vector<st_Sparce_hist *> tttt;
	for(int k=0;k<100;k++){
		st_Sparce_hist * aa = new st_Sparce_hist;
		aa->disp = 100-k;
		aa->weight = 1.0;
		tttt.push_back(aa);
	}

	
	std::sort(tttt.begin(),tttt.end(),cmp);
	for(int k=0;k<100;k++){
		free(tttt[k]);
	}

	int asdf=0;

}

