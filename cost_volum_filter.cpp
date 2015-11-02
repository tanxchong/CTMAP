#include"cost.h"
#include"Param.h"

void cv_cum(cv::Mat & src, cv::Mat & dst,int dir){
	int width = src.cols;
	int height = src.rows;
	dst.create(height,width,CV_64F);
	memset(dst.data,0,sizeof(double)*width*height);
	if(dir==1){
		for(int i=0;i<width;i++){
			((double*)dst.data)[i] = ((double *)src.data)[i];
			for(int j=1;j<height;j++){
				((double*)dst.data)[j*width+i] = ((double*)dst.data)[(j-1)*width+i] + ((double *)src.data)[j*width+i];
			}
		}
	}
	if(dir==0){
		for(int j=0;j<height;j++){
			((double*)dst.data)[j*width] = ((double *)src.data)[j*width];
			for(int i=1;i<width;i++){
				((double*)dst.data)[j*width+i] = ((double*)dst.data)[j*width+i-1] + ((double *)src.data)[j*width+i];
			}
		}
	}

}




void box_filter(cv::Mat & src, cv::Mat & dst_,int r){
	int width = src.cols;
	int height = src.rows;
	cv::Mat dst;
	dst.create(height,width,CV_64F);
	double * dst_ptr = (double*)dst.data;
	cv::Mat imCum;
	src.convertTo(src,CV_64F);
	cv_cum(src,imCum,1);
	double * imCum_ptr = (double *) imCum.data;
	if(r+1>height){
		for(int j=0;j<height;j++){
			for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(height-1)*width+i];
				}
		}
	}else{
		if(2*r+1>height){

			for(int j=0;j<height-r;j++){
				for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(j+r)*width+i];
				}
			}

			for(int j=height-r;j<r+1;j++){
				memcpy(dst_ptr+j*width,imCum_ptr+(height-1)*width,sizeof(double)*width);
			}
			for(int j=r+1;j<height;j++){
				for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(height-1)*width+i] - imCum_ptr [(j-r-1)*width+i];
				}
			}

		}
		else{
			for(int j=0;j<r+1;j++){
				for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(j+r)*width+i];
				}
			}

			for(int j=r+1;j<height-r;j++){
				for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(j+r)*width+i] - imCum_ptr [(j-r-1)*width+i];
				}
			}

			for(int j=height-r;j<height;j++){
				for(int i=0;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [(height-1)*width+i] - imCum_ptr [(j-r-1)*width+i];
				}
			}

		}

	
	}
	imCum.release();
	cv_cum(dst,imCum,0);
	imCum_ptr = (double *) imCum.data;


	if(r+1>width){
		for(int j=0;j<height;j++){
			for(int i=0;i<width;i++){
				dst_ptr[j*width+i] = imCum_ptr [j*width+width-1];
			}
		}
	}

	else{	


		if(2*r+1>width){

			for(int j=0;j<height;j++){
				for(int i=0;i<width-r;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+i+r];
				}
			}

			for(int j=0;j<height;j++){
				for(int i=width-r;i<r+1;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+width-1];
				}
			}

			for(int j=0;j<height;j++){
				for(int i=r+1;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+width-1] - imCum_ptr [j*width+i-r-1];
				}
			}

		}
		else{
			for(int j=0;j<height;j++){
				for(int i=0;i<r+1;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+i+r];
				}
			}

			for(int j=0;j<height;j++){
				for(int i=r+1;i<width-r;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+i+r] - imCum_ptr [j*width+i-r-1];
				}
			}

			for(int j=0;j<height;j++){
				for(int i=width-r;i<width;i++){
					dst_ptr[j*width+i] = imCum_ptr [j*width+width-1] - imCum_ptr [j*width+i-r-1];
				}
			}
		}
	}
	dst_.release();
	dst_ = dst;

}


void cal_constant_val(cv::Mat & I, int r,double eps, std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert)

{
	


	int width = I.cols;
	int height = I.rows;
	
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			Invert[j*width+i].create(3,3,CV_64F);
		}
	}

	
	I.convertTo(I,CV_64FC3);
	
	
	std::vector<cv::Mat> I_channels(3); 
	cv::split(I,I_channels);

	cv::Mat mask_N(height,width,CV_64F,cv::Scalar::all(1));
	cv::Mat N;
	box_filter(mask_N,N,r);

	// mean
	
	cv::Mat mean_I_rr(height,width,CV_64F);
	cv::Mat mean_I_gg(height,width,CV_64F);
	cv::Mat mean_I_bb(height,width,CV_64F);
	cv::Mat mean_I_rg(height,width,CV_64F);
	cv::Mat mean_I_rb(height,width,CV_64F);
	cv::Mat mean_I_gb(height,width,CV_64F);



// II
	cv::Mat II_rr(height,width,CV_64F);
	cv::Mat II_gg(height,width,CV_64F);
	cv::Mat II_bb(height,width,CV_64F);
	cv::Mat II_rb(height,width,CV_64F);
	cv::Mat II_gb(height,width,CV_64F);
	cv::Mat II_rg(height,width,CV_64F);



	cv::Mat var_I_rr(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat var_I_rg(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat var_I_rb(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat var_I_gg(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat var_I_gb(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat var_I_bb(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat mean_I_r(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat mean_I_g(height,width,CV_64F,cv::Scalar::all(0)); 
	cv::Mat mean_I_b(height,width,CV_64F,cv::Scalar::all(0)); 


	/*
mean_I_r = boxfilter(I(:, :, 1), r) ./ N;
mean_I_g = boxfilter(I(:, :, 2), r) ./ N;
mean_I_b = boxfilter(I(:, :, 3), r) ./ N;


*/

	box_filter(I_channels[0],mean_I_r,r);
	box_filter(I_channels[1],mean_I_g,r);
	box_filter(I_channels[2],mean_I_b,r);


	mean_I_r = mean_I_r/N;
	mean_I_g = mean_I_g/N;
	mean_I_b = mean_I_b/N;

/*
           rr, rg, rb
   Sigma = rg, gg, gb
           rb, gb, bb
*/

	

	////////////////////////////////////////////
/*
mean_I_r = boxfilter(I(:, :, 1), r) ./ N;
mean_I_g = boxfilter(I(:, :, 2), r) ./ N;
mean_I_b = boxfilter(I(:, :, 3), r) ./ N;


*/

	box_filter(I_channels[0],mean_I_r,r);
	box_filter(I_channels[1],mean_I_g,r);
	box_filter(I_channels[2],mean_I_b,r);


	mean_I_r = mean_I_r/N;
	mean_I_g = mean_I_g/N;
	mean_I_b = mean_I_b/N;
////////////////////////////////////////////////

/*

var_I_rr = boxfilter(I(:, :, 1).*I(:, :, 1), r) ./ N - mean_I_r .*  mean_I_r; 
var_I_rg = boxfilter(I(:, :, 1).*I(:, :, 2), r) ./ N - mean_I_r .*  mean_I_g; 
var_I_rb = boxfilter(I(:, :, 1).*I(:, :, 3), r) ./ N - mean_I_r .*  mean_I_b; 
var_I_gg = boxfilter(I(:, :, 2).*I(:, :, 2), r) ./ N - mean_I_g .*  mean_I_g; 
var_I_gb = boxfilter(I(:, :, 2).*I(:, :, 3), r) ./ N - mean_I_g .*  mean_I_b; 
var_I_bb = boxfilter(I(:, :, 3).*I(:, :, 3), r) ./ N - mean_I_b .*  mean_I_b; 



*/



	cv::multiply(I_channels[0],I_channels[0],II_rr);
	cv::multiply(I_channels[1],I_channels[1],II_gg);
	cv::multiply(I_channels[2],I_channels[2],II_bb);
	cv::multiply(I_channels[0],I_channels[1],II_rg);
	cv::multiply(I_channels[0],I_channels[2],II_rb);
	cv::multiply(I_channels[1],I_channels[2],II_gb);

	box_filter(II_rr,II_rr,r);
	box_filter(II_gg,II_gg,r);
	box_filter(II_bb,II_bb,r);
	box_filter(II_rg,II_rg,r);
	box_filter(II_rb,II_rb,r);
	box_filter(II_gb,II_gb,r);

	cv::divide(II_rr,N,var_I_rr);
	cv::divide(II_gg,N,var_I_gg);
	cv::divide(II_bb,N,var_I_bb);
	cv::divide(II_rg,N,var_I_rg);
	cv::divide(II_rb,N,var_I_rb);
	cv::divide(II_gb,N,var_I_gb);


	cv::multiply(mean_I_r,mean_I_r, mean_I_rr);
	cv::multiply(mean_I_g,mean_I_g, mean_I_gg);
	cv::multiply(mean_I_b,mean_I_b, mean_I_bb);
	cv::multiply(mean_I_r,mean_I_g, mean_I_rg);
	cv::multiply(mean_I_r,mean_I_b, mean_I_rb);
	cv::multiply(mean_I_g,mean_I_b, mean_I_gb);

	cv::subtract(var_I_rr,mean_I_rr,var_I_rr);
	cv::subtract(var_I_gg,mean_I_gg,var_I_gg);
	cv::subtract(var_I_bb,mean_I_bb,var_I_bb);
	cv::subtract(var_I_rg,mean_I_rg,var_I_rg);
	cv::subtract(var_I_rb,mean_I_rb,var_I_rb);
	cv::subtract(var_I_gb,mean_I_gb,var_I_gb);



	Var_I[0] = var_I_rr;
	Var_I[1] = var_I_rg;
	Var_I[2] = var_I_rb;
	Var_I[3] = var_I_gg;
	Var_I[4] = var_I_gb;
	Var_I[5] = var_I_bb;
	Var_I[6] = mean_I_r;
	Var_I[7] = mean_I_g;
	Var_I[8] = mean_I_b;

	cv::Mat sigma(3,3,CV_64F,cv::Scalar::all(0));

	cv::Mat cov_Ip(1,3,CV_64F,cv::Scalar::all(0));
	double * cov_Ip_ptr = (double *) cov_Ip.data;



	double * sigma_ptr = (double *) sigma.data;

	double * var_I_rr_ptr = (double *) var_I_rr.data;
	double * var_I_gg_ptr = (double *) var_I_gg.data;
	double * var_I_bb_ptr = (double *) var_I_bb.data;
	double * var_I_rg_ptr = (double *) var_I_rg.data;
	double * var_I_rb_ptr = (double *) var_I_rb.data;
	double * var_I_gb_ptr = (double *) var_I_gb.data;



	cv::Mat rr_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat gg_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat bb_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat rg_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat rb_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat gb_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat r_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat g_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat b_box(height,width,CV_64F,cv::Scalar::all(0));
	cv::Mat constant_box(height,width,CV_64F,cv::Scalar::all(0));






	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			sigma_ptr[0] = var_I_rr_ptr[j*width+i]+eps;
			sigma_ptr[1] = var_I_rg_ptr[j*width+i];
			sigma_ptr[2] = var_I_rb_ptr[j*width+i];
			sigma_ptr[3] = var_I_rg_ptr[j*width+i];
			sigma_ptr[4] = var_I_gg_ptr[j*width+i]+eps;
			sigma_ptr[5] = var_I_gb_ptr[j*width+i];
			sigma_ptr[6] = var_I_rb_ptr[j*width+i];
			sigma_ptr[7] = var_I_gb_ptr[j*width+i];
			sigma_ptr[8] = var_I_bb_ptr[j*width+i]+eps;

			cv::Mat sigma_inv = sigma.inv();
			sigma_inv.copyTo(Invert[j*width+i]);

			cv::Mat miu;
			miu.create(3,1,CV_64F);
			((double*)miu.data)[0] = ((double*)mean_I_r.data)[j*width+i];
			((double*)miu.data)[1] =((double*)mean_I_g.data)[j*width+i];
			((double*)miu.data)[2] =((double*)mean_I_b.data)[j*width+i];
			cv::Mat sigma_inv_miu = sigma_inv*miu;
			cv::Mat c = (miu.t()*sigma_inv_miu);


			((double*)rr_box.data)[j*width+i] = ((double*)sigma_inv.data)[0];
			((double*)gg_box.data)[j*width+i] = ((double*)sigma_inv.data)[4];
			((double*)bb_box.data)[j*width+i] = ((double*)sigma_inv.data)[8];
			((double*)rg_box.data)[j*width+i] = 2*((double*)sigma_inv.data)[1];
			((double*)rb_box.data)[j*width+i] = 2*((double*)sigma_inv.data)[2];
			((double*)gb_box.data)[j*width+i] = 2*((double*)sigma_inv.data)[5];

			((double*)r_box.data)[j*width+i] = -2.0*((double*)sigma_inv_miu.data)[0];
			((double*)g_box.data)[j*width+i] = -2.0*((double*)sigma_inv_miu.data)[1];
			((double*)b_box.data)[j*width+i] = -2.0*((double*)sigma_inv_miu.data)[2];
			
			((double*)constant_box.data)[j*width+i] = ((double*)c.data)[0];

			

		}
	}

	box_filter(rr_box,rr_box,r);
	box_filter(gg_box,gg_box,r);
	box_filter(bb_box,bb_box,r);
	box_filter(rg_box,rg_box,r);
	box_filter(rb_box,rb_box,r);
	box_filter(gb_box,gb_box,r);
	box_filter(r_box,r_box,r);
	box_filter(g_box,g_box,r);
	box_filter(b_box,b_box,r);
	box_filter(constant_box,constant_box,r);

	cv::Mat w_p;
	w_p.create(height,width,CV_64F);



	for(int i =0;i<width;i++){
			for(int j=0;j<height;j++){

				 double tt = (((double*)rr_box.data)[j*width+i]*((double*)I_channels[0].data)[j*width+i]*((double*)I_channels[0].data)[j*width+i]
											+((double*)gg_box.data)[j*width+i]*((double*)I_channels[1].data)[j*width+i]*((double*)I_channels[1].data)[j*width+i]
											+((double*)bb_box.data)[j*width+i]*((double*)I_channels[2].data)[j*width+i]*((double*)I_channels[2].data)[j*width+i]
											+((double*)rg_box.data)[j*width+i]*((double*)I_channels[0].data)[j*width+i]*((double*)I_channels[1].data)[j*width+i]
											+((double*)rb_box.data)[j*width+i]*((double*)I_channels[0].data)[j*width+i]*((double*)I_channels[2].data)[j*width+i]
											+((double*)gb_box.data)[j*width+i]*((double*)I_channels[1].data)[j*width+i]*((double*)I_channels[2].data)[j*width+i]
											+((double*)r_box.data)[j*width+i]*((double*)I_channels[0].data)[j*width+i]
											+((double*)g_box.data)[j*width+i]*((double*)I_channels[1].data)[j*width+i]
											+((double*)b_box.data)[j*width+i]*((double*)I_channels[2].data)[j*width+i]
											+((double*)constant_box.data)[j*width+i]+((double*)N.data)[j*width+i])/((double*)N.data)[j*width+i];
					((double*)w_p.data)[j*width+i] = tt;
			}
		}




	Var_I[9] = w_p;




///////////////////////////////////////////end constant////////////////////////////////
}

void Guidedfilter_color(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I){
	int width = p.cols;
	int height = p.rows;
	
	p.convertTo(p,CV_64F);
	I.convertTo(I,CV_64FC3);
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			if(abs(p.ptr<double>(0)[j*width+i])>10000000000.0)
				int asdf=0;
		}
	}

	
	std::vector<cv::Mat> I_channels(3); 
	cv::split(I,I_channels);

	cv::Mat mask_N(height,width,CV_64F,cv::Scalar::all(1));
	cv::Mat N;
	box_filter(mask_N,N,r);










	cv::Mat mean_Ip_r(height,width,CV_64F);
	cv::Mat mean_Ip_g(height,width,CV_64F);
	cv::Mat mean_Ip_b(height,width,CV_64F);
	cv::Mat mean_p;
//covariance 
	cv::Mat cov_Ip_r(height,width,CV_64F);
	cv::Mat cov_Ip_g(height,width,CV_64F);
	cv::Mat cov_Ip_b(height,width,CV_64F);


/////////////////////// calculate constant vals////////////////////////////////




	cv::Mat var_I_rr = Var_I[0];
	cv::Mat var_I_rg = Var_I[1];
	cv::Mat var_I_rb = Var_I[2];
	cv::Mat var_I_gg = Var_I[3];
	cv::Mat var_I_gb = Var_I[4];
	cv::Mat var_I_bb = Var_I[5];

// mean
	cv::Mat mean_I_r = Var_I[6];
	cv::Mat mean_I_g = Var_I[7];
	cv::Mat mean_I_b = Var_I[8];;





	double * var_I_rr_ptr = (double *) var_I_rr.data;
	double * var_I_gg_ptr = (double *) var_I_gg.data;
	double * var_I_bb_ptr = (double *) var_I_bb.data;
	double * var_I_rg_ptr = (double *) var_I_rg.data;
	double * var_I_rb_ptr = (double *) var_I_rb.data;
	double * var_I_gb_ptr = (double *) var_I_gb.data;





//mean_p = boxfilter(p, r) ./ N;
	box_filter(p,mean_p,r);
	mean_p = mean_p/N;

	


	


/////////////////////////////////////////////////////


/*
mean_Ip_r = boxfilter(I(:, :, 1).*p, r) ./ N;
mean_Ip_g = boxfilter(I(:, :, 2).*p, r) ./ N;
mean_Ip_b = boxfilter(I(:, :, 3).*p, r) ./ N;


*/
	cv::multiply(I_channels[0],p,mean_Ip_r);
	cv::multiply(I_channels[1],p,mean_Ip_g);
	cv::multiply(I_channels[2],p,mean_Ip_b);

	box_filter(mean_Ip_r,mean_Ip_r,r);
	box_filter(mean_Ip_g,mean_Ip_g,r);
	box_filter(mean_Ip_b,mean_Ip_b,r);

	cv::divide(mean_Ip_r,N,mean_Ip_r);
	cv::divide(mean_Ip_g,N,mean_Ip_g);
	cv::divide(mean_Ip_b,N,mean_Ip_b);
	
//covariance

/*

cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;


*/
	cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
	cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);





////////////////////////////////////////////////////////

	cv::Mat a(height,width,CV_64FC3,cv::Scalar::all(0));
	cv::Mat b(height,width,CV_64F,cv::Scalar::all(0));

	double * b_ptr = (double *)b.data;
	
	
	
	cv::Mat sigma(3,3,CV_64F,cv::Scalar::all(0));

	cv::Mat cov_Ip(1,3,CV_64F,cv::Scalar::all(0));
	double * cov_Ip_ptr = (double *) cov_Ip.data;



	double * sigma_ptr = (double *) sigma.data;
	double * a_ptr = (double *) a.data;
	double * cov_Ip_r_ptr = (double *)cov_Ip_r.data;
	double * cov_Ip_g_ptr = (double *)cov_Ip_g.data;
	double * cov_Ip_b_ptr = (double *)cov_Ip_b.data;


	cv::Mat aa;
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			sigma_ptr[0] = var_I_rr_ptr[j*width+i]+eps;
			sigma_ptr[1] = var_I_rg_ptr[j*width+i];
			sigma_ptr[2] = var_I_rb_ptr[j*width+i];
			sigma_ptr[3] = var_I_rg_ptr[j*width+i];
			sigma_ptr[4] = var_I_gg_ptr[j*width+i]+eps;
			sigma_ptr[5] = var_I_gb_ptr[j*width+i];
			sigma_ptr[6] = var_I_rb_ptr[j*width+i];
			sigma_ptr[7] = var_I_gb_ptr[j*width+i];
			sigma_ptr[8] = var_I_bb_ptr[j*width+i]+eps;

			cov_Ip_ptr[0] = cov_Ip_r_ptr[j*width+i];
			cov_Ip_ptr[1] = cov_Ip_g_ptr[j*width+i];
			cov_Ip_ptr[2] = cov_Ip_b_ptr[j*width+i];

			
			aa = cov_Ip*(sigma.inv());

			double k1 = a_ptr[(j*width+i)*3] = ((double*)aa.data)[0];
			double k2 = a_ptr[(j*width+i)*3+1] = ((double*)aa.data)[1];
			double k3 = a_ptr[(j*width+i)*3+2] = ((double*)aa.data)[2];

			b_ptr[j*width+i] = ((double*)mean_p.data)[j*width+i] - ((double*)aa.data)[0]*((double*)mean_I_r.data)[j*width+i]
							- ((double*)aa.data)[1]*((double*)mean_I_g.data)[j*width+i]
							- ((double*)aa.data)[2]*((double*)mean_I_b.data)[j*width+i];

		}
	}



	std::vector<cv::Mat> a_channels(3); 
	cv::split(a,a_channels);


	box_filter(a_channels[0],a_channels[0],r);
	box_filter(a_channels[1],a_channels[1],r);
	box_filter(a_channels[2],a_channels[2],r);
	box_filter(b,b,r);


	q = (a_channels[0].mul(I_channels[0])+a_channels[1].mul(I_channels[1])+a_channels[2].mul(I_channels[2])+b)/N;


	I_channels[0].release();
	I_channels[1].release();
	I_channels[2].release();

}

void Guidedfilter_sum(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert_mats){
	int width = p.cols;
	int height = p.rows;
	
	p.convertTo(p,CV_64F);
	I.convertTo(I,CV_64FC3);


	
	std::vector<cv::Mat> I_channels(3); 
	cv::split(I,I_channels);

	cv::Mat mask_N(height,width,CV_64F,cv::Scalar::all(1));
	cv::Mat N;
	box_filter(mask_N,N,r);










	cv::Mat mean_Ip_r(height,width,CV_64F);
	cv::Mat mean_Ip_g(height,width,CV_64F);
	cv::Mat mean_Ip_b(height,width,CV_64F);
	cv::Mat mean_p;
//covariance 
	cv::Mat cov_Ip_r(height,width,CV_64F);
	cv::Mat cov_Ip_g(height,width,CV_64F);
	cv::Mat cov_Ip_b(height,width,CV_64F);


/////////////////////// calculate constant vals////////////////////////////////




	cv::Mat var_I_rr = Var_I[0];
	cv::Mat var_I_rg = Var_I[1];
	cv::Mat var_I_rb = Var_I[2];
	cv::Mat var_I_gg = Var_I[3];
	cv::Mat var_I_gb = Var_I[4];
	cv::Mat var_I_bb = Var_I[5];

// mean
	cv::Mat mean_I_r = Var_I[6];
	cv::Mat mean_I_g = Var_I[7];
	cv::Mat mean_I_b = Var_I[8];
	cv::Mat p_w = Var_I[9];





	double * var_I_rr_ptr = (double *) var_I_rr.data;
	double * var_I_gg_ptr = (double *) var_I_gg.data;
	double * var_I_bb_ptr = (double *) var_I_bb.data;
	double * var_I_rg_ptr = (double *) var_I_rg.data;
	double * var_I_rb_ptr = (double *) var_I_rb.data;
	double * var_I_gb_ptr = (double *) var_I_gb.data;





//mean_p = boxfilter(p, r) ./ N;
	box_filter(p,mean_p,r);
	mean_p = mean_p/N;

	


	


/////////////////////////////////////////////////////


/*
mean_Ip_r = boxfilter(I(:, :, 1).*p, r) ./ N;
mean_Ip_g = boxfilter(I(:, :, 2).*p, r) ./ N;
mean_Ip_b = boxfilter(I(:, :, 3).*p, r) ./ N;


*/
	cv::multiply(I_channels[0],p,mean_Ip_r);
	cv::multiply(I_channels[1],p,mean_Ip_g);
	cv::multiply(I_channels[2],p,mean_Ip_b);

	box_filter(mean_Ip_r,mean_Ip_r,r);
	box_filter(mean_Ip_g,mean_Ip_g,r);
	box_filter(mean_Ip_b,mean_Ip_b,r);

	cv::divide(mean_Ip_r,N,mean_Ip_r);
	cv::divide(mean_Ip_g,N,mean_Ip_g);
	cv::divide(mean_Ip_b,N,mean_Ip_b);
	
//covariance

/*

cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;


*/
	cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
	cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);



	

////////////////////////////////////////////////////////

	cv::Mat a(height,width,CV_64FC3,cv::Scalar::all(0));
	cv::Mat b(height,width,CV_64F,cv::Scalar::all(0));










	double * b_ptr = (double *)b.data;
	
	
	


	cv::Mat cov_Ip(1,3,CV_64F,cv::Scalar::all(0));
	double * cov_Ip_ptr = (double *) cov_Ip.data;




	double * a_ptr = (double *) a.data;
	double * cov_Ip_r_ptr = (double *)cov_Ip_r.data;
	double * cov_Ip_g_ptr = (double *)cov_Ip_g.data;
	double * cov_Ip_b_ptr = (double *)cov_Ip_b.data;


	cv::Mat aa;
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){


			cov_Ip_ptr[0] = cov_Ip_r_ptr[j*width+i];
			cov_Ip_ptr[1] = cov_Ip_g_ptr[j*width+i];
			cov_Ip_ptr[2] = cov_Ip_b_ptr[j*width+i];

			cv::Mat sigma_inv = Invert_mats[j*width+i];

			
			aa = cov_Ip*sigma_inv;


			

			double k1 = a_ptr[(j*width+i)*3] = ((double*)aa.data)[0];
			double k2 = a_ptr[(j*width+i)*3+1] = ((double*)aa.data)[1];
			double k3 = a_ptr[(j*width+i)*3+2] =((double*)aa.data)[2];



			b_ptr[j*width+i] = ((double*)mean_p.data)[j*width+i] - a_ptr[(j*width+i)*3]*((double*)mean_I_r.data)[j*width+i]
							- a_ptr[(j*width+i)*3+1]*((double*)mean_I_g.data)[j*width+i]
							- a_ptr[(j*width+i)*3+2]*((double*)mean_I_b.data)[j*width+i];


		}
	}




	




	std::vector<cv::Mat> a_channels(3); 
	cv::split(a,a_channels);


	box_filter(a_channels[0],a_channels[0],r);
	box_filter(a_channels[1],a_channels[1],r);
	box_filter(a_channels[2],a_channels[2],r);
	box_filter(b,b,r);


	cv::Mat msg_p;
	msg_p.create(height,width,CV_64F);
	


	for(int i =0;i<width;i++){
			for(int j=0;j<height;j++){
				double w_p = ((double*)p_w.data)[j*width+i];
				double pp = ((double*)p.data)[j*width+i];
				((double*)msg_p.data)[j*width+i] = w_p*pp;

			}
		}
	
	q = (a_channels[0].mul(I_channels[0])+a_channels[1].mul(I_channels[1])+a_channels[2].mul(I_channels[2])+b);


	if(!Relaxation){
		q = q-msg_p;
	}

	




	I_channels[0].release();
	I_channels[1].release();
	I_channels[2].release();

}


void Guidedfilter_median(cv::Mat & I, cv::Mat & p, cv::Mat & q, int r, double eps,std::vector<cv::Mat> & Var_I,std::vector<cv::Mat> & Invert_mats){
	int width = p.cols;
	int height = p.rows;
	
	p.convertTo(p,CV_64F);
	I.convertTo(I,CV_64FC3);


	
	std::vector<cv::Mat> I_channels(3); 
	cv::split(I,I_channels);

	cv::Mat mask_N(height,width,CV_64F,cv::Scalar::all(1));
	cv::Mat N;
	box_filter(mask_N,N,r);










	cv::Mat mean_Ip_r(height,width,CV_64F);
	cv::Mat mean_Ip_g(height,width,CV_64F);
	cv::Mat mean_Ip_b(height,width,CV_64F);
	cv::Mat mean_p;
//covariance 
	cv::Mat cov_Ip_r(height,width,CV_64F);
	cv::Mat cov_Ip_g(height,width,CV_64F);
	cv::Mat cov_Ip_b(height,width,CV_64F);


/////////////////////// calculate constant vals////////////////////////////////




	cv::Mat var_I_rr = Var_I[0];
	cv::Mat var_I_rg = Var_I[1];
	cv::Mat var_I_rb = Var_I[2];
	cv::Mat var_I_gg = Var_I[3];
	cv::Mat var_I_gb = Var_I[4];
	cv::Mat var_I_bb = Var_I[5];

// mean
	cv::Mat mean_I_r = Var_I[6];
	cv::Mat mean_I_g = Var_I[7];
	cv::Mat mean_I_b = Var_I[8];
	cv::Mat p_w = Var_I[9];





	double * var_I_rr_ptr = (double *) var_I_rr.data;
	double * var_I_gg_ptr = (double *) var_I_gg.data;
	double * var_I_bb_ptr = (double *) var_I_bb.data;
	double * var_I_rg_ptr = (double *) var_I_rg.data;
	double * var_I_rb_ptr = (double *) var_I_rb.data;
	double * var_I_gb_ptr = (double *) var_I_gb.data;





//mean_p = boxfilter(p, r) ./ N;
	box_filter(p,mean_p,r);
	mean_p = mean_p/N;

	


	


/////////////////////////////////////////////////////


/*
mean_Ip_r = boxfilter(I(:, :, 1).*p, r) ./ N;
mean_Ip_g = boxfilter(I(:, :, 2).*p, r) ./ N;
mean_Ip_b = boxfilter(I(:, :, 3).*p, r) ./ N;


*/
	cv::multiply(I_channels[0],p,mean_Ip_r);
	cv::multiply(I_channels[1],p,mean_Ip_g);
	cv::multiply(I_channels[2],p,mean_Ip_b);

	box_filter(mean_Ip_r,mean_Ip_r,r);
	box_filter(mean_Ip_g,mean_Ip_g,r);
	box_filter(mean_Ip_b,mean_Ip_b,r);

	cv::divide(mean_Ip_r,N,mean_Ip_r);
	cv::divide(mean_Ip_g,N,mean_Ip_g);
	cv::divide(mean_Ip_b,N,mean_Ip_b);
	
//covariance

/*

cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;


*/
	cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
	cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);



	

////////////////////////////////////////////////////////

	cv::Mat a(height,width,CV_64FC3,cv::Scalar::all(0));
	cv::Mat b(height,width,CV_64F,cv::Scalar::all(0));










	double * b_ptr = (double *)b.data;
	
	
	


	cv::Mat cov_Ip(1,3,CV_64F,cv::Scalar::all(0));
	double * cov_Ip_ptr = (double *) cov_Ip.data;




	double * a_ptr = (double *) a.data;
	double * cov_Ip_r_ptr = (double *)cov_Ip_r.data;
	double * cov_Ip_g_ptr = (double *)cov_Ip_g.data;
	double * cov_Ip_b_ptr = (double *)cov_Ip_b.data;


	cv::Mat aa;
	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){


			cov_Ip_ptr[0] = cov_Ip_r_ptr[j*width+i];
			cov_Ip_ptr[1] = cov_Ip_g_ptr[j*width+i];
			cov_Ip_ptr[2] = cov_Ip_b_ptr[j*width+i];

			cv::Mat sigma_inv = Invert_mats[j*width+i];





			
			aa = cov_Ip*sigma_inv;
			


			

			double k1 = a_ptr[(j*width+i)*3] = max(-1.0*Large_T,min(Large_T, ((double*)aa.data)[0]));
			double k2 = a_ptr[(j*width+i)*3+1] = max(-1.0*Large_T,min(Large_T, ((double*)aa.data)[1]));
			double k3 = a_ptr[(j*width+i)*3+2] = max(-1.0*Large_T,min(Large_T, ((double*)aa.data)[2]));



			b_ptr[j*width+i] = max(-1.0*Large_T,min(Large_T,((double*)mean_p.data)[j*width+i] - a_ptr[(j*width+i)*3]*((double*)mean_I_r.data)[j*width+i]
							- a_ptr[(j*width+i)*3+1]*((double*)mean_I_g.data)[j*width+i]
							- a_ptr[(j*width+i)*3+2]*((double*)mean_I_b.data)[j*width+i]));

		}
	}







	std::vector<cv::Mat> a_channels(3); 
	cv::split(a,a_channels);

	cv::Mat seed_N;
	int seed_r = max(1,r/Grid_num);


	box_filter(mask_N,seed_N,seed_r);

	box_filter(a_channels[0],a_channels[0],seed_r);
	box_filter(a_channels[1],a_channels[1],seed_r);
	box_filter(a_channels[2],a_channels[2],seed_r);
	box_filter(b,b,seed_r);

	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){


			int bound_u = max(0,j-r+seed_r);
			int bound_d = min(height-1,j+r-seed_r);
			int bound_l = max(0,i-r+seed_r);
			int bound_r = min(width-1,i+r-seed_r);
			std::vector<double> values;

			for(int v_step = bound_u;v_step<=bound_d;v_step+=2*seed_r){
				for(int h_step = bound_l;h_step<=bound_r;h_step+=2*seed_r){
					double q_i =
							(((double*)(a_channels[0].data))[v_step*width+h_step]*((double*)I_channels[0].data)[j*width+i]
							+ ((double*)(a_channels[1].data))[v_step*width+h_step]*((double*)I_channels[1].data)[j*width+i]
							+ ((double*)(a_channels[2].data))[v_step*width+h_step]*((double*)I_channels[2].data)[j*width+i]
							+ ((double*)b.data)[v_step*width+h_step])/((double*)seed_N.data)[v_step*width+h_step];
							values.push_back(q_i);



				}
			}
			std::sort(values.begin(),values.end());

			//////////////////////////////////////convolution///////////////////////////////////////////



			q.ptr<double>(0)[width*j+i] =values[values.size()/2];


		}
	}



	q = q.mul(N);

	I_channels[0].release();
	I_channels[1].release();
	I_channels[2].release();

};