#include<memory.h>
#include"Param.h"
#include <iostream>


void V_vol_regulation_pottos(double * V, double *cost, int width, int height, int level, double p1, double p2){
	int step_0 = width*height;
	memcpy(V,cost,sizeof(double)*step_0*level);
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			double min_V = min(V[width*y+x],V[step_0*(level-1)+width*y+x]);
			if(cost[width*y+x]>cost[step_0+width*y+x]+p1)
				V[width*y+x] = cost[step_0+width*y+x]+p1;
			if(cost[step_0*(level-1)+width*y+x]>cost[step_0*(level-2)+width*y+x]+p1){
				V[step_0*(level-1)+width*y+x] = cost[step_0*(level-2)+width*y+x]+p1;
			}
			for(int l=1;l<level-1;l++){																//forward tracking

				if(V[step_0*(l)+width*y+x]>cost[step_0*(l+1)+width*y+x]+p1){							//use p1 to regulate
					V[step_0*(l)+width*y+x] = cost[step_0*(l+1)+width*y+x]+p1;
				}
				if(V[step_0*(l)+width*y+x]>cost[step_0*(l-1)+width*y+x]+p1){
					V[step_0*(l)+width*y+x] = cost[step_0*(l-1)+width*y+x]+p1;
				}
				if(cost[step_0*l+width*y+x]<min_V){													//find minimum
					min_V = cost[step_0*l+width*y+x];
				}
			}


			for(int l = 0;l<level;l++){															
				if(V[step_0*(l)+width*y+x]>min_V+p2){
					V[step_0*(l)+width*y+x] = min_V+p2;
				}
			}

			for(int l=0;l<level;l++){
				V[step_0*(l)+width*y+x] -= min_V;														//normlization
			}
		}
	}
}

void V_vol_regulation_linear(double * V, double *cost, int width, int height, int level, float weight, float trunc){

	int step_0 = width*height;
	memcpy(V,cost,sizeof(double)*step_0*level);
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){

			float minimum = V[width*y+x];
			for (int q = 1; q < level; q++) {															// tracking forward
				float prev = V[step_0*(q-1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
				if(minimum>V[step_0*q+width*y+x])
					minimum=V[step_0*q+width*y+x];
			  }
			  for (int q = level-2; q >= 0; q--) {														//tracking backward
				float prev = V[step_0*(q+1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
			  }

			  minimum += trunc;
//normalize 
			  float mini_val = V[width*y+x]; 
			  for(int l = 0;l<level;l++){
				  if(mini_val>V[step_0*l+width*y+x])
					  mini_val = V[step_0*l+width*y+x];
				  if(minimum<V[step_0*l+width*y+x])
					  V[step_0*l+width*y+x] = minimum;
			  }

			  for(int l = 0;l<level;l++){
				   V[step_0*l+width*y+x] -= mini_val;
			  }

		}
	}
}

void V_vol_sumproduct_linear(double * V, double *cost, int width, int height, int level, float weight, float trunc){

	int step_0 = width*height;
	memcpy(V,cost,sizeof(double)*step_0*level);
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){






			float minimum = V[width*y+x];
			for (int q = 1; q < level; q++) {															// tracking forward
				float prev = V[step_0*(q-1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
				if(minimum>V[step_0*q+width*y+x])
					minimum=V[step_0*q+width*y+x];
			  }
			  for (int q = level-2; q >= 0; q--) {														//tracking backward
				float prev = V[step_0*(q+1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
			  }

			  minimum += trunc;
//normalize 
			  float mini_val = V[width*y+x]; 
			  for(int l = 0;l<level;l++){
				  if(mini_val>V[step_0*l+width*y+x])
					  mini_val = V[step_0*l+width*y+x];
				  if(minimum<V[step_0*l+width*y+x])
					  V[step_0*l+width*y+x] = minimum;
			  }

			  for(int l = 0;l<level;l++){
				   V[step_0*l+width*y+x] -= mini_val;
			  }










		}
	}
}































void V_vol_regulation_send(double * V, double *cost, int width, int height, int level, float weight, float trunc){

	int step_0 = width*height;
	memcpy(V,cost,sizeof(double)*step_0*level);
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){

			float minimum = V[width*y+x];
			int minimum_pos = 0;
			for(int q = 1; q < level; q++){
				if(minimum>V[step_0*q+width*y+x]){
					minimum=V[step_0*q+width*y+x];
					minimum_pos = q;
				}
			}

			V[minimum_pos*step_0+width*y+x] = 0;


			for (int q = minimum_pos+1; q < level; q++) {															// tracking forward
				float prev = V[step_0*(q-1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
				if(minimum>V[step_0*q+width*y+x])
					minimum=V[step_0*q+width*y+x];
			  }

			 for (int q = minimum_pos-1; q >= 0; q--) {														//tracking backward
				float prev = V[step_0*(q+1)+width*y+x]+weight;
				if (prev < V[step_0*q+width*y+x])
				  V[step_0*(q)+width*y+x] = prev;
			  }

			 for(int l = 0;l<level;l++){
				  if(trunc<V[step_0*l+width*y+x])
					  V[step_0*l+width*y+x] = trunc;

			  }


		}
	}


}
;							//minimum convolution
void Sum_up_cost(double * V, double *cost, int width, int height, int level){
	int step_0 = width*height;
	for(int x=0;x<width;x++){
		for(int y=0;y<height;y++){
			for(int l=0;l<level;l++){
				V[step_0*(l)+width*y+x] += cost[step_0*(l)+width*y+x];
			}
		}
	}
}

