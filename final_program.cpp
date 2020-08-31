#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 4 //height
#define M 4 //width

clock_t stime,etime;

__global__ void call_to_cuda_for_padding_image(float *d_img,float *d_padd_img)  // for padding image onto global memory
{
	/* thread index calculation */

	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;

	int dst_col=col+1;
	int dst_row=row+1;
	int dst_index=dst_row*(M+2)+dst_col;
	
	if(row<M && col<N)
	{
		
		float pixel=d_img[row*M+col];

		if(dst_row==1)		//for 1st row of padded image matrix
		{
			d_padd_img[dst_col]=pixel;
		}
		if(dst_row==N)		//for last row of padded image matrix
		{
			d_padd_img[dst_index+M+2]=pixel;
		}
		if(dst_col==1)		// for 1st column of padded image matrix
		{
			d_padd_img[dst_index-1]=pixel;
		}
		if(dst_col==M)		//for last column of padded image matrix
		{
			d_padd_img[dst_index+1]=pixel;
		}
		
		

		//for corner pixels
		if((dst_row==1) && (dst_col==1)) // 1st row X 1st column pixel
		{
			d_padd_img[dst_col-1]=pixel;
		}
		if((dst_row==1) && (dst_col==M)) // 1st row X last column pixel
		{
			d_padd_img[dst_col+1]=pixel;
		}
		if((dst_row==N) && (dst_col==1)) // last row X 1st column pixel
		{
			d_padd_img[dst_index+M+1]=pixel;
		}
		if((dst_row==N) && (dst_col==M)) // last row X last column pixel
		{
			d_padd_img[dst_index+M+3]=pixel;
		}

		d_padd_img[dst_index]=pixel;	// for rmaining rows and columns of matrix

	}


}


__global__ void apply_mask(float * d_padd_img,float *d_result,float *d_result1,float *d_final_result)
{
	/* thread index calculation */
	
	int C=blockIdx.x*blockDim.x+threadIdx.x;
	int R= blockIdx.y*blockDim.y+threadIdx.y;

	int i,j,r,c;

__shared__ int mask[3][3];
mask[0][0]=-1;mask[0][1]=-2;mask[0][2]=-1;
mask[1][0]=0;mask[1][1]=0;mask[1][2]=0;
mask[2][0]=1;mask[2][1]=2;mask[2][2]=1;

__shared__ int mask2[3][3];
mask2[0][0]=-1;mask2[0][1]=0;mask2[0][2]=1;
mask2[1][0]=-2;mask2[1][1]=0;mask2[1][2]=2;
mask2[2][0]=-1;mask2[2][1]=0;mask2[2][2]=1;




__shared__ int d_cache_im[6][6];
	
	int s_col=blockIdx.x*blockDim.x+threadIdx.x+1;
	int s_row=blockIdx.y*blockDim.y+threadIdx.y+1;

	int im_index=s_row*(M+2)+s_col;
	
	if((s_col<=M) && (s_row<=N))
	{
		if(threadIdx.y==0)		// first row
		{
			d_cache_im[threadIdx.y][threadIdx.x+1]=d_padd_img[im_index-(M+2)];
		}

		if(threadIdx.y==(blockDim.y-1)) // last row
		{
			d_cache_im[threadIdx.y+1][threadIdx.x+1]=d_padd_img[im_index+(M+2)];
		}
		if(threadIdx.x==0)		// 1st column
		{
			d_cache_im[threadIdx.y+1][threadIdx.x]=d_padd_img[im_index-1];
			
		}
		if(threadIdx.x==(blockDim.x-1)) // last column
		{
			d_cache_im[threadIdx.y+1][threadIdx.x+1]=d_padd_img[im_index+1];
		}
		
		//load corner pixel

		if((threadIdx.x==0) && (threadIdx.y==0)) // 1st row X 1st column pixel
		{
			d_cache_im[threadIdx.y][threadIdx.x]=d_padd_img[im_index-(M+3)];
		}
		if(threadIdx.y==0 && threadIdx.x==(blockDim.x-1)) // 1st row X last column pixel
		{
			d_cache_im[threadIdx.y][threadIdx.x+2]=d_padd_img[im_index-(M+1)];
		}
		if(threadIdx.y==(blockDim.y-1) && threadIdx.x==0) // last row X 1st column pixel
		{
			d_cache_im[threadIdx.y+2][threadIdx.x]=d_padd_img[im_index+M+1];
			
		}
		if(threadIdx.y==(blockDim.y-1) && threadIdx.x==(blockDim.x-1)) // last row X last column pixel
		{
			d_cache_im[threadIdx.y+2][threadIdx.x+2]=d_padd_img[im_index+(M+3)];
			
		}

		d_cache_im[threadIdx.y+1][threadIdx.x+1]=d_padd_img[im_index]; // for rmaining rows and columns of matrix

	
	}
	__syncthreads();

r=threadIdx.y+1;
c=threadIdx.x+1;

if(r<=N && c<=M)
{
	float pixel=0.0,pixel1=0.0;

	for(i=-1;i<=1;i++)
	{
		for(j=-1;j<=1;j++)
		{
			pixel=pixel+d_cache_im[r+i][c+j]*mask[1+i][1+j]; // horizantal mask 
			pixel1=pixel1+d_cache_im[r+i][c+j]*mask2[1+i][1+j]; // vertical mask
		}
	}
	pixel=pixel/9.0;		//calc. avg. value
	d_result[(r-1)*M+(c-1)] = pixel;

	pixel1=pixel1/9.0;
	d_result1[(r-1)*M+(c-1)] = pixel1;

	d_final_result[(r-1)*M+(c-1)]=sqrt((pixel*pixel)+(pixel1*pixel1)); //calculate mean value and store in final matrix

}

}



int main()
{
/* VARIABLE DECLARATION */
	/* for input image matrix */
	float img[M][N]={{150,200,100,170},{100,120,200,110},{210,150,250,100},{50,210,120,150}};
	/* for stroing padded image */
	float padd_img[M+2][N+2]={0};

	float result[M][N]={0};			// to store matrix after applying horizantal mask
	float result1[M][N]={0};		// to store matrix after applying horizantal mask
	float final_result[M][N]={0};	// to store final result matrix after applying mask

	int i,j;		//for indexing


/* MEMORY ALLOCATION ON DEVICE */
	// poi
	float *d_img,*d_padd_img,*d_result1,*d_result,*d_final_result;
	/*
		d_img :- for allocating image matrix on device
		d_padd_img :- for storing padded image 
		d_result :- for storing reslutant matrix after applying hoizantal mask
		d_result1 :- for storing reslutant matrix after applying vertical mask
		d_final_result :- for storing after calculating mean gradient of resultant pixels
	*/

/* SPACE ALLOCATION ON DEVICE MEMROY WITH SIZE = IMAGE MATRIX */
	
	cudaMalloc((void **)&d_img,M*N*sizeof(float));				  // allocating device memory of size = image size
	cudaMalloc((void **)&d_padd_img,(M+2)*(N+2)*sizeof(float));  // allocating device memory for padding image
	cudaMalloc((void **)&d_result,M*N*sizeof(float));			// for resultant matrix after applying horizantal mask
	cudaMalloc((void **)&d_result1,M*N*sizeof(float));		   // for resultant matrix after applying vertical mask
	cudaMalloc((void **)&d_final_result,M*N*sizeof(float));	  // for final matrix 

/* COPYING THE HOST MEMORY TO DEVICE MEMORY */

	cudaMemcpy(d_img,&img,M*N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_padd_img,&padd_img,(M+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);
	
	
/* THREAD CONFIGURATION*/

	int x,y;
	x=(M/2)+1; // NO. OF BLOCKS ON x-DIRECTION
	y=(N/2)+1; // NO. OF BLOCKS ON y-DIRECTION

	dim3 NGRID(x,y); //GRID OF BLOCKS [x][y]
	dim3 NBLOCK(4,4); //BLOCK OF THREADS [4][4]
	

/* CALL TO CUDA FUNCTION'S */
	stime=clock();

	call_to_cuda_for_padding_image<<<NGRID,NBLOCK>>>(d_img,d_padd_img);  
	/*
		call_to_cuda_for_padding_image :- cuda function name

		NGRID :- no. of grids for execution
		NBLOCK :- no. of blocks per grid for execution

		args :  1) d_img :- address of device memory where image is loaded.
				2) d_padd_img :- address of device memory where padded image is to be kept  (result of funtion)

	*/

	apply_mask<<<NGRID,NBLOCK>>>(d_padd_img,d_result,d_result1,d_final_result);
	/*
		d_padd_img :- device memory address where our padded image is stored
		d_result :-	  device memory address for storing matrix after applying horizantal mask
		d_result1 :-  device memory address for storing matrix after applying vertical mask
	*/

	etime=clock();
/* MEMORY TRANSFORMATION FROM DEVICE TO HOST */	

	cudaMemcpy(&padd_img,d_padd_img,(M+2)*(N+2)*sizeof(float),cudaMemcpyDeviceToHost); // copying padded image to host memory
	cudaMemcpy(&result,d_result,(M)*(N)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&result1,d_result1,(M)*(N)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&final_result,d_final_result,(M)*(N)*sizeof(float),cudaMemcpyDeviceToHost);

/* PRINTING THE RESULTS */

/* TO DISPLAY INPUT IMAGE MATRIX */
	printf("++++++++++++++++ INPUT IMAGE MATRIX +++++++++++++++++++\n\n");
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("  %.2f",img[i][j]);
		}
		printf("\n");
	}

	printf("\n++++++++++++++++ PADDED IMAGE +++++++++++++++++++\n\n");
	for(i=0;i<M+2;i++)
	{
		for(j=0;j<N+2;j++)
		{
			printf("%.2f\t",padd_img[i][j]);
			
		}
		printf("\n");
	}
		
	printf("\n+++++++++ AFTER APPLYING HORIZONTAL MASK ++++++++++++\n\n");
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("  %.2f",result[i][j]);
		}
		printf("\n");
	}

	printf("\n+++++++++ AFTER APPLYING VERTICAL MASK ++++++++++++\n\n");
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("  %.2f",result1[i][j]);
		}
		printf("\n");
	}

	printf("\n+++++++++ final result ++++++++++++\n\n");
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("  %.2f",final_result[i][j]);
		}
		printf("\n");
	}



/*FREEING THE SPACE ALLOCATED ON DEVICE MEMORY*/
	cudaFree(d_img);
	cudaFree(d_padd_img);
	cudaFree(d_result);
    cudaFree(d_result1);
	cudaFree(d_final_result);

//END

	printf("\n\n");
	printf("stime :- %d \t etime :- %d",stime,etime);
	printf("\n TIME  : %d clock ticks\n\n",((float)etime-(float)stime));
	return 0;

}
