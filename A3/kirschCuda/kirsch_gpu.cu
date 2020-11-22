/*
* Copyright 2018 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/

#include <helper_cuda.h>
#include "edge.h"

#define TILE_WIDTH 16
#define O_TILE_WIDTH 14
#define MASK_WIDTH 3
#define HALF_MASK_WIDTH 1
#define MASK_WIDTH_SQ 9
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define NUM_MASKS 8 

//@@ INSERT CODE HERE
//Use of const  __restrict__ qualifiers for the mask parameter 
//informs the compiler that it is eligible for constant caching

///Design 1: The size of each thread block matches the size of an 
///output tile. All threads participate in calculating output elements.
__global__ void convolution(unsigned char *I, 
							const int* __restrict__ M,
							unsigned char *P, 
							int channels, 
							int width,
							int height) 
{
#define SM_SIZE_TILE (TILE_WIDTH + MASK_WIDTH - 1)
	
	__shared__ int shared_data[SM_SIZE_TILE*SM_SIZE_TILE];

	//1) loading stage
	
	int col =  blockIdx.x * TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	
	int load_col = col - HALF_MASK_WIDTH;
	int load_row = row - HALF_MASK_WIDTH;

	int size = width * height;
	int channel_offset = blockIdx.z * size;

	int idx = threadIdx.y * SM_SIZE_TILE + threadIdx.x;

	//loading shared memory
	if((load_col >= 0) && (load_col < width) && (load_row >= 0) && (load_row < height)) {
		shared_data[idx] = I[channel_offset + load_row * width + load_col];
	}
	else {
		shared_data[idx] = 0;
	}

	int candidate_x = threadIdx.x + TILE_WIDTH;
	int candidate_y = threadIdx.y + TILE_WIDTH;
	
	int load_col2 = load_col + TILE_WIDTH;
	int load_row2 = load_row + TILE_WIDTH;
	
	if(candidate_x < SM_SIZE_TILE) {
		int idx2 = threadIdx.y * SM_SIZE_TILE + candidate_x;
		if((load_col2 >= 0) && (load_col2 < width) && (load_row >= 0) && (load_row < height)) {
			shared_data[idx2] = I[channel_offset + load_row * width + load_col2];
		}
		else {
			shared_data[idx2] = 0;
		}
	}

	if (candidate_y < SM_SIZE_TILE){
		int idx2 = candidate_y * SM_SIZE_TILE + threadIdx.x;
		if((load_col >= 0) && (load_col < width) && (load_row2 >= 0) && (load_row2 < height)) {
			shared_data[idx2] = I[channel_offset + load_row2 * width + load_col];
		}
		else {
			shared_data[idx2] = 0;
		}
	}

	if(candidate_x < SM_SIZE_TILE && candidate_y < SM_SIZE_TILE) {
		int idx2 = candidate_y * SM_SIZE_TILE + candidate_x;
		if((load_col2 >= 0) && (load_col2 < width) && (load_row2 >= 0) && (load_row2 < height)) {
			shared_data[idx2] = I[channel_offset + load_row2 * width + load_col2];
		}
		else {
			shared_data[idx2] = 0;
		}		
	}


	// sync after loading to shared memory
	__syncthreads();

	int sum = 0;
	int i_x = threadIdx.x;
	int i_y = threadIdx.y;
	// check for correct idx
	if(i_x < TILE_WIDTH  && i_y < TILE_WIDTH) {
		for(int n = 0; n < NUM_MASKS; n++) {
			int isum = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					isum = isum + (M[n * MASK_WIDTH_SQ + k*MASK_WIDTH + l]
							*(int)shared_data[((i_y + k) * SM_SIZE_TILE + (i_x + l))]);
				}
			}

			if(isum > sum)
				sum = isum;
		}

		int icol = col;
		int irow = row;
		if((icol >= 0) && (icol < width) &&(irow >= 0) && (irow < height)) {
			P[channel_offset + irow * width + icol] = mymin(mymax(sum / 8, 0), 255);
		}
	}
}

///Design 2: The size of each thread block matches the size of 
///an input tile. Each thread loads one input element into the 
///shared memory.

__global__ void convolution2(unsigned char *I,
							const int* __restrict__ M,
							unsigned char *P,
							int channels,
							int width,
							int height)
{
	__shared__ unsigned char shared_data[BLOCK_WIDTH*BLOCK_WIDTH];

	int col =  blockIdx.x * O_TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y * O_TILE_WIDTH + threadIdx.y;

	int load_col = col - HALF_MASK_WIDTH;
	int load_row = row - HALF_MASK_WIDTH;
	
	int idx = threadIdx.y * BLOCK_WIDTH + threadIdx.x;

	int size = width * height;
	int channel_offset = blockIdx.z * size;

	//loading shared memory
	if((load_col >= 0) && (load_col < width) && (load_row >= 0) && (load_row < height)) {
		shared_data[idx] = I[channel_offset + load_row * width + load_col];
	}
	else {
		shared_data[idx] = 0;
	}

	// sync after loading to shared memory
	__syncthreads();

	int sum = 0;
	int i_x = threadIdx.x;
	int i_y = threadIdx.y;
	// check for correct idx
	if(i_x < (O_TILE_WIDTH)  && i_y < (O_TILE_WIDTH)) {
		for(int n = 0; n < NUM_MASKS; n++) {
			int isum = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					isum 
						= isum + (M[n * MASK_WIDTH_SQ + k*MASK_WIDTH + l]*
						(int)shared_data[((i_y + k) * BLOCK_WIDTH + (i_x + l))]);
				}
			}

			if(isum > sum)
				sum = isum;
		}

		int icol = col;
		int irow = row;
		if((icol >= 0) && (icol < width) &&(irow >= 0) && (irow < height)) {
			P[channel_offset + irow * width + icol] = mymin(mymax(sum / 8, 0), 255);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU 
////////////////////////////////////////////////////////////////////////////////

extern "C" void kirschEdgeDetectorGPU(
	void *d_ImgDataIn,
	void *d_ImgMaskData,
	void *d_ImgDataOut,
	unsigned imgChannels,
	unsigned imgWidth,
	unsigned imgHeight
)
{
#if 1
	//each channel use blockIdx.z dimension 
/*
	dim3 dimGrid((imgWidth - 1) / TILE_WIDTH + 1,
		(imgHeight - 1) / TILE_WIDTH + 1, 3);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	*/
	dim3 dimGrid((imgWidth - 1) / TILE_WIDTH + 1,
		(imgHeight - 1) / TILE_WIDTH + 1, imgChannels);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution << <dimGrid, dimBlock >> >((unsigned char *)d_ImgDataIn,
		(int *)d_ImgMaskData,
		(unsigned char *)d_ImgDataOut,
		(int)imgChannels,
		(int)imgWidth,
		(int)imgHeight);
#else
	//each channel use blockIdx.z dimension 
/*
	dim3 dimGrid((imgWidth-1) / O_TILE_WIDTH + 1,
	(imgHeight-1) / O_TILE_WIDTH + 1, 3);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
*/
	dim3 dimGrid((imgWidth-1) / O_TILE_WIDTH + 1,
		(imgHeight-1) / O_TILE_WIDTH + 1, imgChannels);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	convolution2 << <dimGrid, dimBlock >> >((unsigned char *)d_ImgDataIn,
		(int *)d_ImgMaskData,
		(unsigned char *)d_ImgDataOut,
		(int)imgChannels,
		(int)imgWidth,
		(int)imgHeight);
#endif
	cudaDeviceSynchronize();
	getLastCudaError("Compute the kirsch edge detection failed\n");
}

