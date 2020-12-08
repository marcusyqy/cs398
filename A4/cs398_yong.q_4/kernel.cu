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
#include "histogram_common.h"

#define BLOCK_SIZE 32

///you may use the following declarations
__global__ void histogram(	unsigned char *input, 
							unsigned int *output,
							int width, 
							int height) {

}
__global__ void cdfScan(unsigned int *input, 
						float *output,
						float width, 
						float height) {

	for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t+stride];
	}
}


__global__ void applyHistogram(	unsigned char *input1, 
								float *input2,
								float *input3, 
								unsigned char *output,
								int width, 
								int height, 
								int channels) 
{
}
////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////




// Utility and system includes
//#include <helper_cuda.h>
//#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

float* histogramCdf = nullptr;
//Internal memory allocation
extern "C" void initHistogram256(void)
{
	checkCudaErrors(cudaMalloc(&histogramCdf, HISTOGRAM256_BIN_COUNT * sizeof(float)));
}

//Internal memory deallocation
extern "C" void closeHistogram256(void)
{
	checkCudaErrors(cudaFree(histogramCdf));
}

extern "C" void histogram256(
	uint *d_Histogram,
	void *d_DataIn,
	void *d_DataOut,
	uint byteCount,
	uint imgWidth,
	uint imgHeight,
	uint imgChannels
)
{
	for (uint i = 0; i < imgChannels; ++i) { 
		
		dim3 dimGrid((imgWidth-1) / BLOCK_SIZE + 1, (imgHeight-1) / BLOCK_SIZE + 1, imgChannels);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

		histogram<<<dimGrid, dimBlock>>>(
			(float*)d_DataIn, (unsigned int*)d_Histogram, (int)imgWidth, (int)imgHeight
		);

		dim3 dimGrid((HISTOGRAM256_BIN_COUNT-1) / BLOCK_SIZE + 1, (HISTOGRAM256_BIN_COUNT-1) / BLOCK_SIZE + 1, imgChannels);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
		
		//cdfScan




	}

}
