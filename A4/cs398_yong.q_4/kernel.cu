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
							int height) 
{
	__shared__ unsigned int pHistogram[HISTOGRAM256_BIN_COUNT];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int wb = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	if (wb < HISTOGRAM256_BIN_COUNT) {
		pHistogram[wb] = 0;
	}
	__syncthreads();

	if(x < width && y < height) {
		int index = y * width + x;
		unsigned char v = (unsigned char)input[index];
		atomicAdd(&pHistogram[(int)v],1);
	}

	__syncthreads();
		
	if(wb < HISTOGRAM256_BIN_COUNT) {
		atomicAdd(&output[wb], pHistogram[wb]);
	}

}
__global__ void cdfScan(unsigned int *input, 
						float *output,
						float width, 
						float height) 
{
	__shared__ unsigned int pHistogram[HISTOGRAM256_BIN_COUNT];

    //int i = blockIdx.x * blockDim.x + threadIdx.x;

	pHistogram[threadIdx.x] = input[threadIdx.x]; //PROBABILITY((float)input[threadIdx.x], width, height);
	pHistogram[threadIdx.x + blockDim.x] = input[threadIdx.x + blockDim.x]; //PROBABILITY((float)input[threadIdx.x + blockDim.x], width, height);
	
	for (unsigned int stride = 1;stride <=  blockDim.x; stride *= 2) {
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index < 2* blockDim.x)
		pHistogram[index] += pHistogram[index-stride];
		__syncthreads();
	}

	__syncthreads();

	for (unsigned int stride =  blockDim.x /2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index+stride < 2*  blockDim.x) {
			pHistogram[index + stride] += pHistogram[index];
		}
	}
	__syncthreads();

	float max_value = pHistogram[HISTOGRAM256_BIN_COUNT - 1];
	float cdfMin = pHistogram[0]/max_value;

	output[threadIdx.x] =  (float)CORRECT_COLOR(pHistogram[threadIdx.x]/max_value, cdfMin);
	output[threadIdx.x + blockDim.x] = (float)CORRECT_COLOR(pHistogram[threadIdx.x + blockDim.x]/max_value, cdfMin);
}


__global__ void applyHistogram(	unsigned char *input1, 
								float *input2,
								unsigned char *output,
								int width, 
								int height, 
								int channels) 
{
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < width && y < height) {
		int index = width * y + x;
		output[index] = (unsigned char)input2[(int)input1[index]];
	}
	
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
		
		uint offset = i * imgWidth* imgHeight;
		uint hist_offset = i * HISTOGRAM256_BIN_COUNT;

		dim3 dimGrid((imgWidth-1) / BLOCK_SIZE + 1, (imgHeight-1) / BLOCK_SIZE + 1, imgChannels);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

		histogram<<<dimGrid, dimBlock>>>(
			(unsigned char*)d_DataIn + offset, (unsigned int*)d_Histogram + hist_offset, (int)imgWidth, (int)imgHeight
		);

		checkCudaErrors(cudaDeviceSynchronize());

		dim3 dimGridCdf(1,1,1);
		dim3 dimBlockCdf(HISTOGRAM256_BIN_COUNT/2, 1, 1);

		cdfScan<<<dimGridCdf,dimBlockCdf>>>(
			(unsigned int*)d_Histogram + hist_offset,
			(float*)histogramCdf,
			(float)imgWidth, (float)imgHeight
		);

		//cdfScan
		checkCudaErrors(cudaDeviceSynchronize());
		
		applyHistogram <<<dimGrid, dimBlock >> > (
			(unsigned char*)d_DataIn + offset,
			(float*)histogramCdf,
			(unsigned char*)d_DataOut + offset,
			(int)imgWidth, (int)imgHeight, (int)imgChannels		
		);
		
		checkCudaErrors(cudaDeviceSynchronize());
	}

}
