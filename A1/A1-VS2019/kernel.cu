/*
* Copyright 2018 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*/

#include <helper_cuda.h>
////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;

__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{

	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < nRowPoints - 1 && j > 0 && j < nRowPoints - 1)
	{
		out[j * nRowPoints + i] = (
			in[j * nRowPoints + i - 1] +
			in[j * nRowPoints + i + 1] +
			in[(j - 1) * nRowPoints + i] +
			in[(j + 1) * nRowPoints + i]
			) * 0.25f;
	}

}
///Shared memory kernel function for heat distribution calculation
__global__ void heatDistrCalcShm(float* in, float* out, uint nRowPoints)
{

}

__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < nRowPoints -1  && j > 0 && j < nRowPoints - 1)
	{
		uint index = j * nRowPoints + i;
		out[index] = in[index];
	}

}

extern "C" void heatDistrGPU(
	float* d_DataIn,
	float* d_DataOut,
	uint nRowPoints,
	uint nIter
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	for (uint k = 0; k < nIter; k++) {
		heatDistrCalc << <DimGrid2, DimBlock >> > ((float*)d_DataIn,
			(float*)d_DataOut,
			nRowPoints);
		getLastCudaError("heatDistrCalc failed\n");
		cudaDeviceSynchronize();
		heatDistrUpdate << < DimGrid2, DimBlock >> > ((float*)d_DataOut,
			(float*)d_DataIn,
			nRowPoints);
		getLastCudaError("heatDistrUpdate failed\n");
		cudaDeviceSynchronize();
	}
}
