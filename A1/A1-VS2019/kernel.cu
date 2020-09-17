/*Start Header
******************************************************************/
/*!
\file kernel.cu
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date Sept 17, 2020
\brief
	gpu computing functions
Copyright (C) 2020 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/




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


__global__ void heatDistrCalcRegion(float* in, float* out, uint nRowPoints, uint minColPoints, uint nColPoints)
{

	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint real_j = blockIdx.y * blockDim.y + threadIdx.y;
	uint j = real_j + minColPoints;

	if (i > 0 && i < nRowPoints - 1 && j >= minColPoints && j < (minColPoints + nColPoints))
	{
		out[real_j * nRowPoints + i] = (
			in[j * nRowPoints + i - 1] +
			in[j * nRowPoints + i + 1] +
			in[(j - 1) * nRowPoints + i] +
			in[(j + 1) * nRowPoints + i]
			) * 0.25f;
	}
}

__global__ void heatDistrUpdateRegion(float* in, float* out, uint nRowPoints, uint minColPoints, uint nColPoints)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint real_j = blockIdx.y * blockDim.y + threadIdx.y;
	uint j = real_j + minColPoints;


	if (i > 0 && i < nRowPoints - 1 && j >= minColPoints && j < (minColPoints + nColPoints))
	{
		uint index_out = real_j * nRowPoints + i;
		uint index = j * nRowPoints + i;
		out[index] = in[index_out];
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

extern "C" void batchHeatDistrGPU(
	float* d_FinalData,
	float* d_TempBuffer1,
	float* d_TempBuffer2,
	uint nRowPoints,
	uint nBatchColPoints,
	uint nIter
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nBatchColPoints) / BLOCK_SIZE), 1);

	uint lastBlock = nRowPoints - 3 * nBatchColPoints - 1;

	for (uint k = 0; k < nIter; k++) {

		//E1 
		//E2
		//C1
		//E3
		//C2
		//E4
		//C3
		//C4

		//E1
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer1,
			nRowPoints, 0, nBatchColPoints);
		getLastCudaError("heatDistrCalc E1 failed\n");

		//E2
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer2,
			nRowPoints, nBatchColPoints, nBatchColPoints);
		getLastCudaError("heatDistrCalc E2 failed\n");

		cudaDeviceSynchronize();

		//C1
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer1,
			(float*)d_FinalData,
			nRowPoints, 0, nBatchColPoints);
		getLastCudaError("heatDistrUpdate C1 failed\n");

		cudaDeviceSynchronize();
		//E3
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer1,
			nRowPoints, 2*nBatchColPoints, nBatchColPoints);
		getLastCudaError("heatDistrCalc E3 failed\n");

		cudaDeviceSynchronize();

		//C2
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer2,
			(float*)d_FinalData,
			nRowPoints, nBatchColPoints, nBatchColPoints);
		getLastCudaError("heatDistrUpdate C2 failed\n");

		cudaDeviceSynchronize();

		//E4
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer2,
			nRowPoints, 3 * nBatchColPoints, lastBlock);
		getLastCudaError("heatDistrCalc E4 failed\n");

		cudaDeviceSynchronize();

		//C3
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer1,
			(float*)d_FinalData,
			nRowPoints, 2 * nBatchColPoints, nBatchColPoints);
		getLastCudaError("heatDistrUpdate C3 failed\n");

		cudaDeviceSynchronize();
		//C4
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer2,
			(float*)d_FinalData,
			nRowPoints, 3 * nBatchColPoints, lastBlock);
		getLastCudaError("heatDistrUpdate C4 failed\n");

		cudaDeviceSynchronize();
	}
}