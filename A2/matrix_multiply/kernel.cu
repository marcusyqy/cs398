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
	__shared__ float sm_memory[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i > 0 && i < nRowPoints - 1 && j > 0 && j < nRowPoints - 1)
	{
		const uint x = threadIdx.x + 1;
		const uint y = threadIdx.y + 1;

		sm_memory[y][ x] = in[j * nRowPoints + i];

		if (x == 1 || i == 1)
		{
			sm_memory[y][x - 1] = in[j * nRowPoints + i - 1];
		}
		if (x == BLOCK_SIZE || i == nRowPoints-2)
		{
			sm_memory[y][x + 1] = in[j * nRowPoints + i + 1];
		}

		if (y == 1 || j == 1)
		{
			sm_memory[y - 1][x] = in[(j - 1) * nRowPoints + i];
		}
		if (y == BLOCK_SIZE || j == nRowPoints -2)
		{
			sm_memory[y + 1][x] = in[(j + 1) * nRowPoints + i];
		}

		__syncthreads();

		out[j * nRowPoints + i] = (
				sm_memory[y][x - 1] +
				sm_memory[y][x + 1] +
				sm_memory[y - 1][x] +
				sm_memory[y + 1][x]
			) * 0.25f;
	}
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
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	uint offset_j = j + minColPoints;

	//printf("%d, %d\n", blockIdx.x, blockIdx.y);
	if (i < nRowPoints && j < nColPoints)
	{
		if (i == 0 || (i == nRowPoints - 1) || offset_j == 0 || (offset_j == nRowPoints - 1))
		{
			out[j * nRowPoints + i] = in[offset_j * nRowPoints + i];
		}
		else
		{
			//printf("setting index with : %d\nWith offset %d, %d,%d\n", j * nRowPoints + i, offset_j, j,i);
			out[j * nRowPoints + i] = (
				in[offset_j * nRowPoints + i - 1] +
				in[offset_j * nRowPoints + i + 1] +
				in[(offset_j - 1) * nRowPoints + i] +
				in[(offset_j + 1) * nRowPoints + i]
				) * 0.25f;
		}
	}

}

__global__ void heatDistrUpdateRegion(float* in, float* out, uint nRowPoints, uint minColPoints, uint nColPoints)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nRowPoints && j < nColPoints)
	{
		uint offset_j = j + minColPoints;
		out[offset_j*nRowPoints + i] = in[j* nRowPoints + i];
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
	dim3 DimGrid2((unsigned int)ceil(((float)nRowPoints) / BLOCK_SIZE), (unsigned int)ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	for (uint k = 0; k < nIter; k++) {
		heatDistrCalc<< <DimGrid2, DimBlock >> > ((float*)d_DataIn,
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
	dim3 DimGrid2((unsigned int)ceil(((float)nRowPoints) / BLOCK_SIZE), (unsigned int)ceil(((float)nBatchColPoints) / BLOCK_SIZE), 1);
	uint lastBlock = nRowPoints - 3 * nBatchColPoints;

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


		cudaDeviceSynchronize();
		getLastCudaError("heatDistrCalc E1 failed\n");
		//E2
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer2,
			nRowPoints, nBatchColPoints, nBatchColPoints);
		
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrCalc E2 failed\n");

		//C1
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer1,
		(float*)d_FinalData,
			nRowPoints, 0, nBatchColPoints);

		cudaDeviceSynchronize();
		getLastCudaError("heatDistrUpdate C1 failed\n");
		//E3
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer1,
			nRowPoints, 2*nBatchColPoints, nBatchColPoints);
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrCalc E3 failed\n");


		//C2
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer2,
			(float*)d_FinalData,
			nRowPoints, nBatchColPoints, nBatchColPoints);
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrUpdate C2 failed\n");


		//E4
		heatDistrCalcRegion << <DimGrid2, DimBlock >> > ((float*)d_FinalData,
			(float*)d_TempBuffer2,
			nRowPoints, 3 * nBatchColPoints, lastBlock);
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrCalc E4 failed\n");


		//C3
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer1,
			(float*)d_FinalData,
			nRowPoints, 2 * nBatchColPoints, nBatchColPoints);
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrUpdate C3 failed\n");

		//C4
		heatDistrUpdateRegion << <DimGrid2, DimBlock >> > ((float*)d_TempBuffer2,
			(float*)d_FinalData,
			nRowPoints, 3 * nBatchColPoints, lastBlock);
		cudaDeviceSynchronize();
		getLastCudaError("heatDistrUpdate C4 failed\n");

	}
}