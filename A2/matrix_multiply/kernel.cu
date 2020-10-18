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

__global__ void MatrixMulGPUCalc(const float* A, const float* B, float* C, uint rowA, uint colA, uint colB)
{
	__shared__ float* ASharedMemory[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float* BSharedMemory[BLOCK_SIZE][BLOCK_SIZE];
	
	uint rowB = colA;

	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	uint x = threadIdx.x;
	uint y = threadIdx.y;

	//load matrices

	if(i < colA && j < rowA)
	{
		ASharedMemory[y][x] = A[j*colA + i];
	}
	if(i < colB && j < rowB)
	{
		BSharedMemory[y][x] = B[j*colB + i];
	}

	// finish loading
	__syncthreads();


	//matrix multiply


}

extern "C" void MatrixMulGPU(
	const float* inA, 
	const float* inB, 
	float* out,
	uint rowA, 
	uint colA, 
	uint colB
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2((unsigned int)ceil(((float)nRowPoints) / BLOCK_SIZE), (unsigned int)ceil(((float)nRowPoints) / BLOCK_SIZE), 1);
	MatrixMulGPUCalc<< < DimGrid2, DimBlock >> >(inA, inB, out, rowA, colA, colB);
}