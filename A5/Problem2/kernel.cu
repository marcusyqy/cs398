/*Start Header
******************************************************************/
/*!
\file kernel.cu
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date October 18, 2020
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

__global__ void MatrixMulGPUCalc(const double* A, const double* B, double* C, uint rowA, uint colA, uint colB)
{
	__shared__ double ASharedMemory[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ double BSharedMemory[BLOCK_SIZE*BLOCK_SIZE];
	
	uint width = colA;

	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	uint x = threadIdx.x;
	uint y = threadIdx.y;

	uint colC = colB;
	//load matrices

	double pValue = 0.0;
	uint endP = ((width - 1) / BLOCK_SIZE) + 1;

	for (uint p = 0; p < endP; ++p)
	{
		uint px = p * BLOCK_SIZE + x;
		uint py = p * BLOCK_SIZE + y;

		ASharedMemory[y * BLOCK_SIZE + x] = (px < colA && j < rowA) ? A[j * colA + px] : 0.0;
		BSharedMemory[y * BLOCK_SIZE + x] = (py < colA && i < colB) ? B[py * colB + i] : 0.0;

		__syncthreads();

		//should be correct
		for (uint k = 0; k < BLOCK_SIZE; ++k)
		{
			pValue += ASharedMemory[y * BLOCK_SIZE + k] * BSharedMemory[k * BLOCK_SIZE + x];
		}
		__syncthreads();
	}


	if (j < rowA && i < colB)
	{
		C[j * colC + i] = pValue;
	}

	
}

extern "C" void MatrixMulGPU(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA, 
	uint colA, 
	uint colB
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2((unsigned int)ceil(((float)colB) / BLOCK_SIZE), (unsigned int)ceil(((float)rowA) / BLOCK_SIZE), 1);
	
	MatrixMulGPUCalc<< < DimGrid2, DimBlock>> >(inA, inB, out, rowA, colA, colB);
	getLastCudaError("MatrixMul failed\n");
	checkCudaErrors(cudaDeviceSynchronize());
}


void host_pin_memory(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB
)
{
	uint rowB = colA;
	// loop through all column tiles for A
	for(uint i = 0; i < colA; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < rowA; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < colB; ++k) {




			}
		}
	}
}

void partial_calc(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB
)
{
	uint rowB = colA;	
	// loop through all column tiles for A
	for(uint i = 0; i < colA; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < rowA; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < colB; ++k) {
				
			}
		}
	}
}

void copy_back(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB
)
{
	uint rowB = colA;
	// loop through all column tiles for A
	for(uint i = 0; i < colA; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < rowA; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < colB; ++k) {




			}
		}
	}
}

//necessary includes
#include <thread>
#include <array>

extern "C" void MatrixMulGPUStream(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB
)
{
	// initializing... 

	// allocate pinned memory
	float* h_A;
	float* h_B;
	float* h_C;

	cudaHostAlloc((void **) &h_A, N * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void **) &h_B, N * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void **) &h_C, N * sizeof(float), cudaHostAllocDefault);


	
	//start running all here ...

	auto t1 = [inA, inB, out, rowA, colA, colB]() { 
		host_pin_memory(inA, inB, out, rowA, colA, colB);
	};
	auto t2 = [inA, inB, out, rowA, colA, colB]() { 
		partial_calc(inA, inB, out, rowA, colA, colB);
	};
	auto t3 = [inA, inB, out, rowA, colA, colB]() { 
		copy_back(inA, inB, out, rowA, colA, colB);
	};

	std::array<std::thread, 3> threads {
		std::thread{t1}, std::thread{t2}, std::thread{t3}
	};

	for(auto& t: threads) {
		t.join();
	}
}