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


cudaStream_t next_stream(cudaStream_t* streams, size_t num_streams)
{
	size_t i{}; 
	for(;;) {
		bool operations_pending = cudaStreamQuery(streams[i]) == cudaErrorNotReady;
		if(!operations_pending)
			break;
		i = (i + 1)%num_streams;
	}

	return streams[i];
}

struct mat_mul
{
	void host_pin_memory();
	void partial_calc();
	void copy_back();
	
	// client data
	const double* inA;
	const double* inB;
	double* out;
	uint rowA;
	uint colA;
	uint colB;
	uint m; 
	uint n;

	// streams
	cudaStream_t* streams;
	size_t s;

	// host pinned memory
	double* h_A;
	double* h_B;
	double* h_C;

	//device memory
	double* d_A;
	double* d_B;
	double* d_C;

	//events 
	cudaEvent_t host_memory_avail, host_memory_ready;
	
};


void mat_mul::host_pin_memory(void)
{
	uint rowB = colA;
	// loop through all column tiles for A
	for(uint i = 0; i < colA; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < rowA; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < colB; ++k) {

				// get next stream
				auto stream = next_stream(streams, s);
				//just to be safe
				cudaStreamSynchronize(stream);
				
				//wait for host to be available
				cudaEventSynchronize(host_memory_avail);
				
				//cudaStreamWaitEvent(stream, event);
				cudaEventRecord(host_memory_ready);

			}
		}
	}
}

void mat_mul::partial_calc(void)
{
	uint rowB = colA;	
	// loop through all column tiles for A
	for(uint i = 0; i < colA; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < rowA; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < colB; ++k) {
				

				
				
				cudaEventSynchronize(host_memory_ready);
				
				// checkCudaErrors(cudaMemcpyAsync((void **) &d_A, m * sizeof(double)));
				// checkCudaErrors(cudaMalloc((void **) &d_B, n * sizeof(double)));
				// // output device memory
				// checkCudaErrors(cudaMalloc((void **) &d_C, m * n * sizeof(double)));
				//do memcpy
				cudaEventRecord(host_memory_avail);
			}
		}
	}
}

void mat_mul::copy_back(void)
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
#include <vector>

static constexpr size_t num_threads_ = 3;

extern "C" void MatrixMulGPUStream(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB,
	uint m,
	uint n,
	uint s
)
{
	// initializing... 
	// allocate pinned memory
	double* h_A;
	double* h_B;
	double* h_C;

	// input host memory
	checkCudaErrors(cudaHostAlloc((void **) &h_A, m * sizeof(double), cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **) &h_B, n * sizeof(double), cudaHostAllocMapped));
	// output host memory
	checkCudaErrors(cudaHostAlloc((void **) &h_C, m * n * sizeof(double), cudaHostAllocMapped));
	
	
	double* d_A;
	double* d_B;
	double* d_C;
	// input device memory
	checkCudaErrors(cudaMalloc((void **) &d_A, m * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **) &d_B, n * sizeof(double)));
	// output device memory
	checkCudaErrors(cudaMalloc((void **) &d_C, m * n * sizeof(double)));

	cudaEvent_t host_memory_avail, host_memory_ready;
	checkCudaErrors(cudaEventCreate(&host_memory_avail));
	checkCudaErrors(cudaEventCreate(&host_memory_ready));

	std::vector<cudaStream_t> streams{(size_t)s, nullptr};
	for(auto& stream : streams) {
		checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}

	mat_mul App{};

	//client data
	App.inA = inA;
	App.inB = inB;
	App.out = out;
	App.rowA = rowA;
	App.colA = colA;
	App.colB = colB;
	App.m = m;
	App.n = n;

	// streams
	App.streams = streams.data();
	App.s = streams.size();

	// host pinned memory 
	App.h_A = h_A;
	App.h_B = h_B;
	App.h_C = h_C;

	//device memory
	App.d_A = d_A;
	App.d_B = d_B;
	App.d_C = d_C;

	//events 
	App.host_memory_avail = host_memory_avail;
	App.host_memory_ready = host_memory_ready;


	//start running all here ...
	auto t1 = [&App]() -> void { 
		App.host_pin_memory();
	};
	auto t2 = [&App]() -> void { 
		App.partial_calc();
	};
	auto t3 = [&App]() -> void { 
		App.copy_back();
	};

	std::array<std::thread, num_threads_> threads {
		std::thread{t1}, std::thread{t2}, std::thread{t3}
	};

	for(auto& t: threads) {
		t.join();
	}


	checkCudaErrors(cudaFreeHost(h_A));
	checkCudaErrors(cudaFreeHost(h_B));
	checkCudaErrors(cudaFreeHost(h_C));

	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	for(auto& stream : streams) {
		checkCudaErrors(cudaStreamDestroy(stream));
	}

	checkCudaErrors(cudaEventDestroy(host_memory_avail));
	checkCudaErrors(cudaEventDestroy(host_memory_ready));
	
}