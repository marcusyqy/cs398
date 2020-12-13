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

//necessary includes
#include <thread>
#include <array>
#include <vector>
#include <utility>

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
	//t1
	void host_pin_memory();
	//t2
	void partial_calc();
	//t3
	void copy_back();
	
	// client data
	const double* inA;
	const double* inB;
	double* out;
	uint rowA;
	uint colA;
	uint colB;
	uint block_size; 
	uint tile_size;

	// streams
	cudaStream_t* streams;
	size_t num_streams;

	// host pinned memory
	double* h_A;
	double* h_B;
	double* h_C;

	//device memory
	double* d_A;
	double* d_B;
	double* d_C;

	// loop stuff
	uint num_A_columns;
	uint num_A_rows;
	uint num_B_columns; 


	//events 
	cudaEvent_t host_memory_avail, host_memory_ready, device_copied_memory;
	
};


void mat_mul::host_pin_memory(void)
{
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {
				// get next stream
				auto stream = next_stream(streams, num_streams);

				//just to be safe
				cudaStreamSynchronize(stream);
				//wait for host to be available
				cudaEventSynchronize(host_memory_avail);

				uint rows = std:min((j + 1)* block_size, rowA) - j*block_size;
				uint col_a = std:min((i + 1)*tile_size, colA) - i*tile_size;
				uint col_b = std:min((k + 1)*block_size, colB) - k*block_size;

				cudaMemcpy2DAsync(
					(void*)h_A,
					(size_t)tile_size,
					(const void*)(inA + , 
					(size_t)colA,
					(size_t)tile_size * sizeof(double),
					(size_t)block_size,
					cudaMemcpyHostToHost,
					stream
				);

				cudaMemcpy2DAsync(
					(void*)h_B,
					(size_t)block_size,
					(const void*)(inB + (colB)*h + w),
					(size_t)colB,
					(size_t)block_size * sizeof(double),
					(size_t)tile_size,
					cudaMemcpyHostToHost,
					stream
				);

				//cudaStreamWaitEvent(stream, event);


				cudaEventRecord(host_memory_ready);
			}
		}
	}
}

void mat_mul::partial_calc(void)
{
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {
				
				auto stream = next_stream(streams, num_streams);
				//just to be safe
				cudaStreamSynchronize(stream);
				
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
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {
				auto stream = next_stream(streams, num_streams);
				
				
				
				//just to be safe
				cudaStreamSynchronize(stream);
			}
		}
	}
}


static constexpr size_t num_threads_ = 3;

extern "C" void MatrixMulGPUStream(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB,
	uint blockSize,
	uint tileSize,
	uint numberOfStreams
)
{
	//pin memory
	

	// initializing... 
	// allocate pinned memory
	double* h_A;
	double* h_B;
	double* h_C;

	// input host memory
	checkCudaErrors(cudaHostAlloc((void **) &h_A, blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **) &h_B, blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
	// output host memory
	checkCudaErrors(cudaHostAlloc((void **) &h_C, blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
	
	// allocate device local memory
	double* d_A;
	double* d_B;
	double* d_C;
	// input device memory
	checkCudaErrors(cudaMalloc((void **) &d_A, blockSize * tileSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **) &d_B, blockSize * tileSize * sizeof(double)));
	// output device memory
	checkCudaErrors(cudaMalloc((void **) &d_C, blockSize * tileSize  * sizeof(double)));

	cudaEvent_t host_memory_avail, host_memory_ready;
	checkCudaErrors(cudaEventCreate(&host_memory_avail));
	checkCudaErrors(cudaEventCreate(&host_memory_ready));

	std::vector<cudaStream_t> streams{(size_t)numberOfStreams, nullptr};
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
	App.block_size = blockSize;
	App.tile_size = tileSize;

	// streams
	App.streams = streams.data();
	App.num_streams = streams.size();

	// host pinned memory 
	App.h_A = h_A;
	App.h_B = h_B;
	App.h_C = h_C;

	//device memory
	App.d_A = d_A;
	App.d_B = d_B;
	App.d_C = d_C;

	//tile stuff
	App.num_A_columns = ( colA - 1 ) / tileSize + 1 
	App.num_A_rows = ( rowA  - 1 ) / blockSize + 1
	App.num_B_columns = ( colB - 1 ) / blockSize + 1

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