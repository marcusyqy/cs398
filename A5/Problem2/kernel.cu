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
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>

class semaphore
{
public:

	semaphore(int count_ = 0) : count{ count_ }
	{}

	void signal()
	{
		std::unique_lock<std::mutex> lck(mtx);
		++count;
		cv.notify_one();
	}

	void wait()
	{
		std::unique_lock<std::mutex> lck(mtx);
		while (count == 0)
		{
			cv.wait(lck);
		}

		--count;
	}

private:

	std::mutex mtx;
	std::condition_variable cv;
	int count;
};


struct tile_data 
{
	//stream
	cudaStream_t stream;

	//host pointers
	double* h_a;
	double* h_b;
	double* h_c;
	
	//device pointers
	double* d_a;
	double* d_b;
	double* d_c;

	// sync objects
	semaphore _12, _23, _31;
	bool init_once = false;
};

struct mat_mul
{
	//t1
	void host_pin_memory();
	//t2
	void partial_calc();
	//t3
	void copy_back();

	size_t next_stream_id(size_t id);
	
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
	//cudaStream_t* streams;
	//size_t num_streams;

	std::vector<tile_data> tiles;
	
	// loop stuff
	uint num_A_columns;
	uint num_A_rows;
	uint num_B_columns; 
	
};

size_t mat_mul::next_stream_id(size_t id)
{
	size_t i = (id + 1) % tiles.size();
	//cudaStreamSynchronize(tiles[i].stream);

	return i;
}


void mat_mul::host_pin_memory(void)
{
	if (tiles.empty())
		return;

	size_t stream_id{ tiles.size() - 1 };
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {
				// get next stream
				stream_id = next_stream_id(stream_id);
				
				auto& t = tiles[stream_id];
				cudaStreamSynchronize(t.stream);

				if (!t.init_once) {
					t.init_once = true;
				}
				else {
					t._31.wait();
				}

				uint offset_rows = j * block_size;
				uint offset_col_a = i * tile_size;
				uint offset_col_b = k * block_size;

				uint diff_rows = std::min((j + 1) * block_size, rowA) - offset_rows;
				uint diff_col_a = std::min((i + 1)*tile_size, colA) - offset_col_a;
				uint diff_col_b = std::min((k + 1)*block_size, colB) - offset_col_b;

				// reset values to 0
				memset(t.h_a, 0, sizeof(double)* block_size * tile_size);
				memset(t.h_b, 0, sizeof(double)* block_size * tile_size);
				memset(t.h_c, 0, sizeof(double)* block_size * block_size);

				//copy a
				for (uint row = 0; row < diff_rows; ++row) {
					memcpy(t.h_a + row * tile_size, inA + (row  + offset_rows )* colA + offset_col_a, diff_col_a * sizeof(double));
				}

				//copy b
				for (uint row = 0; row < diff_col_a; ++row) {
					memcpy(t.h_b + row * block_size, inB + (row + offset_col_a) * colB + offset_col_b, diff_col_b * sizeof(double));
				}

				t._12.signal();
			}
		}
	}
}

void mat_mul::partial_calc(void)
{
	if (tiles.empty())
		return;

	size_t stream_id{ tiles.size() - 1 };
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {

				stream_id = next_stream_id(stream_id);
				auto& t = tiles[stream_id];
				t._12.wait();

				// ... hard part

				uint offset_rows = j * block_size;
				uint offset_col_a = i * tile_size;
				uint offset_col_b = k * block_size;

				uint diff_rows = std::min((j + 1) * block_size, rowA) - offset_rows;
				uint diff_col_a = std::min((i + 1) * tile_size, colA) - offset_col_a;
				uint diff_col_b = std::min((k + 1) * block_size, colB) - offset_col_b;

				//h2d
				cudaMemcpyAsync(
					t.d_a, t.h_a, tile_size * block_size* sizeof(double), cudaMemcpyHostToDevice, t.stream
				);

				cudaMemcpyAsync(
					t.d_b, t.h_b, tile_size * block_size* sizeof(double), cudaMemcpyHostToDevice, t.stream
				);

				//kernel

				dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
				dim3 DimGrid2((unsigned int)ceil(((float)block_size) / BLOCK_SIZE), (unsigned int)ceil(((float)block_size) / BLOCK_SIZE), 1);
				MatrixMulGPUCalc << < DimGrid2, DimBlock,0,t.stream >> > (t.d_a, t.d_b, t.d_c, block_size, tile_size, block_size);

				cudaMemcpyAsync(
					t.h_c, t.d_c, block_size * block_size * sizeof(double), cudaMemcpyDeviceToHost, t.stream
				);

				t._23.signal();
			}
		}
	}
}

void mat_mul::copy_back(void)
{
	if (tiles.empty())
		return;

	size_t stream_id{ tiles.size() - 1 };
	// loop through all column tiles for A
	for(uint i = 0; i < num_A_columns; ++i) {
		//loop over row tiles per A column tile
		for(uint j = 0; j < num_A_rows; ++j) {
			//loop over column tile per B row tile
			for(uint k = 0; k < num_B_columns; ++k) {
				stream_id = next_stream_id(stream_id);

				auto& t = tiles[stream_id];

				t._23.wait();

				cudaStreamSynchronize(t.stream);

				uint offset_rows = j * block_size;
				uint offset_col_a = i * tile_size;
				uint offset_col_b = k * block_size;

				uint diff_rows = std::min((j + 1) * block_size, rowA) - offset_rows;
				uint diff_col_a = std::min((i + 1) * tile_size, colA) - offset_col_a;
				uint diff_col_b = std::min((k + 1) * block_size, colB) - offset_col_b;

				//copy x1 c
				for (uint row = 0; row < diff_rows; ++row) {
					for (uint col = 0; col < diff_col_b; ++col) {
						out[(row + offset_rows) * colB + col + offset_col_b] += t.h_c[row * block_size + col];
					}
				}
				
				t._31.signal();
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
	//initialization
	memset(out, 0, sizeof(double) * rowA* colB);

	std::vector<tile_data> tiles{(size_t)numberOfStreams}; // number of streams

	//allocate resources for streams
	for(auto& t: tiles) {

		//input host memory
		checkCudaErrors(cudaHostAlloc((void **) &(t.h_a), blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
		checkCudaErrors(cudaHostAlloc((void **) &(t.h_b), blockSize * tileSize * sizeof(double), cudaHostAllocMapped));

		// output host memory
		checkCudaErrors(cudaHostAlloc((void **) &(t.h_c), blockSize * blockSize * sizeof(double), cudaHostAllocMapped));
		
		// input device memory
		checkCudaErrors(cudaMalloc((void **) &(t.d_a), blockSize * tileSize * sizeof(double)));
		checkCudaErrors(cudaMalloc((void **) &(t.d_b), blockSize * tileSize * sizeof(double)));

		// output device memory
		checkCudaErrors(cudaMalloc((void **) &(t.d_c), blockSize * blockSize  * sizeof(double)));
		checkCudaErrors(cudaStreamCreateWithFlags(&(t.stream), cudaStreamNonBlocking));

		t.init_once = false;
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

	
	//stream resources
	App.tiles = std::move(tiles);

	//tile stuff
	App.num_A_columns = (colA - 1) / tileSize + 1;
	App.num_A_rows = (rowA - 1) / blockSize + 1;
	App.num_B_columns = (colB - 1) / blockSize + 1;

	
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

	// wait for threads to finish
	for(auto& t: threads) {
		t.join();
	}

	//make sure to be done with everything then dealloc
	checkCudaErrors(cudaDeviceSynchronize());

	//deallocate
	for(auto& t : App.tiles) {
		checkCudaErrors(cudaFreeHost(t.h_a));
		checkCudaErrors(cudaFreeHost(t.h_b));
		checkCudaErrors(cudaFreeHost(t.h_c));

		checkCudaErrors(cudaFree(t.d_a));
		checkCudaErrors(cudaFree(t.d_b));
		checkCudaErrors(cudaFree(t.d_c));
		checkCudaErrors(cudaStreamDestroy(t.stream));
	}

	
}