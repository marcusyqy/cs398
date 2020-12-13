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
	cudaEvent_t events[3] { nullptr, nullptr, nullptr};
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
	
	size_t stream_id;
};

size_t mat_mul::next_stream_id(size_t id)
{
	size_t i = (id + 1) % tiles.size();
	cudaStreamSynchronize(tiles[i].stream);
	//size_t i{}; 
	//for(;;) {
	//	bool operations_pending = cudaStreamQuery(tiles[i].stream) == cudaErrorNotReady;
	//	if(!operations_pending)
	//		break;
	//	i = (i + 1)%tiles.size();
	//}

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

				if (!tiles[stream_id].init_once) {
					tiles[stream_id].init_once = true;
				}
				else {
					tiles[stream_id]._31.wait();
				}
				

				uint offset_rows = j * block_size;
				uint offset_col_a = i * tile_size;
				uint offset_col_b = k * block_size;

				uint rows = std::min((j + 1) * block_size, rowA) - offset_rows;
				uint col_a = std::min((i + 1)*tile_size, colA) - offset_col_a;
				uint col_b = std::min((k + 1)*block_size, colB) - offset_col_b;


				cudaMemcpy2D(
					(void*)tiles[stream_id].h_a,
					(size_t)tile_size,
					(const void*)(inA + colA * offset_rows + offset_col_a),
					(size_t)colA,
					(size_t)col_a * sizeof(double),
					(size_t)rows,
					cudaMemcpyHostToHost
				);

				cudaMemcpy2D(
					(void*)tiles[stream_id].h_b,
					(size_t)block_size,
					(const void*)(inB + (colB)*h + w),
					(size_t)colB,
					(size_t)block_size * sizeof(double),
					(size_t)tile_size,
					cudaMemcpyHostToHost
				);

				cudaMemcpy2D(
					(void*)tiles[stream_id].h_c,
					(size_t)block_size,
					(const void*)(out + (colB)*h + w),
					(size_t)colB,
					(size_t)block_size * sizeof(double),
					(size_t)tile_size,
					cudaMemcpyHostToHost
				);

				tiles[stream_id]._12.signal();
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

				tiles[stream_id]._12.wait();


				tiles[stream_id]._23.signal();
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

				tiles[stream_id]._23.wait();


				//copy x3


				tiles[stream_id]._31.signal();
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
	// // initializing... 
	// // allocate pinned memory
	// double* h_A;
	// double* h_B;
	// double* h_C;

	// // input host memory
	// checkCudaErrors(cudaHostAlloc((void **) &h_A, blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
	// checkCudaErrors(cudaHostAlloc((void **) &h_B, blockSize * tileSize * sizeof(double), cudaHostAllocMapped));
	// // output host memory
	// checkCudaErrors(cudaHostAlloc((void **) &h_C, blockSize * blockSize * sizeof(double), cudaHostAllocMapped));
	
	// // allocate device local memory
	// double* d_A;
	// double* d_B;
	// double* d_C;
	// // input device memory
	// checkCudaErrors(cudaMalloc((void **) &d_A, blockSize * tileSize * sizeof(double)));
	// checkCudaErrors(cudaMalloc((void **) &d_B, blockSize * tileSize * sizeof(double)));
	// // output device memory
	// checkCudaErrors(cudaMalloc((void **) &d_C, blockSize * blockSize  * sizeof(double)));

	// cudaEvent_t host_memory_avail, host_memory_ready;
	// checkCudaErrors(cudaEventCreate(&host_memory_avail));
	// checkCudaErrors(cudaEventCreate(&host_memory_ready));

	std::vector<tile_data> tiles{(size_t)numberOfStreams};
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

		checkCudaErrors(cudaEventCreate(&(t.events[0])));
		checkCudaErrors(cudaEventCreate(&(t.events[1])));
		checkCudaErrors(cudaEventCreate(&(t.events[2])));

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

	// streams
	//App.streams = streams.data();
	//App.num_streams = streams.size();

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

	for(auto& t: threads) {
		t.join();
	}


	for(auto& t : App.tiles) {
		checkCudaErrors(cudaFreeHost(t.h_a));
		checkCudaErrors(cudaFreeHost(t.h_b));
		checkCudaErrors(cudaFreeHost(t.h_c));

		checkCudaErrors(cudaFree(t.d_a));
		checkCudaErrors(cudaFree(t.d_b));
		checkCudaErrors(cudaFree(t.d_c));
		checkCudaErrors(cudaStreamDestroy(t.stream));

		checkCudaErrors(cudaEventDestroy(t.events[0]));
		checkCudaErrors(cudaEventDestroy(t.events[1]));
		checkCudaErrors(cudaEventDestroy(t.events[2]));
	}

	
}