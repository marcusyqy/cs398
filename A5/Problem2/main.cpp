/*Start Header
******************************************************************/
/*!
\file main.cpp
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



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//add into Project/Properties/CUDA C/C++ Additional Include Directories
//C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2\common\inc;
// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

#include "matrix.h"
//#include "kernel.cu"

// project include
#include <stdint.h>

#include <string>
#define epsilon 1.0e-3 
#define error_epsilon 1.0e-6

const static char *sSDKsample = "[Matrix Multiplication]\0";


int main(int argc, char** argv)
{
	
	StopWatchInterface* hTimer = NULL;
	int PassFailFlag = 1;
	//uint nIter;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc < 4) {
		printf
		(
			"Usage [Mode 1]: mm ARows ACols BCols\n\n"
			"Usage[Mode 2]: mm input1 input2 output[tileWidth(Default = 16)]\n\n"
		);
		exit(0);
	}


	std::string arg1 = argv[1];
	std::string arg2 = argv[2];
	std::string arg3 = argv[3];
	std::string arg4{};
	std::string arg5{};
	std::string arg6{};

	bool a5 = false;
	if (argc == 7) {
		printf("\nENABLED A5\n\n");
		arg4 = argv[4];
		arg5 = argv[5];
		arg6 = argv[6];
		a5 = (uint)std::stoi(arg6) == 1 ? false : true;
	}


	double* AMat{ nullptr }, *BMat{ nullptr }, *Output{ nullptr };

	if ((arg1.find_first_of('.') == std::string::npos) &&
	    (arg2.find_first_of('.') == std::string::npos) &&
	    (arg3.find_first_of('.') == std::string::npos))
	{
		// generate matrix
		printf("create random matricies\n\n");

		//initialize
		uint ARow = (uint)std::stoi(arg1);
		uint ACol = (uint)std::stoi(arg2);
		uint BRow = ACol; // for sanity.
		uint BCol = (uint)std::stoi(arg3);

		uint CRow = ARow;
		uint CCol = BCol;

		AMat = new double[ARow * ACol];
		BMat = new double[BRow * BCol];
		Output = new double[CRow * CCol];

		sdkCreateTimer(&hTimer);

		//randomize first two
		RandomizeMatrix(AMat, ACol, ARow);
		RandomizeMatrix(BMat, BCol, BRow);

		//save both
		SaveMatrix("Input0.raw", AMat, ARow, ACol);
		SaveMatrix("Input1.raw", BMat, BRow, BCol);

		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// maybe calculate cpu side
		MatrixMulCPU(AMat, BMat, Output, ARow, ACol, BCol);

		sdkStopTimer(&hTimer);
		float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); 

		printf("CPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(BCol* ARow * sizeof(double)) * 1.0e-6) / dAvgSecs);
		printf("CPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
			(1.0e-6 * (double)(BCol * ARow * sizeof(double)) / dAvgSecs), dAvgSecs, BCol * ARow * sizeof(double));

		SaveMatrix("Output.raw", Output, ARow, BCol);


		double* AMatGpu{ nullptr }, * BMatGpu{ nullptr }, * OutputGpu{ nullptr }, * OutputCmp{ nullptr };

		// set logfile name and start logs
		printf("[%s] - Starting...\n", sSDKsample);

		//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
		int dev = findCudaDevice(argc, (const char**)argv);

		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
			deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

		printf("Initializing data...\n");
		printf("...reading input data\n");
		sdkCreateTimer(&hTimer);
		printf("...allocating CPU memory.\n");

		printf("Initializing data...\n");
		printf("...reading input data\n");
		printf("...allocating GPU memory and copying input data\n\n");

		checkCudaErrors(cudaMalloc((void**)&AMatGpu, ARow * ACol * sizeof(double)));
		checkCudaErrors(cudaMalloc((void**)&BMatGpu, BRow * BCol * sizeof(double)));
		checkCudaErrors(cudaMalloc((void**)&OutputGpu, ARow * BCol * sizeof(double)));

		//initPoints(h_DataGPU, h_DataGPU, nRowPoints);
		checkCudaErrors(cudaMemcpy(AMatGpu, AMat, ARow * ACol * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(BMatGpu, BMat, BRow * BCol * sizeof(double), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaDeviceSynchronize());

		sdkResetTimer(&hTimer);

		sdkStartTimer(&hTimer);
		MatrixMulGPU(AMatGpu, BMatGpu, OutputGpu, ARow, ACol, BCol);
		sdkStopTimer(&hTimer);

		dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
		printf("GPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(BCol * ARow * sizeof(double)) * 1.0e-6) / dAvgSecs);
		printf("GPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
			(1.0e-6 * (double)(BCol * ARow * sizeof(double)) / dAvgSecs), dAvgSecs, BCol * ARow * sizeof(double));

		sdkDeleteTimer(&hTimer);

		OutputCmp = new double[CRow * CCol];
		checkCudaErrors(cudaMemcpy(OutputCmp, OutputGpu, CRow * CCol * sizeof(double), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaDeviceSynchronize());
		for (uint i = 0; i < CRow * CCol; ++i)
		{
			if (std::abs(Output[i] - OutputCmp[i]) > error_epsilon && std::abs(Output[i] - OutputCmp[i]) > error_epsilon * std::abs(OutputCmp[i]))
			{
				PassFailFlag = 0;
				printf("Test failed at %d, with %f, %f", i, Output[i], OutputCmp[i]);
				break;
			}
		}

		if (PassFailFlag)
			printf("\n\nTest PASSED");

		//free relevant stuff
		checkCudaErrors(cudaFree(AMatGpu));
		checkCudaErrors(cudaFree(BMatGpu));
		checkCudaErrors(cudaFree(OutputGpu));
		delete[] OutputCmp;

		sdkDeleteTimer(&hTimer);

	}
	else if((arg1.find_first_of('.') != std::string::npos) &&
		(arg2.find_first_of('.') != std::string::npos) &&
		(arg3.find_first_of('.') != std::string::npos))
	{
		// calculate matrix
		// read all 3 arguments
		uint ARow, ACol, BRow, BCol, CRow, CCol;

		double* AMatGpu{ nullptr }, * BMatGpu{ nullptr }, * OutputGpu{ nullptr }, * OutputCmp{ nullptr };

		AMat = LoadMatrix(arg1.c_str(), &ARow, &ACol);
		BMat = LoadMatrix(arg2.c_str(), &BRow, &BCol);

		// set logfile name and start logs
		printf("[%s] - Starting...\n", sSDKsample);

		//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
		int dev = findCudaDevice(argc, (const char**)argv);

		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
			deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

		printf("Initializing data...\n");
		printf("...reading input data\n");
		sdkCreateTimer(&hTimer);
		printf("...allocating CPU memory.\n");

		printf("Initializing data...\n");
		printf("...reading input data\n");
		printf("...allocating GPU memory and copying input data\n\n");

		if (!a5) {

			checkCudaErrors(cudaMalloc((void**)&AMatGpu, ARow* ACol * sizeof(double)));
			checkCudaErrors(cudaMalloc((void**)&BMatGpu, BRow* BCol * sizeof(double)));
			checkCudaErrors(cudaMalloc((void**)&OutputGpu, ARow* BCol * sizeof(double)));

			//initPoints(h_DataGPU, h_DataGPU, nRowPoints);
			checkCudaErrors(cudaMemcpy(AMatGpu, AMat, ARow* ACol * sizeof(double), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(BMatGpu, BMat, BRow* BCol * sizeof(double), cudaMemcpyHostToDevice));

			checkCudaErrors(cudaDeviceSynchronize());

			sdkResetTimer(&hTimer);

			sdkStartTimer(&hTimer);
			MatrixMulGPU(AMatGpu, BMatGpu, OutputGpu, ARow, ACol, BCol);
			sdkStopTimer(&hTimer);

			float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
			printf("GPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(BCol* ARow * sizeof(double)) * 1.0e-6) / dAvgSecs);
			printf("GPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
				(1.0e-6 * (double)(BCol * ARow * sizeof(double)) / dAvgSecs), dAvgSecs, BCol* ARow * sizeof(double));

			Output = LoadMatrix(arg3.c_str(), &CRow, &CCol);

			OutputCmp = new double[CRow * CCol];

			checkCudaErrors(cudaMemcpy(OutputCmp, OutputGpu, CRow* CCol * sizeof(double), cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaDeviceSynchronize());
			for (uint i = 0; i < CRow * CCol; ++i)
			{
				if (std::abs(Output[i] - OutputCmp[i]) > error_epsilon && std::abs(Output[i] - OutputCmp[i]) > error_epsilon * std::abs(OutputCmp[i]))
				{
					PassFailFlag = 0;
					printf("Test failed at %d, with %f, %f", i, Output[i], OutputCmp[i]);
					break;
				}
			}

			if (PassFailFlag)
				printf("\n\nTest PASSED");

			//free relevant stuff
			checkCudaErrors(cudaFree(AMatGpu));
			checkCudaErrors(cudaFree(BMatGpu));
			checkCudaErrors(cudaFree(OutputGpu));
			delete[] OutputCmp;
			sdkDeleteTimer(&hTimer);
		}
		else {
			uint blockSize = (uint)std::stoi(arg4);
			uint tileSize = (uint)std::stoi(arg5);
			uint numberOfStreams = (uint)std::stoi(arg6);

			// ... 
			//checkCudaErrors(cudaMalloc((void**)&AMatGpu, ARow * ACol * sizeof(double)));
			//checkCudaErrors(cudaMalloc((void**)&BMatGpu, BRow * BCol * sizeof(double)));
			////allocate on cpu side instead
			//checkCudaErrors(cudaHostAlloc((void**)&OutputGpu, ARow* BCol * sizeof(double), cudaHostAllocDefault));

			//AMatGpu = (double*)malloc(ARow * ACol * sizeof(double));
			//BMatGpu = (double*)malloc(BRow * BCol * sizeof(double));
			//OutputGpu = (double*)malloc(ARow * BCol * sizeof(double));

			OutputCmp = new double[CRow * CCol];

			//checkCudaErrors(cudaMemcpy(AMatGpu, AMat, ARow * ACol * sizeof(double), cudaMemcpyHostToDevice));
			//checkCudaErrors(cudaMemcpy(BMatGpu, BMat, BRow * BCol * sizeof(double), cudaMemcpyHostToDevice));

			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
			MatrixMulGPUStream(AMat, BMat, OutputCmp, ARow, ACol, BCol, blockSize, tileSize, numberOfStreams);
			sdkStopTimer(&hTimer);

			float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
			printf("Streams GPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(BCol * ARow * sizeof(double)) * 1.0e-6) / dAvgSecs);
			printf("Streams GPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
				(1.0e-6 * (double)(BCol * ARow * sizeof(double)) / dAvgSecs), dAvgSecs, BCol * ARow * sizeof(double));

			Output = LoadMatrix(arg3.c_str(), &CRow, &CCol);

			//OutputCmp = OutputGpu;//new double[CRow * CCol];

			//checkCudaErrors(cudaMemcpy(OutputCmp, OutputGpu, CRow * CCol * sizeof(double), cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaDeviceSynchronize());
			for (uint i = 0; i < CRow * CCol; ++i)
			{
				if (std::abs(Output[i] - OutputCmp[i]) > error_epsilon && std::abs(Output[i] - OutputCmp[i]) > error_epsilon * std::abs(OutputCmp[i]))
				{
					PassFailFlag = 0;
					printf("Test failed at %d, with %f, %f", i, Output[i], OutputCmp[i]);
					break;
				}
			}

			if (PassFailFlag)
				printf("\n\nTest PASSED");

			//free relevant stuff
			//checkCudaErrors(cudaFree(AMatGpu));
			//checkCudaErrors(cudaFree(BMatGpu));
			//checkCudaErrors(cudaFreeHost(OutputGpu));
			delete[] OutputCmp;
			sdkDeleteTimer(&hTimer);
		}
	}
	else
	{
		printf(
			"invalid parameters!\n\n"
			"Usage [Mode 1]: mm ARows ACols BCols\n\n"
			"Usage[Mode 2]: mm input1 input2 output[tileWidth(Default = 16)]\n\n"
		);
		exit(0);
	}

	delete[] AMat;
	delete[] BMat;
	delete[] Output;
	
	return 0;
}
