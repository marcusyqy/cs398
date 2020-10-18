/*Start Header
******************************************************************/
/*!
\file main.cpp
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
	float* d_DataIn;
	float* d_DataOut;
	StopWatchInterface* hTimer = NULL;
	int PassFailFlag = 1;
	uint count;
	uint nIter;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 4) {
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

	float* AMat{ nullptr }, *BMat{ nullptr }, *Output{ nullptr };

	if ((arg1.find_first_not_of('.') == std::string::npos) &&
	    (arg2.find_first_not_of('.') == std::string::npos) &&
	    (arg3.find_first_not_of('.') == std::string::npos))
	{
		// generate matrix
		printf("create random matricies\n\n");

		//initialize
		uint ARows = (uint)std::stoi(arg1);
		uint ACols = (uint)std::stoi(arg2);
		uint BRows = ACols; // for sanity.
		uint BCols = (uint)std::stoi(arg3);

		uint CRows = ARows;
		uint CCols = BCols;

		AMat = new float[ARows * ACols];
		BMat = new float[BRows * BCols];
		Output = new float[CRows * CCols];

		sdkCreateTimer(&hTimer);

		//randomize first two
		RandomizeMatrix(AMat, ACols, ARows);
		RandomizeMatrix(BMat, BCols, BRows);

		//save both
		SaveMatrix("Input1.raw", AMat, ARows, ACols);
		SaveMatrix("Input2.raw", BMat, BRows, BCols);

		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		// maybe calculate cpu side
		MatrixMulCPU(AMat, BMat, Output, ARows, ACols, BCols);
		SaveMatrix("Output.raw", Output, ACols, ACols);

		sdkStopTimer(&hTimer);
		float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); 

		printf("CPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float)) * 1.0e-6) / dAvgSecs);
		printf("CPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
			(1.0e-6 * (double)(count * sizeof(float)) / dAvgSecs), dAvgSecs, count * sizeof(float));

		sdkDeleteTimer(&hTimer);

	}
	else if((arg1.find_first_not_of('.') != std::string::npos) &&
		(arg2.find_first_not_of('.') != std::string::npos) &&
		(arg3.find_first_not_of('.') != std::string::npos))
	{
		// calculate matrix
		// read all 3 arguments

		uint ARow, ACol, BRow, BCol, CRow, CCol;

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
		uint nRowPoints = atoi(argv[1]);
		count = nRowPoints*nRowPoints;
		float *h_DataGPU = (float *)malloc(count*sizeof(float));

		printf("...allocating GPU memory and copying input data\n\n");
		checkCudaErrors(cudaMalloc((void **)&d_DataIn, count*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_DataOut, count*sizeof(float)));

		//initPoints(h_DataGPU, h_DataGPU, nRowPoints);
		checkCudaErrors(cudaMemcpy(d_DataIn, h_DataGPU, count*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_DataOut, h_DataGPU, count*sizeof(float), cudaMemcpyHostToDevice));
			
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);


		sdkDeleteTimer(&hTimer);
		Output = LoadMatrix(arg3.c_str(), &CRow, &CCol);


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
