/*
* Copyright 2018 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"
#include <stdint.h>
#include "ppm.h"

const static char *sSDKsample = "[histogram equalization]\0";

int main(int argc, char **argv)
{
	uint  *h_HistogramCPU;
	uint  *h_HistogramGPU;
	uchar *d_Data;
	uchar *d_DataOut;
	uint  *d_Histogram;
	StopWatchInterface *hTimer = NULL;
	int PassFailFlag = 1;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 4) {
		printf
		("Usage: histogram <InFile.pgm> <CPU Output File> <GPU Output File> \n\n");
		exit(0);
	}

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	printf("...reading input data\n");
	printf("...allocating CPU memory.\n");

	Image_t *imgHandle;
	imgHandle = readPPM(argv[1]);
	if (!imgHandle) {
		printf("%s file not exists\n\n", argv[1]);
		exit(0);
	}
	uchar *h_Data = imgHandle->data;
	int imageWidth = imgHandle->width;
	int imageHeight = imgHandle->height;

	int imageChannels = 1;//grey
	int colorDepth = 255;
	double dAvgSecs;
	uint byteCount = imageWidth*imageHeight*imageChannels;

	uchar *output_image = (uchar *)malloc(sizeof(uchar) * byteCount);

	h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT*sizeof(uint));

	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_DataOut, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));

//	printf("Initializing histogram...\n");
	initHistogram256();
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	histogram256(d_Histogram, d_Data, d_DataOut, byteCount,
				imageWidth, imageHeight, imageChannels );

//	printf("\nValidating GPU results...\n");
//	printf(" ...reading back GPU results\n");
	checkCudaErrors(cudaMemcpy(output_image, d_DataOut, byteCount, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT *sizeof(uint), cudaMemcpyDeviceToHost));

//	printf("Shutting down histogram equalization...\n\n\n");
	sdkStopTimer(&hTimer);
	closeHistogram256();

	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) ;
	printf("histogram256() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1);

	printf("Shutting down...\n");

	checkCudaErrors(cudaFree(d_Histogram));
	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_DataOut));
	
	Image_t *imgOutput = new Image_t;
	imgOutput->width = imageWidth;
	imgOutput->height = imageHeight;
	imgOutput->data = output_image;
	writePPM(argv[3], imgOutput);
	free(output_image);

	cudaDeviceReset();

	h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	float* h_HistogramCPUCdf = (float*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(float));
	uchar *output_imageCPU = (uchar *)malloc(sizeof(uchar) * byteCount);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
//	printf(" ...histogram256CPU()\n");
	histogram256CPU(
		h_HistogramCPU,
		h_HistogramCPUCdf,
		h_Data,
		output_imageCPU,
		byteCount,
		imageWidth,
		imageHeight,
		imageChannels
	);
//	printf(" ...histogram256CPU() finishing\n");

	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); // (double)numRuns;
	printf("histogram256CPU() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("histogram256CPU, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount);
	printf("Shutting down...\n");

	sdkDeleteTimer(&hTimer);

	printf(" ...comparing the results\n");
	PassFailFlag = 1;
	for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++) 
		if (h_HistogramGPU[i] != h_HistogramCPU[i] )
		{
			PassFailFlag = 0;
			break;
		}

	printf(PassFailFlag ? " ...CPU and GPU histogram equalization results match\n\n" : " ***CPU and GPU histogram equalization results do not match!!!***\n\n");

	imgOutput->width = imageWidth;
	imgOutput->height = imageHeight;
	imgOutput->data = output_imageCPU;
	writePPM(argv[2], imgOutput);
	free(output_imageCPU);
	delete imgOutput;
	free(h_HistogramCPUCdf);

	free(h_HistogramGPU);
	free(h_HistogramCPU);
	free(h_Data);

	printf("%s - Test Summary\n", sSDKsample);
#if 0
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		return -1;
	}

	printf("Test passed\n");
#endif

	return 0;
}
