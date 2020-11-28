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

#include <helper_cuda.h>
#include "histogram_common.h"

#define BLOCK_SIZE 32
///you may use the following declarations
__global__ void histogram(	unsigned char *input, 
							unsigned int *output,
							int width, 
							int height) {
}
__global__ void cdfScan(unsigned int *input, 
						float *output,
						float width, 
						float height) {
}


__global__ void applyHistogram(	unsigned char *input1, 
								float *input2,
								float *input3, 
								unsigned char *output,
								int width, 
								int height, 
								int channels) 
{
}
////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////

//Internal memory allocation
extern "C" void initHistogram256(void)
{
}

//Internal memory deallocation
extern "C" void closeHistogram256(void)
{
}

extern "C" void histogram256(
	uint *d_Histogram,
	void *d_DataIn,
	void *d_DataOut,
	uint byteCount,
	uint imgWidth,
	uint imgHeight,
	uint imgChannels
)
{

}
