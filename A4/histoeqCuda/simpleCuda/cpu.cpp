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

#include <assert.h>
#include "histogram_common.h"

extern "C" void histogram256CPU(
	uint *h_Histogram,
	float *h_HistogramCdf,
	void *h_DataIn,
	void *h_DataOut,
	uint byteCount,
	uint imgWidth,
	uint imgHeight,
	uint imgChannels
)
{
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        h_Histogram[i] = 0;

    assert(sizeof(uint) == 4 && (byteCount % 4) == 0);
    for (int i = 0; i < (byteCount / 4); i++)
    {
        uint data = ((uint *)h_DataIn)[i];
			h_Histogram[(data >> 0) & 0xFFU]++;
			h_Histogram[(data >> 8) & 0xFFU]++;
			h_Histogram[(data >> 16) & 0xFFU]++;
			h_Histogram[(data >> 24) & 0xFFU]++;
    }

	for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		h_HistogramCdf[i] = 0;

	for (int i = 0; i < HISTOGRAM256_BIN_COUNT; i++) {
		float sum = 0.0;
		for (int j = 0; j < i + 1; j++) {
			sum += (float)PROBABILITY((float)h_Histogram[j],imgWidth,imgHeight);
		}
		h_HistogramCdf[i] += sum;
	}

	float cdfMin = h_HistogramCdf[0];
	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			int i = row*imgWidth + col;
			*((unsigned char*)h_DataOut + i) = (unsigned char)CORRECT_COLOR(h_HistogramCdf[*((unsigned char*)h_DataIn + i)], cdfMin);
		}
	}
}
