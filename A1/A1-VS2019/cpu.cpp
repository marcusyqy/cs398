/*
* Copyright 2018 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms 
* is strictly prohibited.
*/
#include "heat.h"
#include <stdio.h>
extern "C" void initPoints(
	float *pointIn,
	float *pointOut,
	uint nRowPoints
)
{
	for (uint i = 0; i < nRowPoints; ++i)
	{
		for (uint j = 0; j < nRowPoints; ++j)
		{
			float value = 0.0f;
			uint index = j * nRowPoints + i;

			if (i == 0 || i == nRowPoints - 1)
			{
				if (j >= 10 && j <= 30)
				{
					value = 65.56f;
				}
				else
				{
					value = 26.67f;
				}
			}

			if (j == 0 || j == nRowPoints - 1)
			{
				value = 26.67f;
			}

			pointIn[index] = value;
			pointOut[index] = value;
		}
	}
}

extern "C" void heatDistrCPU(
	float *pointIn,
	float *pointOut,
	uint nRowPoints,
	uint nIter
)
{
	for (uint iter = 0; iter < nIter; ++iter)
	{
		// main loop
		for (uint j = 0; j < nRowPoints; ++j)
		{
			for (uint i = 0; i < nRowPoints; ++i)
			{
				if (i > 0 && i < nRowPoints - 1 && j > 0 && j < nRowPoints - 1)
				{
					pointOut[j * nRowPoints + i] = (
							pointIn[j * nRowPoints + i - 1] +
							pointIn[j * nRowPoints + i + 1] +
							pointIn[(j - 1 )* nRowPoints + i]+
							pointIn[(j + 1 )* nRowPoints + i]
						) *0.25f;

				}
			}
		}

		// update
		for (uint j = 0; j < nRowPoints; ++j)
		{
			for (uint i = 0; i < nRowPoints; ++i)
			{
				uint index = j * nRowPoints + i;
				pointIn[index] = pointOut[index];
			}
		}
	}
}
