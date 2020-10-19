/*Start Header
******************************************************************/
/*!
\file cpu.cpp
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date Sept 17, 2020
\brief
	cpu computing functions
Copyright (C) 2020 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

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
