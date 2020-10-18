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

#include "matrix.h"
#include <stdio.h>

#include <fstream>


static inline float frand() noexcept
{
	// normalized value
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

extern "C" void RandomizeMatrix(float* matrix, uint row, uint col)
{
	for (uint i{}; i < col; ++i)
	{
		for (uint j{}; j < row; ++j)
		{
			matrix[j * col + i] = frand();
		}
	}
}


extern "C" void MatrixMulCPU(
	const float* inA, 
	const float* inB, 
	float* out, 
	uint rowA, 
	uint colA, 
	uint colB)
{
	uint rowB = colA; 

	for (uint i = 0; i < rowA; ++i)
	{
		for (uint j = 0; j < colB; ++j)
		{
			out[j * colB + i] = 0.0f;
			for (uint k = 0; k < colA; ++k)
			{
				out[j * colB + i] += inA[i*colA + k] * inB[k*colB + j];
			}
		}
	}
}

extern "C" float* LoadMatrix(const char* fileName, uint* row, uint* col)
{
	// load matrix here




	printf(
		"height = %d width = %d\n"
		"read random matrix from file %s, Row = %d, Col = %d\n",
		*row, *col, fileName, *row, *col
	);
}


extern "C" float* SaveMatrix(
	const char* fileName,
	const float* matrix,
	uint row,
	uint col
)
{

}