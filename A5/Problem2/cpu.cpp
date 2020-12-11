/*Start Header
******************************************************************/
/*!
\file cpu.cpp
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date October 18, 2020
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
#include <iomanip>


static inline double frand() noexcept
{
	// normalized value
	return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
}

extern "C" void RandomizeMatrix(double* matrix, uint row, uint col)
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
	const double* inA, 
	const double* inB, 
	double* out, 
	uint rowA, 
	uint colA, 
	uint colB)
{
	uint rowB = colA; 

	for (uint i = 0; i < rowA; ++i)
	{
		for (uint j = 0; j < colB; ++j)
		{
			out[i * colB + j] = 0.0f;
			for (uint k = 0; k < colA; ++k)
			{
				out[i * colB + j] += inA[i*colA + k] * inB[k*colB + j];
			}
		}
	}
}

extern "C" double* LoadMatrix(const char* fileName, uint* row, uint* col)
{
	// load matrix here
	std::ifstream file{ fileName };

	if (!file)
	{
		printf("file does not exist! %s\n", fileName);
		std::exit(0);
	}

	//uint i = 0;
	uint width, height;

	file >> height >> width;
	printf("height = %d width = %d\n", height, width);

	double* matrix = new double[width * height];

	for (uint i = 0; i < width*height; ++i)
	{
		file >> matrix[i];
	}

	*col = width;
	*row = height;

	printf(
		"read random matrix from file %s, Row = %d, Col = %d\n",
		fileName, *row, *col
	);

	return matrix;
}


extern "C" void SaveMatrix(
	const char* fileName,
	const double* matrix,
	uint row,
	uint col
)
{
	std::ofstream file{ fileName };

	file << row << " " << col << std::endl;

	// to match sample outputs
	file << std::fixed << std::setprecision(20);

	for (uint i = 0; i < row; ++i)
	{
		for (uint j = 0; j < col; ++j)
		{
			file << matrix[i * col + j];
			if (j + 1 < col)
				file << " ";
		}
		if (i + 1 < row)
			file << std::endl;
	}
}