/*Start Header
******************************************************************/
/*!
\file matrix.h
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date December 14, 2020
\brief
	matrix cpu/gpu header functions
Copyright (C) 2020 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#ifndef _MATRIX_H_
#define _MATRIX_H_

 

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

using cand_t = double;

////////////////////////////////////////////////////////////////////////////////
// Reference CPU version  
////////////////////////////////////////////////////////////////////////////////
extern "C" void RandomizeMatrix(
	double* pointIn,
	uint col,
	uint row
);

extern "C" void MatrixMulCPU(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA, 
	uint colA, 
	uint colB
);

extern "C" double* LoadMatrix(
	const char* fileName, 
	uint * row,
	uint * col
);

extern "C" void SaveMatrix(
	const char* fileName,
	const double* matrix,
	uint row,
	uint col
);


////////////////////////////////////////////////////////////////////////////////
// GPU version + shm
////////////////////////////////////////////////////////////////////////////////

extern "C" void MatrixMulGPU(
	const double* inA,
	const double* inB,
	double* out,
	uint rowA,
	uint colA,
	uint colB
);

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
);

#endif
