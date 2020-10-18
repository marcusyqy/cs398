/*Start Header
******************************************************************/
/*!
\file matrix.h
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date October 18, 2020
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

////////////////////////////////////////////////////////////////////////////////
// Reference CPU version  
////////////////////////////////////////////////////////////////////////////////
extern "C" void RandomizeMatrix(
	float* pointIn,
	uint col,
	uint row
);

extern "C" void MatrixMulCPU(
	const float* inA, 
	const float* inB, 
	float* out,
	uint rowA, 
	uint colA, 
	uint colB
);

extern "C" float* LoadMatrix(
	const char* fileName, 
	uint * row,
	uint * col
);

extern "C" float* SaveMatrix(
	const char* fileName,
	const float* matrix,
	uint row,
	uint col
);


////////////////////////////////////////////////////////////////////////////////
// GPU version + shm
////////////////////////////////////////////////////////////////////////////////

extern "C" void MatrixMulGPU(
	const float* inA,
	const float* inB,
	float* out,
	uint rowA,
	uint colA,
	uint colB
);

#endif
