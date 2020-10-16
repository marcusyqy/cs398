/*Start Header
******************************************************************/
/*!
\file cpu.cpp
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date Sept 17, 2020
\brief
	cpu & gpu forward declared functions
Copyright (C) 2020 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#ifndef HEAT_H
#define HEAT_H

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// Reference CPU version  
////////////////////////////////////////////////////////////////////////////////
extern "C" void initPoints(
	float *pointIn,
	float *pointOut,
	uint nRowPoints
);

extern "C" void heatDistrCPU(
	float *pointIn,
	float *pointOut,
	uint nRowPoints,
	uint nIter
);

////////////////////////////////////////////////////////////////////////////////
// GPU version 
////////////////////////////////////////////////////////////////////////////////

extern "C" void heatDistrGPU(
	float *d_DataIn,
	float *d_DataOut,
	uint nRowPoints,
	uint nIter
);


extern "C" void batchHeatDistrGPU(
	float* d_FinalData,
	float* d_TempBuffer1,
	float* d_TempBuffer2,
	uint nRowPoints,
	uint nBatchColPoints,
	uint nIter
);
#endif
