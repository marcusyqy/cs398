/*Start Header
******************************************************************/
/*!
\file bmp.h
\author Yong Quanyi Marcus, yong.q, 390005818
\par email: yong.q\@digipen.edu
\date November 25, 2020
\brief
    declaration of bmp read
Copyright (C) 2020 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/
#ifndef _BMP
#define _BMP

#pragma pack (push, 1)
typedef struct {
    char id1;
    char id2;
    unsigned int file_size;
    unsigned int reserved;
    unsigned int bmp_data_offset;
    unsigned int bmp_header_size;
    unsigned int width;
    unsigned int height;
    unsigned short int planes;
    unsigned short int bits_per_pixel;
    unsigned int compression;
    unsigned int bmp_data_size;
    unsigned int h_resolution;
    unsigned int v_resolution;
    unsigned int colors;
    unsigned int important_colors;
} bmp_header;
#pragma pack(pop)
extern void bmp_read(char *str, bmp_header *header, unsigned char **data);
extern void bmp_write(char *str, bmp_header *header, unsigned char *data);

#endif
