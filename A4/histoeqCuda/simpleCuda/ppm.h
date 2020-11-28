#pragma once
#ifndef __PPM_H
#define __PPM_H

typedef struct st_Image_t {
	int width;
	int height;
	int channels;
	int pitch;
	unsigned char *data;
} Image_t;
Image_t* readPPM(const char* fileName);
void writePPM(const char* fileName, Image_t *img);

#endif