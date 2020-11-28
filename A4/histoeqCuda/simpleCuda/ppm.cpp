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
#include "ppm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define MAX_CHARS_PER_LINE (1<<8) 

void skipComment(FILE *fr, char *line)
{
//	char line[MAX_CHARS_PER_LINE] = { 0 };

	while (fgets(line, MAX_CHARS_PER_LINE-1, fr) != NULL) {
		char *p = line;
		size_t len = strlen(line);
		printf("%d \n", len);
		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = 0;    // strip newline or carriage rtn    
		while (isspace(*p))    // advance to first non-whitespace  
			p++;
		// skip lines beginning with '#' or '@' or blank lines 
		if (*p == '#' || *p == '@' || !*p) {
			printf("%s\n", line);
			break;
		}
	}

	return;
}

Image_t* readPPM(const char* fileName) 
{
	int width;
	int height;
	int maximum;
	// open the file to read just the header reading
	FILE* fr = fopen(fileName, "rb");
	if (!fr) return NULL;
	char *pSix = new char[MAX_CHARS_PER_LINE];
	// formatted read of header
	fscanf(fr, "%s\n", pSix);
	if (pSix == NULL)	return NULL;
	// check to see if it's a PGM image file
	if (strncmp(pSix, "P5", 10) != 0) {
		printf("Sorry they are not the same\n");
	}
	else {
		//printf("They are the same\n");
	}
	printf("reading the comment\n");
#if 0
	if (fgets(pSix, MAX_CHARS_PER_LINE - 1, fr) != NULL) {
		printf("%s", pSix);
	}
#endif
	///better to check whether this is a comment line
	fscanf(fr, "%d %d\n", &width, &height);
	fscanf(fr, "%d\n", &maximum);

	// check to see if they were stored properly
	//printf("PSix: %s\n", pSix);
	printf("Width: %d\n", width);
	printf("Height: %d\n", height);
	printf("maximum: %d\n", maximum);

	//support gray only
	int size = width * height;

	// allocate array for pixels
	unsigned char* pixels = new unsigned char[size];
	
	fread(pixels, width * sizeof(unsigned char), height, fr);

	// close file
	fclose(fr);

	Image_t* img = new Image_t;
	img->channels = 1;
	img->height = height;
	img->width = width;
	img->data = pixels;
	return img;

} // end of readPPM 


void writePPM(const char* fileName, Image_t *img) 
{
	// open the file to read just the header reading
	FILE* fr = fopen(fileName, "wb+");

	// write formatted header
	fprintf(fr, "%s\n", "P5");
	fprintf(fr, "%s\n", "#Created via PPM Export");
	fprintf(fr, "%d %d\n", img->width, img->height);
	fprintf(fr, "%d\n", 255);//depth

	int size = img->width * img->height;

	// write into file
	if  (fwrite(img->data, sizeof(unsigned char), size, fr)!=size) {
		printf("Failed to write PPM file\n");
		return;
	} // end of for loop

	  // close file
	fclose(fr);

	return;

} // end of writePPM 