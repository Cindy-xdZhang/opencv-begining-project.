// ---------------------------------------------------------------------------
// bitmap.cpp
//
// handle MS bitmap I/O. For portability, we don't use the data structure 
// defined in Windows.h. However, there is some strange thing, the size of our 
// structure is different from what it should be though we define it in the 
// same way as MS did. So, there is a hack, we use the hardcoded constant, 14, 
// instead of the sizeof to calculate the size of the structure.  You are not 
// supposed to worry about this part. However, I will appreciate if you find 
// out the reason and let me know. Thanks.
// ---------------------------------------------------------------------------
#include<iostream>
#include "bitmap.h"
 
BMP_BITMAPFILEHEADER bmfh; 
BMP_BITMAPINFOHEADER bmih; 

template <class T>
void swapBytes(T* val){
	static int typeSize;
	static char *start, *end;
	typeSize = sizeof(T);

	start = (char*)val;
	end = start + typeSize - 1;

	while (start < end){
		*start ^= *end ^= *start ^= *end;
		start++;
		end--;
	}
}

// Bitmap data returned is (R,G,B) tuples in row-major order.
unsigned char* readBMP(const char* fname, int& width, int& height)
{ 
	FILE* file; 
	BMP_DWORD pos; 
 
	if ( (file=fopen( fname, "rb" )) == NULL )  
		return NULL; 
	 
//	I am doing fread( &bmfh, sizeof(BMP_BITMAPFILEHEADER), 1, file ) in a safe way. :}
	fread( &(bmfh.bfType), 2, 1, file); 
	fread( &(bmfh.bfSize), 4, 1, file); 
	fread( &(bmfh.bfReserved1), 2, 1, file); 
	fread( &(bmfh.bfReserved2), 2, 1, file); 
	fread( &(bmfh.bfOffBits), 4, 1, file); 

#ifndef WIN32
	swapBytes(&(bmfh.bfType));
	swapBytes(&(bmfh.bfSize));
	swapBytes(&(bmfh.bfReserved1));
	swapBytes(&(bmfh.bfReserved2));
	swapBytes(&(bmfh.bfOffBits));
#endif

	pos = bmfh.bfOffBits; 
 
	fread( &bmih, sizeof(BMP_BITMAPINFOHEADER), 1, file ); 

#ifndef WIN32
	swapBytes(&(bmih.biBitCount));
	swapBytes(&(bmih.biClrImportant));
	swapBytes(&(bmih.biClrUsed));
	swapBytes(&(bmih.biCompression));
	swapBytes(&(bmih.biHeight));
	swapBytes(&(bmih.biPlanes));
	swapBytes(&(bmih.biSize));
	swapBytes(&(bmih.biSizeImage));
	swapBytes(&(bmih.biWidth));
	swapBytes(&(bmih.biXPelsPerMeter));
	swapBytes(&(bmih.biYPelsPerMeter));
#endif

	// error checking
	if ( bmfh.bfType!= 0x4d42 ) {	// "BM" actually
		return NULL;
	}
	if ( bmih.biBitCount != 24 )  
		return NULL; 
/*
 	if ( bmih.biCompression != BMP_BI_RGB ) {
		return NULL;
	}
*/
	fseek( file, pos, SEEK_SET ); 
 
	width = bmih.biWidth; 
	height = bmih.biHeight; 
 
	int padWidth = width * 3; 
	int pad = 0; 
	if ( padWidth % 4 != 0 ) 
	{ 
		pad = 4 - (padWidth % 4); 
		padWidth += pad; 
	} 
	int bytes = height*padWidth; 
 
	unsigned char *data = new unsigned char [bytes]; 

	int result = fread( data, bytes, 1, file ); 
	
	if (!result) {
		delete [] data;
		return NULL;
	}

	fclose( file );
	
	// shuffle bitmap data such that it is (R,G,B) tuples in row-major order
	int i, j;
	j = 0;
	unsigned char temp;
	unsigned char* in;
	unsigned char* out;

	in = data;
	out = data;

	for ( j = 0; j < height; ++j )
	{
		for ( i = 0; i < width; ++i )
		{
			out[1] = in[1];
			temp = in[2];
			out[2] = in[0];
			out[0] = temp;

			in += 3;
			out += 3;
		}
		in += pad;
	}
			  
	return data; 
} 

void readBMP( const char* fname, float*& fImg, int& width, int& height ){
	unsigned char* Img = readBMP(fname, width, height);
	fImg = new float[3*width*height];
	int x,y,index;
	for(y = 0, index = 0; y < height; y++){
		for(x = 0; x < width; x++, index++){
			fImg[3*index] = Img[3*index] / 255.0f;
			fImg[3*index+1] = Img[3*index+1] / 255.0f;
			fImg[3*index+2] = Img[3*index+2] / 255.0f;
		}
	}
	delete [] Img;
}

void readBMP( const char* fname, float*& fImgR, float*& fImgG, float*& fImgB, int& width, int& height ){
	unsigned char* Img = readBMP(fname, width, height);
	fImgR = new float[width*height];
	fImgG = new float[width*height];
	fImgB = new float[width*height];
	int x,y,index;
	for(y = 0, index = 0; y < height; y++){
		for(x = 0; x < width; x++, index++){
			fImgR[index] = Img[3*index] / 255.0f;
			fImgG[index] = Img[3*index+1] / 255.0f;
			fImgB[index] = Img[3*index+2] / 255.0f;
		}
	}
	delete [] Img;
}
 
void writeBMP(const char* iname, int width, int height, unsigned char* data) 
{ 
	int bytes, pad;
	bytes = width * 3;
	pad = (bytes%4) ? 4-(bytes%4) : 0;
	bytes += pad;
	bytes *= height;

	bmfh.bfType = 0x4d42;    // "BM"
	bmfh.bfSize = sizeof(BMP_BITMAPFILEHEADER) + sizeof(BMP_BITMAPINFOHEADER) + bytes;
	bmfh.bfReserved1 = 0;
	bmfh.bfReserved2 = 0;
	bmfh.bfOffBits = /*hack sizeof(BMP_BITMAPFILEHEADER)=14, sizeof doesn't work?*/ 
					 14 + sizeof(BMP_BITMAPINFOHEADER);

	bmih.biSize = sizeof(BMP_BITMAPINFOHEADER);
	bmih.biWidth = width;
	bmih.biHeight = height;
	bmih.biPlanes = 1;
	bmih.biBitCount = 24;
	bmih.biCompression = BMP_BI_RGB;
	bmih.biSizeImage = 0;
	bmih.biXPelsPerMeter = (int)(100 / 2.54 * 72);
	bmih.biYPelsPerMeter = (int)(100 / 2.54 * 72);
	bmih.biClrUsed = 0;
	bmih.biClrImportant = 0;

#ifndef WIN32
	swapBytes(&(bmfh.bfType));
	swapBytes(&(bmfh.bfSize));
	swapBytes(&(bmfh.bfReserved1));
	swapBytes(&(bmfh.bfReserved2));
	swapBytes(&(bmfh.bfOffBits));
	swapBytes(&(bmih.biBitCount));
	swapBytes(&(bmih.biClrImportant));
	swapBytes(&(bmih.biClrUsed));
	swapBytes(&(bmih.biCompression));
	swapBytes(&(bmih.biHeight));
	swapBytes(&(bmih.biPlanes));
	swapBytes(&(bmih.biSize));
	swapBytes(&(bmih.biSizeImage));
	swapBytes(&(bmih.biWidth));
	swapBytes(&(bmih.biXPelsPerMeter));
	swapBytes(&(bmih.biYPelsPerMeter));
#endif

	FILE *outFile=fopen(iname, "wb"); 

	//	fwrite(&bmfh, sizeof(BMP_BITMAPFILEHEADER), 1, outFile);
	fwrite( &(bmfh.bfType), 2, 1, outFile); 
	fwrite( &(bmfh.bfSize), 4, 1, outFile); 
	fwrite( &(bmfh.bfReserved1), 2, 1, outFile); 
	fwrite( &(bmfh.bfReserved2), 2, 1, outFile); 
	fwrite( &(bmfh.bfOffBits), 4, 1, outFile); 

	fwrite(&bmih, sizeof(BMP_BITMAPINFOHEADER), 1, outFile); 

	bytes /= height;
	unsigned char* scanline = new unsigned char [bytes];
	for ( int j = 0; j < height; ++j )
	{
		memcpy( scanline, data + j*3*width, bytes );
		for ( int i = 0; i < width; ++i )
		{
			unsigned char temp = scanline[i*3];
			scanline[i*3] = scanline[i*3+2];
			scanline[i*3+2] = temp;
		}
		fwrite( scanline, bytes, 1, outFile);
	}

	delete [] scanline;

	fclose(outFile);
}

void writeBMP( const char* iname, int width, int height, float* data ){
	int x,y,index;
	unsigned char* Img = new unsigned char[3*width*height];
	for(y = 0, index = 0; y < height; y++){
		for(x = 0; x < width; x++, index++){
			if( data[3*index] < 0 ) Img[3*index] = 0;
			else if( data[3*index] > 1 ) Img[3*index] = 255;
			else Img[3*index] = (unsigned char)(data[3*index]*255.0f);
			if( data[3*index+1] < 0 ) Img[3*index+1] = 0;
			else if( data[3*index+1] > 1 ) Img[3*index+1] = 255;
			else Img[3*index+1] = (unsigned char)(data[3*index+1]*255.0f);
			if( data[3*index+2] < 0 ) Img[3*index+2] = 0;
			else if( data[3*index+2] > 1 ) Img[3*index+2] = 255;
			else Img[3*index+2] = (unsigned char)(data[3*index+2]*255.0f);
		}
	}
	writeBMP(iname, width, height, Img);
	delete [] Img;
}
void writeBMP( const char* iname, int width, int height, float* dataR, float* dataG, float* dataB){
	int x,y,index;
	unsigned char* Img = new unsigned char[3*width*height];
	for(y = 0, index = 0; y < height; y++){
		for(x = 0; x < width; x++, index++){
			if( dataR[index] < 0 ) Img[3*index] = 0;
			else if( dataR[index] > 1 ) Img[3*index] = 255;
			else Img[3*index] = (unsigned char)(dataR[index]*255.0f);
			if( dataG[index] < 0 ) Img[3*index+1] = 0;
			else if( dataG[index] > 1 ) Img[3*index+1] = 255;
			else Img[3*index+1] = (unsigned char)(dataG[index]*255.0f);
			if( dataB[index] < 0 ) Img[3*index+2] = 0;
			else if( dataB[index] > 1 ) Img[3*index+2] = 255;
			else Img[3*index+2] = (unsigned char)(dataB[index]*255.0f);
		}
	}
	writeBMP(iname, width, height, Img);
	delete [] Img;
}