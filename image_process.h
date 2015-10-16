#ifndef IMAGE_PROCESS_SYTRUS
#define IMAGE_PROCESS_SYTRUS

#include <opencv2/imgproc/imgproc.hpp>

/////////////////////////////////////////////////////////////////////
// erotion, dilation, opening and closing processes for binary images
// the following functions assume a BINARY input of type CV_8UC1 (i.e. 0 or 255 grayscale) with width and height > 1

// in place erotion process with the nearest 8 neighbours 
//void erosion(cv::Mat &m, bool erodeEdge, long* counter = 0);

// in place dilation process with the nearest 8 neighbours 
//void dilation(cv::Mat &m, long* counter = 0);

// in place opening process with the nearest 8 neighbours 
void opening(cv::Mat &m, int layer, bool erodeEdge = true);					//with specified layer of pixels to be eroded before dilation
void opening(cv::Mat &m, double percentage = 61.8, bool erodeEdge = true);	//with specified percentage of pixels to be eroded before dilation

// in place closing process with the nearest 8 neighbours 
void closing(cv::Mat &m, int layer, bool erodeEdge = false);				//with specified layer of pixels to be dilated before erosion
void closing(cv::Mat &m, double percentage = 61.8, bool erodeEdge = false);	//with specified percentage of pixels to be dilated before erosion

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//Otsu method for turning a gray scale image to binary
//the following function assume an input of type CV_8UC1

// in place Otsu method for conversion of gray scale to binary
void toBinary(cv::Mat &m);

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//diaphragm filter
//the following function assume inputs of same size and type CV_8UC1 and binary diaphragm

// in place diaphragm filter of m by d
void diaphragm(cv::Mat &m, const cv::Mat &d);

/////////////////////////////////////////////////////////////////////
#endif