#include "image_process.h"
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////
// erotion, dilation, opening and closing processes for binary images

// in place erotion process with the nearest 8 neighbours 
void erosion(cv::Mat &m, bool erodeEdge, long* counter = 0){
	int c = m.cols, r = m.rows;
	//assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	//mark pixels to be removed by changing its value to 127 
	for (int i = 1; i < r-1; ++i)
		for (int j = 1; j < c-1; ++j)
			if ( m.at<uchar>(i, j) == 255 &&
				(m.at<uchar>(i + 1, j) == 0 || m.at<uchar>(i - 1, j) == 0 ||
				m.at<uchar>(i, j + 1) == 0 || m.at<uchar>(i, j - 1) == 0 ||
				m.at<uchar>(i + 1, j + 1) == 0 || m.at<uchar>(i - 1, j + 1) == 0 ||
				m.at<uchar>(i - 1, j + 1) == 0 || m.at<uchar>(i - 1, j - 1) == 0))
				m.at<uchar>(i, j) = 127;
	//process edges
	if (erodeEdge){
		for (int i = 0; i < r; ++i){
			m.at<uchar>(i, 0) = 0;
			m.at<uchar>(i, c - 1) = 0;
		}
		for (int j = 1; j < c - 1; ++j){
			m.at<uchar>(0, j) = 0;
			m.at<uchar>(r - 1, j) = 0;
		}
	}
	else{
		for (int i = 1; i < r - 1; ++i){
			if ( m.at<uchar>(i, 0) == 255 && 
				(m.at<uchar>(i, 1) == 0 || m.at<uchar>(i + 1, 0) == 0 || m.at<uchar>(i - 1, 0) == 0
				|| m.at<uchar>(i + 1, 1) == 0 || m.at<uchar>(i - 1, 1) == 0))
				m.at<uchar>(i, 0) = 127;
			if ( m.at<uchar>(i, c - 1) == 255 && 
				(m.at<uchar>(i, c - 2) == 0 || m.at<uchar>(i + 1, c - 1) == 0 || m.at<uchar>(i - 1, c - 1) == 0
				|| m.at<uchar>(i + 1, c - 2) == 0 || m.at<uchar>(i - 1, c - 2) == 0))
				m.at<uchar>(i, c - 1) = 127;
		}
		for (int j = 1; j < c - 1; ++j){
			if ( m.at<uchar>(0, j) == 255 && 
				(m.at<uchar>(1, j) == 0 || m.at<uchar>(0, j + 1) == 0 || m.at<uchar>(0, j - 1) == 0
				|| m.at<uchar>(1, j + 1) == 0 || m.at<uchar>(1, j - 1) == 0))
				m.at<uchar>(0, j) = 127;
			if ( m.at<uchar>(r - 1, j) == 255 && 
				(m.at<uchar>(r - 2, j) == 0 || m.at<uchar>(r - 1, j + 1) == 0 || m.at<uchar>(r - 1, j - 1) == 0
				|| m.at<uchar>(r - 2, j + 1) == 0 || m.at<uchar>(r - 2, j - 1) == 0))
				m.at<uchar>(r - 1, j) = 127;
		}
		if (m.at<uchar>(0, 0) == 255 && (m.at<uchar>(1, 0) == 0 || m.at<uchar>(0, 1) == 0 || m.at<uchar>(1, 1) == 0))
			m.at<uchar>(0, 0) = 127;
		if (m.at<uchar>(0, c - 1) == 255 && (m.at<uchar>(1, c - 1) == 0 || m.at<uchar>(0, c - 2) == 0 || m.at<uchar>(1, c - 2) == 0))
			m.at<uchar>(0, c - 1) = 127;
		if (m.at<uchar>(r - 1, 0) == 255 && (m.at<uchar>(r - 2, 0) == 0 || m.at<uchar>(r - 1, 1) == 0 || m.at<uchar>(r - 2, 1) == 0))
			m.at<uchar>(r - 1, 0) = 127;
		if (m.at<uchar>(r - 1, c - 1) == 255 && (m.at<uchar>(r - 2, c - 1) == 0 || m.at<uchar>(r - 1, c - 2) == 0 || m.at<uchar>(r - 2, c - 2) == 0))
			m.at<uchar>(r - 1, c - 1) = 127;
	}
	//remove all marked pixels
	if (counter != 0)
		for (int i = 0; i < r; ++i){
			for (int j = 0; j < c; ++j)
				if (m.at<uchar>(i, j) == 127){
					m.at<uchar>(i, j) = 0;
					(*counter)--;
				}
		}
	else
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				if (m.at<uchar>(i, j) == 127)
					m.at<uchar>(i, j) = 0;
}

// in place dilation process with the nearest 8 neighbours 
void dilation(cv::Mat &m, long* counter = 0){
	int c = m.cols, r = m.rows;
	//assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	//mark pixels to be added by changing its value to 127 
	for (int i = 1; i < r - 1; ++i)
		for (int j = 1; j < c - 1; ++j)
			if (m.at<uchar>(i, j) == 255){
				if (m.at<uchar>(i + 1, j) == 0)
					m.at<uchar>(i + 1, j) = 127;
				if (m.at<uchar>(i - 1, j) == 0)
					m.at<uchar>(i - 1, j) = 127;
				if (m.at<uchar>(i, j + 1) == 0)
					m.at<uchar>(i, j + 1) = 127;
				if (m.at<uchar>(i, j - 1) == 0)
					m.at<uchar>(i, j - 1) = 127;
				if (m.at<uchar>(i + 1, j + 1) == 0)
					m.at<uchar>(i + 1, j + 1) = 127;
				if (m.at<uchar>(i - 1, j + 1) == 0)
					m.at<uchar>(i - 1, j + 1) = 127;
				if (m.at<uchar>(i - 1, j + 1) == 0)
					m.at<uchar>(i - 1, j + 1) = 127;
				if (m.at<uchar>(i - 1, j - 1) == 0)
					m.at<uchar>(i - 1, j - 1) = 127;
			}
	//process edges
	for (int i = 1; i < r - 1; ++i){
		if (m.at<uchar>(i, 0) == 255){
			if (m.at<uchar>(i + 1, 0) == 0)
				m.at<uchar>(i + 1, 0) = 127;
			if (m.at<uchar>(i - 1, 0) == 0)
				m.at<uchar>(i - 1, 0) = 127;
			if (m.at<uchar>(i, 1) == 0)
				m.at<uchar>(i, 1) = 127;
			if (m.at<uchar>(i + 1, 1) == 0)
				m.at<uchar>(i + 1, 1) = 127;
			if (m.at<uchar>(i - 1, 1) == 0)
				m.at<uchar>(i - 1, 1) = 127;
		}
		if (m.at<uchar>(i, c - 1) == 255){
			if (m.at<uchar>(i + 1, c - 1) == 0)
				m.at<uchar>(i + 1, c - 1) = 127;
			if (m.at<uchar>(i - 1, c - 1) == 0)
				m.at<uchar>(i - 1, c - 1) = 127;
			if (m.at<uchar>(i, c - 2) == 0)
				m.at<uchar>(i, c - 2) = 127;
			if (m.at<uchar>(i + 1, c - 2) == 0)
				m.at<uchar>(i + 1, c - 2) = 127;
			if (m.at<uchar>(i - 1, c - 2) == 0)
				m.at<uchar>(i - 1, c - 2) = 127;
		}
	}
	for (int j = 1; j < c - 1; ++j){
		if (m.at<uchar>(0, j) == 255){
			if (m.at<uchar>(1, j) == 0)
				m.at<uchar>(1, j) = 127;
			if (m.at<uchar>(0, j + 1) == 0)
				m.at<uchar>(0, j + 1) = 127;
			if (m.at<uchar>(0, j - 1) == 0)
				m.at<uchar>(0, j - 1) = 127; 
			if (m.at<uchar>(1, j + 1) == 0)
				m.at<uchar>(1, j + 1) = 127;
			if (m.at<uchar>(1, j - 1) == 0)
				m.at<uchar>(1, j - 1) = 127;
		}
		if (m.at<uchar>(r - 1, j) == 255){
			if (m.at<uchar>(r - 2, j) == 0)
				m.at<uchar>(r - 2, j) = 127;
			if (m.at<uchar>(r - 1, j + 1) == 0)
				m.at<uchar>(r - 1, j + 1) = 127;
			if (m.at<uchar>(r - 1, j - 1) == 0)
				m.at<uchar>(r - 1, j - 1) = 127;
			if (m.at<uchar>(r - 2, j + 1) == 0)
				m.at<uchar>(r - 2, j + 1) = 127;
			if (m.at<uchar>(r - 2, j - 1) == 0)
				m.at<uchar>(r - 2, j - 1) = 127;
		}
	}
	if (m.at<uchar>(0, 0) == 255){
		if (m.at<uchar>(1, 0) == 0)
			m.at<uchar>(1, 0) = 127;
		if (m.at<uchar>(0, 1) == 0)
			m.at<uchar>(0, 1) = 127;
		if (m.at<uchar>(1, 1) == 0)
			m.at<uchar>(1, 1) = 127;
	}
	if (m.at<uchar>(0, c - 1) == 255){
		if (m.at<uchar>(1, c - 1) == 0)
			m.at<uchar>(1, c - 1) = 127;
		if (m.at<uchar>(0, c - 2) == 0)
			m.at<uchar>(0, c - 2) = 127;
		if (m.at<uchar>(1, c - 2) == 0)
			m.at<uchar>(1, c - 2) = 127;
	}
	if (m.at<uchar>(r - 1, c - 1) == 255){
		if (m.at<uchar>(r - 2, c - 1) == 0)
			m.at<uchar>(r - 2, c - 1) = 127;
		if (m.at<uchar>(r - 1, c - 2) == 0)
			m.at<uchar>(r - 1, c - 2) = 127;
		if (m.at<uchar>(r - 2, c - 2) == 0)
			m.at<uchar>(r - 2, c - 2) = 127;
	}
	if (m.at<uchar>(r - 1, 0) == 255){
		if (m.at<uchar>(r - 2, 0) == 0)
			m.at<uchar>(r - 2, 0) = 127;
		if (m.at<uchar>(r - 1, 1) == 0)
			m.at<uchar>(r - 1, 1) = 127;
		if (m.at<uchar>(r - 2, 1) == 0)
			m.at<uchar>(r - 2, 1) = 127;
	}
	//add all marked pixels
	if (counter != 0){
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				if (m.at<uchar>(i, j) == 127){
					m.at<uchar>(i, j) = 255;
					(*counter)--;
				}
	}
	else
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				if (m.at<uchar>(i, j) == 127)
					m.at<uchar>(i, j) = 255;
}

// in place opening process with the nearest 8 neighbours 
void opening(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	for (int i = 0; i < layer; ++i)
		erosion(m, erodeEdge);
	for (int i = 0; i < layer; ++i)
		dilation(m);
}
void opening(cv::Mat &m, double percentage, bool erodeEdge){
	//initialization, check for appropriate input
	int c = m.cols, r = m.rows, layer = 0;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	if (percentage <= 0.)
		return;
	if (percentage >= 100.){
		for (int i = 0; i < r; ++i)
			for (int j = 0; j < c; ++j)
				m.at<uchar>(i, j) = 0;
		return;
	}
	long pixel = 0;
	//count pixels and get the number of pixels to be removed
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < c; ++j)
			if (m.at<uchar>(i, j) == 255)
				++pixel;
	pixel = long(pixel * percentage / 100);
	//erode while pixel > 0
	while (pixel > 0){
		erosion(m, erodeEdge, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		dilation(m);
}

// in place closing process with the nearest 8 neighbours 
void closing(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	for (int i = 0; i < layer; ++i)
		dilation(m);
	for (int i = 0; i < layer; ++i)
		erosion(m, erodeEdge);
}
void closing(cv::Mat &m, double percentage, bool erodeEdge){
	//initialization, check for appropriate input
	int c = m.cols, r = m.rows, layer = 0;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	if (percentage <= 0.)
		return;
	long pixel = 0;
	//count pixels and get the number of pixels to be removed
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < c; ++j)
			if (m.at<uchar>(i, j) == 255)
				++pixel;
	pixel = long(pixel * percentage / 100);
	//erode while pixel > 0
	while (pixel > 0){
		dilation(m, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		erosion(m, erodeEdge);
}

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//Otsu method for turning a gray scale image to binary

// in place Otsu method for conversion of gray scale to binary
void toBinary(cv::Mat &m){
	assert(m.type() == CV_8SC1);
	long hist[256];
	const int r = m.rows, c = m.cols;
	int	thrs1,thrs2;
	double var = -1.;
	//build up histogram by gray scale
	for (int i = 0; i < 256; ++i)
		hist[i] = 0;
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < c; ++j)
			++(hist[m.at<uchar>(i, j)]);
	//perform search for threshold
	double mu0, mu1, w0, w1, px;
	for (int t = 0; t < 257; ++t){
		//re-initialize
		mu0 = 0.;	mu1 = 0.;
		w0 = 0.;	w1 = 0.;
		//load mu0 w0
		px = 0.;
		for (int i = 0; i < t; ++i){
			px += hist[i];
			mu0 += i * hist[i];
		}
		w0 = px / (c * r);
		mu0 /= px;
		//load mu1 w1
		px = 0.;
		for (int i = t; i < 256; ++i){
			px += hist[i];
			mu1 += i * hist[i];
		}
		w1 = px / (c * r);
		mu1 /= px;
		//compare and update
		px = w0*w1*(mu0-mu1)*(mu0-mu1);
		if (px > var){
			var = px;
			thrs2 = t;
			thrs1 = t;
		}
		else if (px == var)
			thrs2 = t;
	}
	//convert m to binary
	thrs1 = (thrs1 + thrs2) / 2;
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < c; ++j)
			if (m.at<uchar>(i, j) < thrs1)
				m.at<uchar>(i, j) = 0;
			else
				m.at<uchar>(i, j) = 255;
}

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//diaphragm filter
//the following function assume inputs of type CV_8UC1 and d binary

// in place diaphragm filter of m by d
void diaphragm(cv::Mat &m, const cv::Mat &d){
	assert(m.type() == CV_8UC1);
	assert(d.type() == CV_8UC1);
	assert(m.size == d.size);
	for (int i = 0; i < m.rows; ++i)
		for (int j = 0; j < m.cols; ++j)
			if (d.at<uchar>(i, j) == 0)
				m.at<uchar>(i, j) = 0;
}

/////////////////////////////////////////////////////////////////////