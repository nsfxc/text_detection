#include "image_process.h"
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////
// erotion, dilation, opening and closing processes for binary images

// in place erotion process with the nearest 8 neighbours 
void erosion8(cv::Mat &m, bool erodeEdge, long* counter = 0){
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
void dilation8(cv::Mat &m, long* counter = 0){
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
void opening8(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	for (int i = 0; i < layer; ++i)
		erosion8(m, erodeEdge);
	for (int i = 0; i < layer; ++i)
		dilation8(m);
}
void opening8(cv::Mat &m, double percentage, bool erodeEdge){
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
		erosion8(m, erodeEdge, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		dilation8(m);
}

// in place closing process with the nearest 8 neighbours 
void closing8(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	for (int i = 0; i < layer; ++i)
		dilation8(m);
	for (int i = 0; i < layer; ++i)
		erosion8(m, erodeEdge);
}
void closing8(cv::Mat &m, double percentage, bool erodeEdge){
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
		dilation8(m, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		erosion8(m, erodeEdge);
}

// in place erotion process with the nearest 4 neighbours 
void erosion4(cv::Mat &m, bool erodeEdge, long* counter = 0){
	int c = m.cols, r = m.rows;
	//assert(m.type() == CV_8UC1 && c > 1 && r > 1);
	//mark pixels to be removed by changing its value to 127 
	for (int i = 1; i < r - 1; ++i)
		for (int j = 1; j < c - 1; ++j)
			if (m.at<uchar>(i, j) == 255 &&
				(m.at<uchar>(i + 1, j) == 0 || m.at<uchar>(i - 1, j) == 0 ||
				m.at<uchar>(i, j + 1) == 0 || m.at<uchar>(i, j - 1) == 0))
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
			if (m.at<uchar>(i, 0) == 255 &&
				(m.at<uchar>(i, 1) == 0 || m.at<uchar>(i + 1, 0) == 0 || m.at<uchar>(i - 1, 0) == 0))
				m.at<uchar>(i, 0) = 127;
			if (m.at<uchar>(i, c - 1) == 255 &&
				(m.at<uchar>(i, c - 2) == 0 || m.at<uchar>(i + 1, c - 1) == 0 || m.at<uchar>(i - 1, c - 1) == 0))
				m.at<uchar>(i, c - 1) = 127;
		}
		for (int j = 1; j < c - 1; ++j){
			if (m.at<uchar>(0, j) == 255 &&
				(m.at<uchar>(1, j) == 0 || m.at<uchar>(0, j + 1) == 0 || m.at<uchar>(0, j - 1) == 0))
				m.at<uchar>(0, j) = 127;
			if (m.at<uchar>(r - 1, j) == 255 &&
				(m.at<uchar>(r - 2, j) == 0 || m.at<uchar>(r - 1, j + 1) == 0 || m.at<uchar>(r - 1, j - 1) == 0))
				m.at<uchar>(r - 1, j) = 127;
		}
		if (m.at<uchar>(0, 0) == 255 && (m.at<uchar>(1, 0) == 0 || m.at<uchar>(0, 1) == 0))
			m.at<uchar>(0, 0) = 127;
		if (m.at<uchar>(0, c - 1) == 255 && (m.at<uchar>(1, c - 1) == 0 || m.at<uchar>(0, c - 2) == 0))
			m.at<uchar>(0, c - 1) = 127;
		if (m.at<uchar>(r - 1, 0) == 255 && (m.at<uchar>(r - 2, 0) == 0 || m.at<uchar>(r - 1, 1) == 0))
			m.at<uchar>(r - 1, 0) = 127;
		if (m.at<uchar>(r - 1, c - 1) == 255 && (m.at<uchar>(r - 2, c - 1) == 0 || m.at<uchar>(r - 1, c - 2) == 0))
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

// in place dilation process with the nearest 4 neighbours 
void dilation4(cv::Mat &m, long* counter = 0){
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
		}
		if (m.at<uchar>(i, c - 1) == 255){
			if (m.at<uchar>(i + 1, c - 1) == 0)
				m.at<uchar>(i + 1, c - 1) = 127;
			if (m.at<uchar>(i - 1, c - 1) == 0)
				m.at<uchar>(i - 1, c - 1) = 127;
			if (m.at<uchar>(i, c - 2) == 0)
				m.at<uchar>(i, c - 2) = 127;
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
		}
		if (m.at<uchar>(r - 1, j) == 255){
			if (m.at<uchar>(r - 2, j) == 0)
				m.at<uchar>(r - 2, j) = 127;
			if (m.at<uchar>(r - 1, j + 1) == 0)
				m.at<uchar>(r - 1, j + 1) = 127;
			if (m.at<uchar>(r - 1, j - 1) == 0)
				m.at<uchar>(r - 1, j - 1) = 127;
		}
	}
	if (m.at<uchar>(0, 0) == 255){
		if (m.at<uchar>(1, 0) == 0)
			m.at<uchar>(1, 0) = 127;
		if (m.at<uchar>(0, 1) == 0)
			m.at<uchar>(0, 1) = 127;
	}
	if (m.at<uchar>(0, c - 1) == 255){
		if (m.at<uchar>(1, c - 1) == 0)
			m.at<uchar>(1, c - 1) = 127;
		if (m.at<uchar>(0, c - 2) == 0)
			m.at<uchar>(0, c - 2) = 127;
	}
	if (m.at<uchar>(r - 1, c - 1) == 255){
		if (m.at<uchar>(r - 2, c - 1) == 0)
			m.at<uchar>(r - 2, c - 1) = 127;
		if (m.at<uchar>(r - 1, c - 2) == 0)
			m.at<uchar>(r - 1, c - 2) = 127;
	}
	if (m.at<uchar>(r - 1, 0) == 255){
		if (m.at<uchar>(r - 2, 0) == 0)
			m.at<uchar>(r - 2, 0) = 127;
		if (m.at<uchar>(r - 1, 1) == 0)
			m.at<uchar>(r - 1, 1) = 127;
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

// in place opening process with the nearest 4 neighbours 
void opening4(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);

	for (int i = 0; i < layer; ++i)
		erosion4(m, erodeEdge);
	for (int i = 0; i < layer; ++i)
		dilation4(m);
}
void opening4(cv::Mat &m, double percentage, bool erodeEdge){
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
		erosion4(m, erodeEdge, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		dilation4(m);
}

// in place closing process with the nearest 4 neighbours 
void closing4(cv::Mat &m, int layer, bool erodeEdge){
	int c = m.cols, r = m.rows;
	assert(m.type() == CV_8UC1 && c > 1 && r > 1);

	for (int i = 0; i < layer; ++i)
		dilation4(m);
	for (int i = 0; i < layer; ++i)
		erosion4(m, erodeEdge);
}
void closing4(cv::Mat &m, double percentage, bool erodeEdge){
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
		dilation4(m, &pixel);
		++layer;
	}
	for (int i = 0; i < layer; ++i)
		erosion4(m, erodeEdge);
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

/////////////////////////////////////////////////////////////////////
//binary inverse
//the following function assume inputs of same size and type CV_8UC1 and binary diaphragm

// in place inverse binary image
void inverseBinary(cv::Mat &m){
	assert(m.type() == CV_8UC1);
	for (int i = 0; i < m.rows; ++i)
		for (int j = 0; j < m.cols; ++j)
				m.at<uchar>(i, j) ^= 0xFF;
}

/////////////////////////////////////////////////////////////////////
struct blob{
	int x1, x2, y1, y2;
	int size, index;
	int centrex,centrey;
};

int distance(int x1, int y1, int x2, int y2){
	return (x1 - x2)* (x1 - x2) + (y1 - y2) * (y1 - y2);
}

/////////////////////////////////////////////////////////////////////
//find all connected composant of a binary image

void FindBlobs(const Mat &img, Mat &components)
{
	Mat binary = (img < 128);
	Mat labelImage(binary.size(),CV_32S);

	int nLabels = connectedComponents(binary,labelImage,8);
	vector<blob> blobs(nLabels);
	for(int i = 0; i < nLabels; i++) {
		blobs[i].x1 = -1;
		blobs[i].y1 = -1;
		blobs[i].x2 = -1;
		blobs[i].y2 = -1;
		blobs[i].size = 0;
		blobs[i].centrex = 0;
		blobs[i].centrey = 0;
		blobs[i].index = -1;
	}
	for(int x = 0; x < binary.rows; x++){
		for(int y = 0; y < binary.cols; y++){
			int index = labelImage.at<int>(x,y);
			blobs[index].x1 = (blobs[index].x1 == -1 || x < blobs[index].x1)?x:blobs[index].x1;
			blobs[index].y1 = (blobs[index].y1 == -1 || y < blobs[index].y1)?y:blobs[index].y1;
			blobs[index].x2 = (blobs[index].x2 == -1 || x > blobs[index].x2)?x:blobs[index].x2;
			blobs[index].y2 = (blobs[index].y2 == -1 || y > blobs[index].y2)?y:blobs[index].y2;
			blobs[index].size++;
			blobs[index].centrex += x;
			blobs[index].centrey += y;
		}
	}
	for (int i = 0; i < nLabels; i++){
		blobs[i].centrex = blobs[i].centrex/blobs[i].size;
		blobs[i].centrex = blobs[i].centrex/blobs[i].size;
	}
	int k = 0;
	for(int i = 0; i < nLabels; i++){
		if (blobs[i].index== -1) {
			blobs[i].index = k;
			k++;
		}
		for(int j = i+1; j < nLabels; j++){
			if (distance(blobs[i].centrex,blobs[i].centrey,blobs[j].centrex, blobs[j].centrey) < 9)
				blobs[j].index = blobs[i].index;
		}
	}
	for (int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			components.at<int>(i,j) = blobs[labelImage.at<int>(i,j)].index;
		}
	}
}
