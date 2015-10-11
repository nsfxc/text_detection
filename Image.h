//
// Created by czx on 01/10/15.
//

#ifndef TEXT_DETECTION_IMAGE_H
#define TEXT_DETECTION_IMAGE_H

#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Pixel.h"
using namespace cv;
using namespace std;
class Image{
private:
    Pixel **pxs;
public:
    int rows, cols,w,h;
    Image(Mat &I,int w, int h);
    Image();
    Pixel at(int x, int y);
};

Mat gradient(const Mat& I, const Mat& H);
Mat XSobel(const Mat& I);
Mat YSobel(const Mat& I);
Mat XYSobel(const Mat& I);
Mat YXSobel(const Mat& I);
Mat average(const Mat& Ix, const Mat& Iy, const Mat& Ixy, const Mat& Iyx);

void showImage(const Mat& I);

#endif //TEXT_DETECTION_IMAGE_H

