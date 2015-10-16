#include <iostream>
#include "Image.h"
#include "Pixel.h"
#include "kmeans.h"
#include "image_process.h"

using namespace cv;

void image_process_test(){
	Mat I = imread("../test4_result.jpg"), tmp;
	cvtColor(I, tmp, CV_BGR2GRAY);
	imshow("tmp", tmp);
	toBinary(tmp);
	imshow("binary", tmp);
	waitKey(0);
	destroyAllWindows();

	//assert(false);

	closing(tmp);
	imshow("closing", tmp);
	opening(tmp);
	imshow("opening", tmp);
	waitKey(0);
	destroyAllWindows();

	cvtColor(imread("../test4.jpg"), I, CV_BGR2GRAY);
	imshow("original", I);
	diaphragm(I, tmp);
	imshow("diaphragm", I);
	waitKey(0);

}
int main() {
    //test of input and sobel
    Mat test = imread("../test4.jpg");
    Mat Ix = XSobel(test);;
    Mat Iy = YSobel(test);
    Mat Ixy = XYSobel(test);
    Mat Iyx = YXSobel(test);
    Mat averI = average(Ix,Iy,Ixy,Iyx);
    showImage(averI);
    Image *I = new Image[4];
    I[0] = Image(Ix,5,5);
    I[1] = Image(Iy,5,5);
    I[2] = Image(Ixy,5,5);
    I[3] = Image(Iyx,5,5);
    Mat result = kmeans(2,4,I);
    showImage(result);
    return 0;
}