#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_process.h"

using namespace cv;

int main(){
	Mat I = imread("../splash.tiff"), tmp;
	cvtColor(I, tmp, CV_BGR2GRAY);
	imshow("tmp", tmp);
	toBinary(tmp);
	imshow("binary", tmp);
	waitKey(0);
	destroyAllWindows();

	closing(tmp);
	imshow("closing", tmp);
	opening(tmp);
	imshow("opening", tmp);
	waitKey(0);
	destroyAllWindows();

	cvtColor(imread("../lenna.png"), I, CV_BGR2GRAY);
	imshow("original", I);
	diaphragm(I, tmp);
	imshow("diaphragm", I);
	waitKey(0);
	return 0;
}