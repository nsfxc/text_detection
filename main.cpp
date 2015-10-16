#include <iostream>
#include "Image.h"
#include "Pixel.h"
#include "kmeans.h"
#include "image_process.h"

//for seperate adjustment of parameters
void test_opening(){
	Mat result = imread("../test_images/test4_raw.jpg"), tmp;
	cvtColor(result, result, CV_BGR2GRAY);
	toBinary(result);
	cvtColor(imread("../test_images/test4.jpg"), tmp, CV_BGR2GRAY);
	imshow("original-grey",tmp);
	imshow("original-result", result);
	closing4(result,200.);
	opening8(result,40.,false);
	showImage(result);
	diaphragm(tmp, result);
	showImage(tmp);
}

int main() {
	//for seperate adjustment of parameters
	//test_opening();
	//return 0;

    //test of input and sobel
    Mat test = imread("../test_images/test9.jpg");
    Mat Ix = XSobel(test);;
    Mat Iy = YSobel(test);
    Mat Ixy = XYSobel(test);
    Mat Iyx = YXSobel(test);
    Mat averI = average(Ix,Iy,Ixy,Iyx);
    showImage(averI);
    //test kmeans
    Image *I = new Image[4];
    I[0] = Image(Ix,5,5);
    I[1] = Image(Iy,5,5);
    I[2] = Image(Ixy,5,5);
    I[3] = Image(Iyx,5,5);
    Mat result = kmeans(2,4,I),tmp;
	cvtColor(test, tmp, CV_BGR2GRAY);
	//toBinary(result);
    showImage(result);
	imwrite("../test_images/test9_raw.jpg", result);
	closing4(result, 200.);
	opening8(result, 40., false);
	showImage(result);
	diaphragm(tmp, result);
	showImage(tmp);
	imwrite("../test_images/test9_result.jpg", tmp);
    return 0;
}
