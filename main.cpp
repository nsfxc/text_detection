#include <iostream>
#include "Image.h"
#include "Pixel.h"
#include "kmeans.h"

int main() {
    //test of input and sobel
    Mat test = imread("fruits.jpg");
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