#include <iostream>
#include "Image.h"
#include "Pixel.h"
#include "kmeans.h"
#include "image_process.h"
int main() {
    //test of input and sobel
    Mat test = imread("test4.jpg");
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
    Mat result = kmeans(2,4,I);
    showImage(result);
    opening(result,2,false);
    showImage(result);
    imwrite("test4_result.jpg",result);
    return 0;
}