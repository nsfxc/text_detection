//
// Created by czx on 08/10/15.
//

#ifndef TEXT_DETECTION_PIXEL_H
#define TEXT_DETECTION_PIXEL_H
#define Sqr(x)((x)*(x))
#define foreachPixel(x,y,r,c) for(int x = 0; x < r; x++ )for(int y = 0; y < c; y++)
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class Pixel :public Point {
public:
    double mu,sigma,Eg,Et,I,H;
    Pixel(){
        this->x = 0;
        this->y = 0;
    }
    Pixel(int x, int y){
        this->x = x;
        this->y = y;
    }
    void setFeatures(double mu, double sigma, double Eg, double Et, double I, double H){
        this->mu = mu;
        this->sigma = sigma;
        this->Eg = Eg;
        this->Et = Et;
        this->I = I;
        this->H = H;
    }
    float distanceEuclidian(Pixel p);
    float distanceFeature(Pixel p);
    void plusFeatures(Pixel p);
    void divideFeatures(double d);
};

Pixel features(const int x, const int y, const int w, const int h, const Mat I);

#endif //TEXT_DETECTION_PIXEL_H
