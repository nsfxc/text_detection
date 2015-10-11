//
// Created by czx on 08/10/15.
//

#include "Pixel.h"
#include <stdexcept>
#include<iostream>
Pixel features(const int x, const int y,const int w, const int h, const Mat E){
    if (E.cols <= x+w || E.rows <= y+h)
        throw std::invalid_argument("Wrong rows number!");
    double mu = 0, sigma = 0,Eg = 0, Et = 0, I = 0, H = 0;
    for(int i = x; i <= x+w; i++){
        for(int j = y; j <= h+y; j++){
            mu += (float)E.at<uchar>(j,i);
            Eg += (float)E.at<uchar>(j,i) * (float)E.at<uchar>(j,i);
            Et += (float)E.at<uchar>(j,i) * log((float)E.at<uchar>(j,i));
            I += Sqr((j-y+1)-(i-x+1))*(float)E.at<uchar>(j,i);
            H += 1.0/(1.0+Sqr((i-x+1)-(j-y+1)))*(float)E.at<uchar>(j,i);
        }
    }
    mu = mu/((float)w*h);
    for(int i= x; i <= x+w; i++)
        for(int j = y; j <= y+h; j++)
            sigma += ((float)E.at<uchar>(j,i)-mu)*((float)E.at<uchar>(j,i)-mu);
    sigma = sqrt(sigma/((float)w*h));
    Pixel p(x,y);
    p.setFeatures(mu,sigma,Eg,Et,I,H);
    return p;
}

float Pixel::distanceEuclidian(Pixel p) {
    return sqrt(Sqr(p.x-this->x) + Sqr(p.y-this->y));
}

float Pixel::distanceFeature(Pixel p) {
    return sqrt(Sqr(p.mu - this->mu) + Sqr(p.sigma-this->sigma) + Sqr(p.Eg - this->Eg) + Sqr(p.Et - this->Et) + Sqr(p.H - this->H) + Sqr(p.I - this->I));
}

void Pixel::plusFeatures(Pixel p) {
    this->mu += p.mu;
    this->sigma += p.sigma;
    this->Et += p.Et;
    this->Eg += p.Eg;
    this->H += p.H;
    this->I += p.I;
}

void Pixel::divideFeatures(double d) {
    this->mu /= d;
    this->sigma /= d;
    this->Et /= d;
    this->Eg /= d;
    this->H /= d;
    this->I /= d;
}