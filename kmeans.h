//
// Created by czx on 11/10/15.
//

#ifndef TEXTDETECTION_KMEANS_H
#define TEXTDETECTION_KMEANS_H
#include"Image.h"
#include <vector>
using namespace std;

const int iter = 2;
Mat kmeans(int nbcentre,int nbi,Image *I);

#endif //TEXTDETECTION_KMEANS_H
