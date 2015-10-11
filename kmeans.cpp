//
// Created by czx on 11/10/15.
//

#include "kmeans.h"
#include<iostream>
int findCenter(int x, int y, int nbcenter, int nbi,Pixel **centers,Image *I) {
    int closed = 0;
    float min_dis = 0;
    for (int j = 0; j < nbi; j++)
        min_dis += I[j].at(x, y).distanceFeature(centers[0][j]);
    for (int c = 1; c < nbcenter; c++) {
        float dis = 0;
        for (int j = 0; j < nbi; j++)
            dis += I[j].at(x, y).distanceFeature(centers[c][j]);
        if (dis < min_dis) {
            closed = c;
            min_dis = dis;
        }
    }
    return closed;
}

void updateCenter(int center, int nbi,Pixel **centers, Image *I,Mat &result){
    for(int j = 0; j < nbi; j++)
        centers[center][j].setFeatures(0,0,0,0,0,0);
    int count = 0;
    foreachPixel(x,y,I[0].rows,I[0].cols){
            if ((int)result.at<uchar>(x,y) == center){
                count ++;
                for (int j = 0; j < nbi; j++)
                    centers[center][j].plusFeatures(I[j].at(x,y));
            }
        }
    if(count != 0)
        for(int j = 0; j < nbi; j++)
            centers[center][j].divideFeatures(count);
}

Mat kmeans(int nbcenter,int nbi, Image *I){
    Pixel **centers = new Pixel*[nbcenter];
    int r = I[0].rows, c = I[0].cols;
    Mat result(r,c,CV_8U);

    for(int i = 0; i < nbcenter; i++){
        int x = rand()%c, y = rand()%r;
        centers[i] = new Pixel[nbi];
        for(int j = 0; j < nbi; j++){
            Pixel p = I[j].at(y,x);
            centers[i][j] = Pixel(y,x);
            centers[i][j].setFeatures(p.mu,p.sigma,p.Eg,p.Et,p.I,p.H);
        }
    }

    for(int i = 0; i < iter;i++) {
        foreachPixel(x, y, r, c){
            int closed = findCenter(x,y,nbcenter,nbi,centers,I);
            result.at<uchar>(x, y) = (uchar) closed;
        }

        for(int center = 0; center < nbcenter; center++){
            updateCenter(center,nbi,centers,I,result);
        }
    }
    return  result;
};