//
// Created by czx on 01/10/15.
//

#include "Image.h"

Mat gradient(const Mat& I, const Mat& H){
    int rn = I.rows,cn = I.cols;
    Mat G(rn,cn,CV_32FC1, Scalar::all(0));
    assert(H.rows == H.cols && H.rows%2 == 1);
    int m = (H.rows-1)/2;
    for(int i = m; i < rn-m; i++){
        for(int j = m; j < cn-m; j++){
            float r = 0.,g = 0., b = 0.;
            for(int p = i-m; p <= i+m; p++){
                for(int q = j-m; q <= j+m; q++){
                    r += H.at<int>(p-i+m,q-j+m) * (float)I.at<Vec3b>(p,q).val[2];
                    g += H.at<int>(p-i+m,q-j+m) * (float)I.at<Vec3b>(p,q).val[1];
                    b += H.at<int>(p-i+m,q-j+m) * (float)I.at<Vec3b>(p,q).val[0];
                }
            }
            G.at<float>(i,j) = sqrt((r*r+g*g+b*b)/3);
        }
    }
    Mat dst, dst_uchar(rn,cn,CV_8U, Scalar::all(0));
    normalize(G,dst,1,255,NORM_MINMAX,CV_32FC1,Mat());
    for(int i = 0; i < rn; i++){
        for(int j = 0; j < cn; j++){
            dst_uchar.at<uchar>(i,j) = (uchar)dst.at<float>(i,j);
        }
    }
    return dst_uchar;
}

Mat XSobel(const Mat& I){
    int H[3][3] ={{-1,-2,-1},{0,0,0},{1,2,1}};
    return gradient(I,Mat(3,3,CV_32S,&H));
}

Mat YSobel(const Mat &I){
    int H[3][3] ={{-1,0,1},{-2,0,2},{-1,0,1}};
    return gradient(I,Mat(3,3,CV_32S,&H));
}

Mat XYSobel(const Mat& I){
    int H[3][3] ={{0,1,2},{-1,0,1},{-2,-1,0}};
    return gradient(I,Mat(3,3,CV_32S,&H));
}

Mat YXSobel(const Mat& I){
    int H[3][3] ={{-2,-1,0},{-1,0,1},{0,1,2}};
    return gradient(I,Mat(3,3,CV_32S,&H));
}

Mat average(const Mat& Ix, const Mat& Iy, const Mat& Ixy, const Mat& Iyx){
    int r = Ix.rows;
    int c = Ix.cols;
    Mat I(r,c,CV_8U);
    foreachPixel(i,j,r,c){
            float x = (float)Ix.at<uchar>(i,j);
            float y = (float)Iy.at<uchar>(i,j);
            float xy = (float)Ixy.at<uchar>(i,j);
            float yx = (float)Iyx.at<uchar>(i,j);
            I.at<uchar>(i,j) =(uchar)((x+y+xy+yx)/4.);
    }
    return I;
}

void showImage(const Mat& I){
    imshow("Image",I);
    waitKey();
}

Image::Image(Mat &I,int w, int h) {
    this->cols = I.cols;
    this->rows = I.rows;
    this->pxs = new Pixel*[this->rows];
    for(int i = 0; i < this->rows;i++)
        this->pxs[i] = new Pixel[this->cols];
    foreachPixel(i,j,this->rows,this->cols){
            if (i + h < this->rows && j + w < this->cols )
                pxs[i][j] = features(j,i,w,h,I);
            else {
                pxs[i][j] = Pixel(0, 0);
                pxs[i][j].setFeatures(0,0,0,0,0,0);
            }
        }
}

Pixel Image::at(int x, int y) {
    return pxs[x][y];
}

Image::Image(){
    this->cols = 0;
    this->rows = 0;
}