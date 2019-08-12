// lazy snapping.cpp : 定义控制台应用程序的入口点。
//
#include "MaxProdBP.h"
#include "Bitmap.h"
#include "mrf.h"
#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<vector>
#include<iostream>
using namespace cv;
using namespace std;
#define K_MEANS_CLUSTERS_NUM 64
string MainWindowName = "Mouse operation";//主窗口（进行鼠标操作的原图窗口）
string MaskWindowName = "Mask Image";//附属窗口（记录鼠标操作的黑白窗口）
bool g_bLeftButtonDown = false;
bool g_brightButtonDown = false;
string IMAGE_PATH_PREFIX = "data1/";
string IMAGE_OUTPUT_PATH = "output/";
vector<Mat>KmeansResult;
Mat g_mMaskImg;//记录划线的图
Mat srcimg1D, E1_1D;
Point g_pPrevious; // previous point position
Point g_pCurrent; // current mouse position 这两句对点的声明如果放进了mousecallback函数里就会出问题
void on_MouseCallBack(int event, int x, int y, int flags, void* params);
Mat UIdesign(const Mat&  srcImg);vector<Mat>Kmeans(Mat &mark, Mat &srcImg);//聚类返回vector0为前景的聚类结果 vector1为后景
Mat coutdata(Mat &mark, Mat &srcImg);//返回1维的E1能量值矩阵
Mat cv1D(Mat &mark);//返回列向量只转换8uc3
MRF::CostVal dCost(int pix, int i)
{
	double value;
	//queryPt[0] = ImageF[0][pix];
	//queryPt[1] = ImageF[1][pix];
	//queryPt[2] = ImageF[2][pix];
	//double value = 0.0;
	//switch (i){
	//case 0:
	//	kdTree_Back->annkSearch(						    // search
	//		queryPt,						// query point
	//		k_neighbors,					// number of near neighbors
	//		nnIdx,							// nearest neighbors (returned)
	//		dists,							// distance (returned)
	//		eps);							// error bound
	//	value = dists[0];
	//	break;
	//case 1:
	//	kdTree_Fore->annkSearch(						    // search
	//		queryPt,						// query point
	//		k_neighbors,					// number of near neighbors
	//		nnIdx,							// nearest neighbors (returned)
	//		dists,							// distance (returned)
	//		eps);							// error bound
	//	value = dists[0];
	//	break;
	//default:
	//	break;
	//}
	//return value;
	switch (i)
	{
	case 1:
		value = E1_1D.at<Vec2d>(pix, 0)[1];
		//value = 100;
		break;
	case 0:
		value = E1_1D.at<Vec2d>(pix, 0)[0]; 
		//value = -100;
		break;
	default:
		break;
	}
	return value;
}
MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{

	//float R = ImageF[0][pix1] - ImageF[0][pix2];
	//float G = ImageF[1][pix1] - ImageF[1][pix2];
	//float B = ImageF[2][pix1] - ImageF[2][pix2];
	//double C = R*R + G*G + B*B;
	//C = sqrt(C);
	//C = 1 / (1 + C);
	double R = (double)(srcimg1D.at<Vec3b>(pix1, 0)[2] - srcimg1D.at<Vec3b>(pix2, 0)[2]);
	double G = (double)(srcimg1D.at<Vec3b>(pix1, 0)[1] - srcimg1D.at<Vec3b>(pix2, 0)[1]);
	double B = (double)(srcimg1D.at<Vec3b>(pix1, 0)[0] - srcimg1D.at<Vec3b>(pix2, 0)[0]);
	double C = R*R + G*G + B*B;
	C = sqrt(C);
	C = 1 / (1 + C);
	return abs(i - j)*C * 300;//一定范围内 lameda越大边界惩罚越大，边界区分效果应该越好，lameda小了（我之前设为1）会出现蝴蝶的花斑等处出现空洞（前景里出现极小的蓝色背景，看着及其让人厌恶）
	return 0;

}// 计算E2 pixl pix2 为两个点的位置,pix1 pix2 为图的一维序号，ij为两个点的XI XJ(0或1)
Mat CutPhoto(Mat &srcImg, Mat & E1_1D);
int main()
{
	Mat srcImg = imread(IMAGE_PATH_PREFIX + "1.jpg");
    srcimg1D = cv1D(srcImg); 
	Mat markmat=UIdesign(srcImg);
	//E1_1D = coutdata(markmat, srcImg);
	imwrite(IMAGE_OUTPUT_PATH + "butterfly.jpg", CutPhoto(srcImg, coutdata(markmat, srcImg)));
	return 0;
}
void on_MouseCallBack(int event, int x, int y, int flags, void* params)//借用网上的交互式鼠标代码，opencv，自己改成了可以左右双键操作的
{
	Mat& img = *(Mat*)params;
	switch (event)
	{
	case EVENT_MOUSEMOVE:
	{
		if (g_bLeftButtonDown)
		{
			g_pCurrent = Point(x, y); // current mouse position

			//liftpLinePoints.push_back(g_pCurrent); // add current mouse position

			line(img, g_pPrevious, g_pCurrent, Scalar(0, 0, 255), 8, 8); // draw line on the input image 前景用红色画在原图的副本上和空白纸上
			line(g_mMaskImg, g_pPrevious, g_pCurrent, Scalar(0, 0, 255), 8, 8); // draw line on the mask image

			g_pPrevious = g_pCurrent; // after drawing, current mouse position should be previous mouse position of next movement
		}
		if (g_brightButtonDown)
		{
			g_pCurrent = Point(x, y); // current mouse position
			//rightpLinePoints.push_back(g_pCurrent); // add current mouse position

			line(img, g_pPrevious, g_pCurrent, Scalar(255,0, 0 ), 8, 8); // 后景用蓝色画在原图的副本上//b g r
			line(g_mMaskImg, g_pPrevious, g_pCurrent, Scalar( 255,0, 0), 8, 8); // 讲画的标记线画在空白纸上
			g_pPrevious = g_pCurrent; // after drawing, current mouse position should be previous mouse position of next movement
		}
	}
		break;
	case EVENT_LBUTTONDOWN:
	{
		g_bLeftButtonDown = true;

		g_pPrevious = Point(x, y);

		//liftpLinePoints.push_back(g_pPrevious);
	}
		break;
	case EVENT_LBUTTONUP:
	{
		g_bLeftButtonDown = false;
	}
		break;
	case EVENT_RBUTTONDOWN:
	{
		g_brightButtonDown = true;

		g_pPrevious = Point(x, y);

		//rightpLinePoints.push_back(g_pPrevious);
	}
		break;
	case EVENT_RBUTTONUP:
	{
		g_brightButtonDown = false;
	}
		break;


	}
}
Mat UIdesign(const Mat&  srcImg)
{

	Mat tempImg;//复制srcimage用于操作的
	Size srcSize = srcImg.size();
	g_mMaskImg = Mat(srcSize.height, srcSize.width, CV_8UC3);
	g_mMaskImg = Scalar::all(0);

	srcImg.copyTo(tempImg);

	namedWindow(MainWindowName);
	namedWindow(MaskWindowName);

	// set mouse call back function
	setMouseCallback(MainWindowName, on_MouseCallBack, (void*)&tempImg);

	while (1)
	{
		imshow(MainWindowName, tempImg);
		imshow(MaskWindowName, g_mMaskImg);

		if (waitKey(10) == 27)//每个循环等待10ms，有按ESC键则退出
		{
			break;
		}
	}
	imwrite(IMAGE_OUTPUT_PATH + "mask1.jpg", tempImg);
	//imwrite(IMAGE_OUTPUT_PATH + "mask2.jpg", g_mMaskImg);
	return  g_mMaskImg;

}
vector<Mat>Kmeans(Mat &mark, Mat &srcImg)
{
	vector<Vec3b>seedFsample;
	vector<Vec3b>seedBsample;
	//Mat test(srcImg.size(),srcImg.type());
	for (int i = 0; i < (int)mark.rows; i++)
		for (int j = 0; j <(int)mark.cols; j++)
		{
		if (mark.at<Vec3b>(i, j)[2] == 255 && mark.at<Vec3b>(i, j)[1] == 0)
			seedFsample.push_back(srcImg.at<Vec3b>(i, j));
		if (mark.at<Vec3b>(i, j)[0] == 255 && mark.at<Vec3b>(i, j)[1] == 0)
			seedBsample.push_back(srcImg.at<Vec3b>(i, j));
		}
	if (seedBsample.size() == 0 || seedFsample.size() == 0)
	{
		cout << "please draw at least one fore ground line and one back ground line!";
		getchar();
		exit(0);
	}
	//RNG rng(12345);
	Mat seedF = Mat::zeros(seedFsample.size(), 1, CV_32FC3);//创建样本矩阵
	Mat seedB = Mat::zeros(seedBsample.size(), 1, CV_32FC3);//创建样本矩阵
	for (int i = 0; i <(int) seedFsample.size(); i++)
		seedF.at<Vec3f>(i,0) = seedFsample[i];
	for (int i = 0; i <(int)seedBsample.size(); i++)
		seedB.at<Vec3f>(i,0) = seedBsample[i];
	Mat clustersF;
	Mat clustersB;
	//randShuffle(seedF, 1, &rng); //因为要聚类，所以先随机打乱里面的点
	//randShuffle(seedB, 1, &rng);
	//聚类，KMEANS PP CENTERS Use kmeans++ center initialization by Arthur and Vassilvitskii  
	kmeans(seedF, K_MEANS_CLUSTERS_NUM, clustersF, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1), 2, KMEANS_PP_CENTERS);
	kmeans(seedB, K_MEANS_CLUSTERS_NUM, clustersB, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1), 2, KMEANS_PP_CENTERS);
	vector<	Mat>result;
	result.push_back(clustersF);//0
	result.push_back(clustersB);//1
	result.push_back(seedF);//2
	result.push_back(seedB);//3
	return result;
}
Mat coutdata(Mat &mark, Mat &srcImg)
{
	//for (int i = 0; i < (int)mark.rows; i++)
	//	for (int j = 0; j <(int)mark.cols; j++)
	//	{
	//	if (mark.at<Vec3b>(i, j)[2] == 255 && mark.at<Vec3b>(i, j)[1] == 0)
	//		labelMat.at<uchar>(i, j)=1;
	//	if (mark.at<Vec3b>(i, j)[0] == 255 && mark.at<Vec3b>(i, j)[1] == 0)
	//		labelMat.at<uchar>(i, j) = 0;
	//	}
	//
	Mat datamat(srcImg.size(), CV_64FC2);//包含每个点的e1（1） e1（0）
	Mat distanceMat(srcImg.size(), CV_64FC2);//包含每个点的dif dib
	KmeansResult = Kmeans(mark, srcImg);
	Vec3d  meanKnF[K_MEANS_CLUSTERS_NUM], meanKnB[K_MEANS_CLUSTERS_NUM];
	int pointnum_f = 0, pointnum_b = 0;
	for (int i = 0; i < K_MEANS_CLUSTERS_NUM; i++)
	{
		pointnum_f = 0; pointnum_b = 0;
		for (int m = 0; m < (int)KmeansResult[0].rows; m++)//64个 knf类
		{
			 if (KmeansResult[0].at<int>(m) == i)
			 {
				meanKnF[i] += KmeansResult[2].at<Vec3f>(m, 0);
				pointnum_f++;
			 }
		}
		for (int m = 0; m <(int)KmeansResult[1].rows; m++)//64个 knb类
		{
			if (KmeansResult[1].at<int>(m) == i)
			{
				meanKnB[i] += KmeansResult[3].at<Vec3f>(m, 0);
				pointnum_b++;
			}
		}
		meanKnF[i] = meanKnF[i] / pointnum_f;
		meanKnB[i] = meanKnB[i] / pointnum_b;
	}//计算第64个类的平均color: knf knb
	
	for (int i = 0; i <(int)distanceMat.rows; i++)
		for (int j = 0; j< (int)distanceMat.cols; j++)
		{
		if ((mark.at<Vec3b>(i, j)[2] == 255 && mark.at<Vec3b>(i, j)[1] == 0) || (mark.at<Vec3b>(i, j)[0] == 255 && mark.at<Vec3b>(i, j)[1] == 0))
			continue;
		double mindistacnef = (double) INFINITY, distacnef;
		double mindistacneb = (double)INFINITY, distacneb;
		for (int k = 0; k < 64; k++)//求min||ci-knf|| ，min||ci-knb||
		{
			double R = (double)(srcImg.at<Vec3b>(i, j)[2] - meanKnF[k][2]);
			double G = (double)(srcImg.at<Vec3b>(i, j)[1] - meanKnF[k][1]);
			double B = (double)(srcImg.at<Vec3b>(i, j)[0] - meanKnF[k][0]);
			distacnef = sqrt(R*R + G*G + B*B);// || ci - knf ||
			R = (double)(srcImg.at<Vec3b>(i, j)[2] - meanKnB[k][2]);
			G = (double)(srcImg.at<Vec3b>(i, j)[1] - meanKnB[k][1]);
			B = (double)(srcImg.at<Vec3b>(i, j)[0] - meanKnB[k][0]);
			distacneb = sqrt(R*R + G*G + B*B);//||ci-knb||
			if (mindistacnef > distacnef)
			{
				mindistacnef = distacnef;
			}
			if (mindistacneb > distacneb)
			{
				mindistacneb = distacneb;
			}
		}
		distanceMat.at<Vec2d>(i, j)[1] = mindistacnef;//dif
		distanceMat.at<Vec2d>(i, j)[0] = mindistacneb;//dib
		}
	for (int i = 0; i < (int)srcImg.rows; i++)
		for (int j = 0; j < (int)srcImg.cols; j++)
		{
		if ((mark.at<Vec3b>(i, j)[2] == 255 && mark.at<Vec3b>(i, j)[1] == 0))//前景点
		{
			datamat.at<Vec2d>(i, j)[1] = 0.0;
			datamat.at<Vec2d>(i, j)[0] = (double)INFINITY;
			continue;
		}
		if ((mark.at<Vec3b>(i, j)[0] == 255 && mark.at<Vec3b>(i, j)[1] == 0))//后景点
		{
			datamat.at<Vec2d>(i, j)[1] = (double)INFINITY;
			datamat.at<Vec2d>(i, j)[0] = 0.0;
			continue;
		}
		datamat.at<Vec2d>(i, j)[1] = distanceMat.at<Vec2d>(i, j)[1] / (distanceMat.at<Vec2d>(i, j)[1] + distanceMat.at<Vec2d>(i, j)[0]);//前景
		datamat.at<Vec2d>(i, j)[0] = distanceMat.at<Vec2d>(i, j)[0] / (distanceMat.at<Vec2d>(i, j)[1] + distanceMat.at<Vec2d>(i, j)[0]);//后景
		}
	Mat mat1D(datamat.cols*datamat.rows, 1, CV_64FC2);
	int k = 0;
	for (int i = 0; i < (int)datamat.rows; i++)
		for (int j = 0; j <(int)datamat.cols; j++, k++)
			mat1D.at<Vec2d>(k, 0) = datamat.at<Vec2d>(i, j);
	return mat1D;
}
Mat cv1D(Mat &mark)
{
	Mat mat1D(mark.cols*mark.rows, 1, mark.type());
	int k = 0;
	for (int i = 0; i <(int)mark.rows; i++)
		for (int j = 0; j <(int)mark.cols; j++, k++)
		mat1D.at<Vec3b>(k, 0) = mark.at<Vec3b>(i, j);
	return mat1D;
}
Mat CutPhoto(Mat &srcImg, Mat & E1_1D)
{
	MRF* mrf;
	uchar *labels;
	labels = new uchar[srcImg.cols* srcImg.rows];
	EnergyFunction *energy;
	DataCost *data = new DataCost(dCost);
	SmoothnessCost *smooth = new SmoothnessCost(fnCost);
	energy = new EnergyFunction(data, smooth);
	float t;
	mrf = new MaxProdBP(srcImg.cols, srcImg.rows, 2, energy);
	mrf->initialize();
	mrf->clearAnswer();
	mrf->optimize(3, t); // run for 5 iterations, store time t it took 
	MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
	MRF::EnergyVal E_data = mrf->dataEnergy();
	for (int pix = 0; pix < srcImg.cols* srcImg.rows; pix++) {
		labels[pix] = mrf->getLabel(pix);//get the lables for each pixel
	}
	Mat finalmat(srcImg.size(), CV_8UC3);
	int k = 0;
	for (int i = 0; i < finalmat.rows; i++)
		for (int j = 0; j < finalmat.cols; j++, k++)
		{
		if (labels[k] == 1)
			finalmat.at<Vec3b>(i, j) = srcImg.at<Vec3b>(i, j);
		if (labels[k] == 0)
			finalmat.at<Vec3b>(i, j) = 255;
		}
	imshow("final", finalmat);	
	waitKey(0);
	return finalmat;

}