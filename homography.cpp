// homography.cpp : 定义控制台应用程序的入口点。
//备注：这次试验做的不是很好，主要在于：超定方程求解上。ransac后如果用自己的svd：：slovez与标准库ransac误差在5左右，用svd分解误差在1左右，用非ransac标准库在0.5左右。具体问题应该不在我的ransac构架上了，而在于
//超定方程求解上，日后可以继续研究一下，另外这个代码鲁棒性很差，使用文件目录下的两张平移bmp图，误差及其明显在5到10左右。但用两张旋转图效果又非常之好，对一般的风景图效果也较为不错。原理上已经了然：1求homo：把原有
//点坐标为未知数的等式转换为9个homo未知数的方程，当点对有多对时这就成了超定方程，svd法解超定方程（会用就ok），值得注意的是不论是矩阵计算还是算距离的时候：归一化（坐标的z变为1）
//2ransac就是利用随机抽样一致性通过不断随机抽取，迭代出最好的模型（内点最多的模型）

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<iostream>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#define therrhold 3
cv::Mat FindHomography(cv::Mat &srcimage1, cv::Mat &srcimage2);
cv::Mat FindHomoInRansacWay(cv::Mat &srcimage1, cv::Mat &srcimage2, int times);
int main()
{
	cv::Mat image1 = cv::imread("1.bmp");
	cv::Mat image2 = cv::imread("2.bmp");
	cv::Mat srcimage1, srcimage2;
	resize(image1, srcimage1, cv::Size(), 0.5, 0.5);
	resize(image2, srcimage2, cv::Size(), 0.5, 0.5);
	std::cout << "ransac：" << FindHomoInRansacWay(srcimage1, srcimage2, 2000) << std::endl;
	cv::waitKey(0);
	getchar();
	return 0;
	//Mat image_1 = imread("1.jpg", 1);
	//Mat image_2 = imread("2.jpg", 1);
	//Mat image1, image2;
	//resize(image_1, image1, Size(), 0.5, 0.5);
	//resize(image_2, image2, Size(), 0.5, 0.5);
	//Ptr<FeatureDetector> detector = new SurfFeatureDetector(300);//探测器
	//Ptr<DescriptorExtractor>extractor = new SurfFeatureDetector;//匹配器
	//vector<DMatch> matches;//匹配器用来装特征向量
	//vector<KeyPoint>keyPoint1, keyPoint2;//存放两幅图片特征点
	////第一步：SURF方式找到图像的特征点
	//detector->detect(image1, keyPoint1);
	//detector->detect(image2, keyPoint2);
	////画出找到的特征点
	//Mat img_KeyPoint_1, img_KeyPoint_2;
	//drawKeypoints(image1, keyPoint1, img_KeyPoint_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//drawKeypoints(image2, keyPoint2, img_KeyPoint_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("点图1", img_KeyPoint_1);
	//imshow("点图2", img_KeyPoint_2);
	////第二步：通过特征点找到特征向量，也就是描述子descriptions
	//Mat  descriptions1, descriptions2;
	//extractor->compute(image1, keyPoint1, descriptions1);
	//extractor->compute(image2, keyPoint2, descriptions2);
	////cv::BFMatcher matcher(NORM_L2, true);//实例化一个匹配器
	//FlannBasedMatcher matcher;
	//matcher.match(descriptions1, descriptions2, matches);//匹配向量存放在<容器matches>中
	//double max_dist = 0; double min_dist = 100;
	//for (int i = 0; i < descriptions1.rows; i++)
	//{
	//	double dist = matches[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist)  max_dist = dist;
	//}
	//vector<DMatch> good_matches;
	//for (int i = 0; i < descriptions1.rows; i++)
	//{
	//	if (matches[i].distance < 2 * min_dist)
	//	{
	//		good_matches.push_back(matches[i]);
	//	}
	//}
	//std::cout << matches.size() << std::endl;
	//getchar();
	// return 0;
}

cv::Mat FindHomography( std::vector<cv::Point2f >& srclist1,  std::vector<cv::Point2f> &srclist2) 
{
	int n = std::min(srclist1.size(), srclist2.size());
	cv::Mat mtx=cv::Mat::zeros(2*n, 9, CV_64FC1);
	for (int i = 0; i <= 2*n-1; i += 2)
	{
		mtx.at<double>(i, 0) = srclist1[(i + 2) / 2-1].x;
		mtx.at<double>(i, 1) = srclist1[(i + 2) / 2-1].y;
		mtx.at<double>(i, 2) = 1.0;
		mtx.at<double>(i, 6) = -(srclist1[(i + 2) / 2-1].x*srclist2[(i + 2) / 2-1].x);
		mtx.at<double>(i, 7) = -(srclist1[(i + 2) / 2-1].y*srclist2[(i + 2) / 2-1].x);
		mtx.at<double>(i, 8) = -srclist2[(i + 2) / 2-1].x;
	}
	for (int i = 1; i <= 2 * n-1; i += 2)
	{
		mtx.at<double>(i, 3) = srclist1[(i + 1) / 2 - 1].x;
		mtx.at<double>(i, 4) = srclist1[(i + 1) / 2 - 1].y;
		mtx.at<double>(i, 5) = 1.0;
		mtx.at<double>(i, 6) =  -(srclist1[(i + 1) / 2 - 1].x*srclist2[(i + 1) / 2 - 1].y);
		mtx.at<double>(i, 7) = 0 - (srclist1[(i + 1) / 2 - 1].y*srclist2[(i + 1) / 2 - 1].y);
		mtx.at<double>(i, 8) = 0 - srclist2[(i + 1) / 2 - 1].y;
	}
	//cv::Mat result = cv::Mat::zeros(9,1, CV_64FC1);
	//cv::SVD::solveZ(mtx, result);
	//result = result / (result.at<double>(8,0));
	//cv::Mat B = result.reshape(0, 3);
	//return B;
	cv::SVD thissvd(mtx, cv::SVD::FULL_UV);
	 cv::Mat U = thissvd.u;
	 cv::Mat S = thissvd.w;
	 cv::Mat VT = thissvd.vt;
	 cv::Mat V = VT.t();
	 cv::Mat result = cv::Mat::zeros(9, 1, CV_64FC1);
	 for (int i = 0; i < 9; i++)
	 result.at<double>(i, 0) = V.at<double>(i, V.cols - 1);
	 result = result / (result.at<double>(8, 0));
	 cv::Mat B = result.reshape(0, 3);
	 return B;//svd分解法会比slovez准确一些
}
cv::Mat FindHomoInRansacWay(cv::Mat &srcimage1, cv::Mat &srcimage2,int times)
{  //SURF检测器检测特征点
	int minHessian = 300;
	cv::SURF detector(minHessian);
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	detector.detect(srcimage1, keypoints_1);
	detector.detect(srcimage2, keypoints_2);
	//计算SURF描述子（由特征点和原图构成的特征向量）
	cv::SURF extractor;
	cv::Mat descriptors_1, descriptors_2;
	extractor.compute(srcimage1, keypoints_1, descriptors_1);
	extractor.compute(srcimage2, keypoints_2, descriptors_2);
	//采用FLANN匹配
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors_1, descriptors_2,matches);
	//cv::Mat img_matces;
	//cv::drawMatches(srcimage1, keypoints_1, srcimage2, keypoints_2, matches, img_matces, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//cv::imshow("a",img_matces);
	//std::cout << " matches"<<matches.size()<<std::endl;	
	std::vector<cv::Point2f>list1, list2;
	for (int i = 0; i<(int)matches.size(); i++)
	{
			list1.push_back(keypoints_1[matches[i].queryIdx].pt);
			list2.push_back(keypoints_2[matches[i].trainIdx].pt);
		}
		std::cout << std::endl << "标准库ransac：" << std::endl << cv::findHomography(list1, list2, CV_RANSAC, therrhold) << std::endl;
	    list1.clear();
		list2.clear();
/////////////////////////////ransac/////////////////////////////////
	int  maxnumber=0;
	std::vector<cv::DMatch> Finalmatchlist,templist;
	srand((unsigned)(time(NULL)));
	while (times--)
	{
		bool noagain=false;
		int num[4], exnum[4] = {0,0,0,0};
		do
		{
			noagain = false;
			num[0] = rand() % (matches.size());
			num[1] = rand() % (matches.size());
			num[2] = rand() % (matches.size());
			num[3] = rand() % (matches.size());
			if (num[0] != exnum[0] && num[1] != exnum[1] && num[2] != exnum[2] && num[3] != exnum[3])
			{
				noagain = true;
				exnum[0] = num[0];
				exnum[1] = num[1];
				exnum[2] = num[2];
				exnum[3] = num[3];
			}
			//std::cout << num[0] << " " << num[1] << " " << num[2] << " " << num[3] << std::endl;
		}
		while (num[0] == num[1]||noagain==false);
		std::vector<cv::Point2f> srclist1, srclist2;
		for (int i = 0; i < 4; i++)
		{
			srclist1.push_back(keypoints_1[matches[num[i]].queryIdx].pt);
			srclist2.push_back(keypoints_2[matches[num[i]].trainIdx].pt);
		}
		cv::Mat homo = FindHomography(srclist1, srclist2);
		   srclist1.clear();
		   srclist2.clear();
		   for (int i = 0; i<(int)matches.size(); i++)
		{	double distance;
			cv::Mat pt=cv::Mat(3,1,CV_64FC1);
			pt.at<double>(0, 0) = keypoints_1[matches[i].queryIdx].pt.x;//x
			pt.at<double>(1, 0) = keypoints_1[matches[i].queryIdx].pt.y;//y
			pt.at<double>(2, 0) = 1.0;
			cv::Mat pt_img = homo*pt;//X'
			pt_img = pt_img / pt_img.at<double>(2,0);//归一化
			distance = pow((pt_img.at<double>(0, 0) - keypoints_2[matches[i].trainIdx].pt.x), 2) + pow((pt_img.at<double>(1, 0) - keypoints_2[matches[i].trainIdx].pt.y), 2);
		   distance = sqrt(distance);//欧氏距离
		   if (distance <therrhold)
			   templist.push_back(matches[i]);
		}
		//std:: cout << templist.size() << std::endl;
		if (templist.size() >maxnumber)
		{
			maxnumber = templist.size();
			Finalmatchlist.clear();
			for (int i = 0; i <(int)templist.size(); i++)
				Finalmatchlist.push_back(templist[i]);
		}templist.clear();
		
	}  
//////////////////////////得到最终计算的匹配列////////////////////////////////
	if (!Finalmatchlist.size())
	{
		std::cout << " wrong match process ,press any key to exit!";
		getchar();
		exit(1);
	}
	std::vector<cv::Point2f> srclist1, srclist2;
	for (int i = 0; i < (int)Finalmatchlist.size(); i++)
	{ 
		srclist1.push_back(keypoints_1[Finalmatchlist[i].queryIdx].pt);
		srclist2.push_back(keypoints_2[Finalmatchlist[i].trainIdx].pt);
	}	
	std::cout << "matches" << matches.size() << std::endl;
	std::cout << "Finalmatchlist" << Finalmatchlist.size() << std::endl;
	return cv::findHomography(srclist1, srclist2);//最后应用自己的svd分解求解超定方程，但会导致比标准库偏一些。
}
