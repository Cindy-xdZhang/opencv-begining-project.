/// 全景拼接.cpp : 定义控制台应用程序的入口点。
//*下面注释掉网上的库函数例程代码*/
//bool try_use_gpu = false;
//vector<Mat> imgs;
//using namespace std;	
//using namespace cv;
//string IMAGE_PATH_PREFIX = "data1/";
//Mat img = imread(IMAGE_PATH_PREFIX+"1.jpg");
//	imgs.push_back(img);
//	img = imread(IMAGE_PATH_PREFIX + "2.jpg");
//	imgs.push_back(img);
//	img = imread(IMAGE_PATH_PREFIX + "3.jpg");
//	imgs.push_back(img);
//	img = imread(IMAGE_PATH_PREFIX + "4.jpg");
//	imgs.push_back(img);
//	img = imread(IMAGE_PATH_PREFIX + "5.jpg");
//	imgs.push_back(img);
//	img = imread(IMAGE_PATH_PREFIX + "6.jpg");
//	imgs.push_back(img);
//	cout << "finish "<<imgs.size()<<endl;
//	Mat pano;//拼接结果图片
//	//Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
//	Stitcher stitcher = Stitcher::createDefault(true);
//	Stitcher::Status status = stitcher.stitch(imgs, pano);
//	if (status != Stitcher::OK)
//	{
//		cout << "Can't stitch images, error code = " << int(status) << endl;
//		return -1;
//	}
//	imwrite(result_name, pano); 
//	imwrite(result_name, pano);
//	waitKey(0);
//	return 0;

//void CalcCorners(const Mat& H, const Mat& src)
//{
//	double v2[] = { 0, 0, 1 };//左上角
//	double v1[3];//变换后的坐标值
//	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//
//	V1 = H * V2;
//	//左上角(0,0,1)
//	corners.left_top.x = v1[0] / v1[2];
//	corners.left_top.y = v1[1] / v1[2];
//
//	//左下角(0,src.rows,1)
//	v2[0] = 0;
//	v2[1] = src.rows;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.left_bottom.x = v1[0] / v1[2];
//	corners.left_bottom.y = v1[1] / v1[2];
//
//	//右上角(src.cols,0,1)
//	v2[0] = src.cols;
//	v2[1] = 0;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.right_top.x = v1[0] / v1[2];
//	corners.right_top.y = v1[1] / v1[2];
//
//	//右下角(src.cols,src.rows,1)
//	v2[0] = src.cols;
//	v2[1] = src.rows;
//	v2[2] = 1;
//	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
//	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
//	V1 = H * V2;
//	corners.right_bottom.x = v1[0] / v1[2];
//	corners.right_bottom.y = v1[1] / v1[2];
//
//}

//typedef struct
//{
//	Point2f left_top;
//	Point2f left_bottom;
//	Point2f right_top;
//	Point2f right_bottom;
//}four_corners_t;
//four_corners_t corners;
#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<iostream>
#include <opencv2/stitching/stitcher.hpp>
using namespace cv;
using namespace std;
string IMAGE_PATH_PREFIX = "data1/";
cv::Mat Stitcher2pic(const cv::Mat &srcimage1, const cv::Mat &srcimage2);
cv::Mat Stitcher3pic(cv::Mat &srcimage1, cv::Mat &srcimage2, cv::Mat& image1_2);
std::vector<cv::Mat> imgs;
cv::Mat imagedisplacement(cv::Mat &src, int x, int y);
cv::Mat Stitcher4pic(cv::Mat &srcimage1, cv::Mat &srcimage2, cv::Mat& image123);
cv::Mat Stitcher6pic(cv::Mat &srcimage1, cv::Mat &srcimage2);
int main()
 {
	Mat img = imread(IMAGE_PATH_PREFIX + "1.jpg");
	resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 img = imread(IMAGE_PATH_PREFIX + "2.jpg");	
	 resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 img = imread(IMAGE_PATH_PREFIX + "3.jpg");	
	 resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 img = imread(IMAGE_PATH_PREFIX + "4.jpg");	
	 resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 img = imread(IMAGE_PATH_PREFIX + "5.jpg");	
	 resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 img = imread(IMAGE_PATH_PREFIX + "6.jpg");	
	 resize(img, img, cv::Size(), 0.5, 0.5);
	 imgs.push_back(img);
	 cout << "finish " << imgs.size() << endl;
	 //imgs[1] = imagedisplacement(imgs[1], 100, 100);
	 cv::Mat k1 = Stitcher2pic(imgs[1], imgs[2]);
	 cv::Mat k2= Stitcher3pic(imgs[5], imgs[2], k1);
	// imshow("k2", k2);//pic236
	 cv::Mat k3 = Stitcher2pic(imgs[0], imgs[3]);//pic145
	 cv::Mat k4 = Stitcher3pic(imgs[4], imgs[3], k3);
	 k4 = imagedisplacement(k4 ,-380,0);
	// imshow("k4", k4);
	 cv::Mat k5 = Stitcher6pic(k2,k4);
//	 imshow("k5", k5);
	 
	 waitKey(0);

	 return 0;
 }

//int main()
//{
//
//	cv::Mat img = cv::imread(IMAGE_PATH_PREFIX + "1.jpg"); 
//	resize(img, img, cv::Size(), 0.5, 0.5);
//		imgs.push_back(img);	
//    img = cv::imread(IMAGE_PATH_PREFIX + "2.jpg");
//	resize(img, img, cv::Size(), 0.5, 0.5);
//	imgs.push_back(img); 
//	img = cv::imread(IMAGE_PATH_PREFIX + "3.jpg");
//	resize(img, img, cv::Size(), 0.5, 0.5);
//	imgs.push_back(img);
//	img = cv::imread(IMAGE_PATH_PREFIX + "4.jpg");
//	resize(img, img, cv::Size(), 0.5, 0.5);
//	imgs.push_back(img);
//	cv::Mat k1=Stitcher2pic(imgs[0], imgs[1]);
//	k1=imagedisplacement(k1, 500, 300);
//	k1 = Stitcher3pic(imgs[2], imgs[1], k1);
//	Stitcher4pic(imgs[3], imgs[1], k1);
//	cv::waitKey(0);
//
//}
cv::Mat imagedisplacement(cv::Mat &src,int x,int y)
{
	cv::Mat dst;
	cv::Size dst_sz = src.size();
	//定义平移矩阵  
	cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
	t_mat.at<float>(0, 0) = 1;
	t_mat.at<float>(0, 2) = x;
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) =y;
	cv::warpAffine(src, dst, t_mat, dst_sz);
	//cv::imshow("result", dst);
	return dst;
	cv::waitKey(0);
}
cv::Mat Stitcher2pic(const cv::Mat &srcimage1, const cv::Mat &srcimage2)//完成两张图的匹配
{ //SURF检测器检测特征点
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
	std::vector<cv::DMatch> matches, goodmatches;
	matcher.match(descriptors_1, descriptors_2, matches);
	std::vector<cv::Point2f>list1, list2;
	double min_dist = 100;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < min_dist) min_dist = matches[i].distance;
	}
	for (int i = 0; i < descriptors_1.rows; i++)//筛选好的匹配点
	{
		if (matches[i].distance < 4 * min_dist)
		{
			goodmatches.push_back(matches[i]);
		}
	}

	for (int i = 0; i < (int)goodmatches.size(); i++)
	{
		list1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
		list2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
	}
	//计算homography
	cv::Mat h = cv::findHomography(list1, list2, CV_RANSAC, 3);
	list1.clear();
	list2.clear();
	//cv::Mat AdjustMat = (cv::Mat_<double>(3, 3) << 1.0, 0,0, 0, 1.0, 0, 0, 0, 1.0);
	//cv::Mat adhomo=AdjustMat*h;
	//图像配准  
	cv::Mat imageTransform1;
	warpPerspective(srcimage1, imageTransform1, h, cv::Size(srcimage2.cols * 2, srcimage2.rows * 2));
	//imshow("直接经过透视矩阵变换", imageTransform1);
	cv::Mat dst(srcimage2.rows + srcimage1.rows, srcimage2.cols + srcimage1.cols, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(cv::Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	srcimage2.copyTo(dst(cv::Rect(0, 0, srcimage1.cols, srcimage1.rows)));
	//cv::imshow("b_dst1", dst);
	return dst;
}
//	////------------------------------------------------------------------------
//	//CalcCorners(h, srcimage1);
//	//int start = MAX(corners.left_top.x, corners.left_bottom.x);
//	//double processWidth = srcimage2.cols - start;//重叠区域的宽度  
//	//std::cout << "processWidth" << processWidth<<endl;
//	//std::cout << "Width" << dst.cols << endl;
//	//int rows = min(srcimage2.rows, imageTransform1.rows);
//	//for (int i = 0; i <rows; i++)
//	//{
//	//	for (int j = 0; j < processWidth; j++)
//	//	{
//	//		if (!imageTransform1.at<Vec3b>(i, j + start)[1] )
//	//			continue;
//	//		dst.at<Vec3b>(i, j + start) = srcimage2.at<Vec3b>(i, j + start)*(double)((processWidth - j) / processWidth) + imageTransform1.at<Vec3b>(i, j + start)*(double)(j / processWidth);
//	//	}
//	//}
//	//double processWidth = srcimage2.cols/8;
//	////int rows = min(srcimage2.rows, imageTransform1.rows);
//	////for (int i = 0; i < rows; i++)
//	////	{
//	////		for (int j = 0; j < processWidth*2; j++)
//	////		{
//	////			dst.at<Vec3b>(i, j + srcimage2.cols - processWidth) = srcimage2.at<Vec3b>(i, j + srcimage2.cols - processWidth)*0.5 + imageTransform1.at<Vec3b>(i, j + srcimage2.cols - processWidth)*0.5;
//	////			//dst.at<Vec3b>(i, j + start + 10) = srcimage2.at<Vec3b>(i, j + start + 10)*0.5 + imageTransform1.at<Vec3b>(i, j + start + 10)*0.5;
//	////				//cout << dst.at<Vec3b>(i, j + start) << endl;
//	////		}
//	////	}
//	//cv::imshow("dst", dst);
//}
//// Stitcher3pic改变了标准图2的位置了，因此Stitcher4pic直接算homo不需要先平移标准视角图后再算
cv::Mat Stitcher3pic(cv::Mat &srcimage1,  cv::Mat &srcimage2,cv::Mat& image1_2)
{
	image1_2 = imagedisplacement(image1_2, 400, 200);
    cv::Mat temp(srcimage2.rows + srcimage1.rows, srcimage2.cols + srcimage1.cols, CV_8UC3);
	srcimage2.copyTo(temp(cv::Rect(0, 0, srcimage2.cols, srcimage2.rows)));
   srcimage2 = imagedisplacement(temp, 400, 200);
	// SURF检测器检测特征点
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
	std::vector<cv::DMatch> matches, goodmatches;
	matcher.match(descriptors_1, descriptors_2, matches);
	std::vector<cv::Point2f>list1, list2;
	double min_dist = 100;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < min_dist) min_dist = matches[i].distance;
	}
	for (int i = 0; i < descriptors_1.rows; i++)//筛选好的匹配点
	{
		if (matches[i].distance < 4 * min_dist)
		{
			goodmatches.push_back(matches[i]);
		}
	}

	for (int i = 0; i < (int)goodmatches.size(); i++)
	{
		list1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
		list2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
	}
	//计算homography
	cv::Mat h = cv::findHomography(list1, list2, CV_RANSAC, 3);
	list1.clear();
	list2.clear();
	/*cv::Mat AdjustMat = (cv::Mat_<double>(3, 3) << 1.0, 0,200, 0, 1.0, 100, 0, 0, 1.0);
	cv::Mat adhomo=AdjustMat*h;*/
	//图像配准  
	cv::Mat imageTransform1;
	warpPerspective(srcimage1, imageTransform1, h, cv::Size(srcimage1.cols*2.2, srcimage1.rows *2.2));
	//imshow("直接经过透视矩阵变换", imageTransform1);
	cv::Mat dst(srcimage2.rows + srcimage1.rows, srcimage2.cols + srcimage1.cols, CV_8UC3);
	
	imageTransform1.copyTo(dst(cv::Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	for (int i = 0; i < image1_2.rows; i++)
		for (int j = 0; j < image1_2.cols; j++)
		{
		if (!image1_2.at<Vec3b>(i, j)[0])
			continue;
		dst.at<Vec3b>(i, j) = image1_2.at<Vec3b>(i, j);
		}
	//cv::imshow("333", dst);
	return dst;
	
}
//cv::Mat Stitcher4pic(cv::Mat &srcimage1, cv::Mat &srcimage2, cv::Mat& image123)
//{
//	cv::Mat temp(image123.rows, image123.cols , CV_8UC3);
//	srcimage2.copyTo(temp(cv::Rect(0, 0, srcimage2.cols , srcimage2.rows)));
//	/*srcimage2 = imagedisplacement(srcimage2, 100, 100);
//	image123 = imagedisplacement(image123, 100, 100);*/
//	//cv::imshow("b", image123);
//	// SURF检测器检测特征点
//	int minHessian = 300;
//	cv::SURF detector(minHessian);
//	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
//	detector.detect(srcimage1, keypoints_1);
//	detector.detect(srcimage2, keypoints_2);
//	//计算SURF描述子（由特征点和原图构成的特征向量）
//	cv::SURF extractor;
//	cv::Mat descriptors_1, descriptors_2;
//	extractor.compute(srcimage1, keypoints_1, descriptors_1);
//	extractor.compute(srcimage2, keypoints_2, descriptors_2);
//	//采用FLANN匹配
//	cv::FlannBasedMatcher matcher;
//	std::vector<cv::DMatch> matches, goodmatches;
//	matcher.match(descriptors_1, descriptors_2, matches);
//	std::vector<cv::Point2f>list1, list2;
//	double min_dist = 100;
//	for (int i = 0; i < descriptors_1.rows; i++)
//	{
//		if (matches[i].distance < min_dist) min_dist = matches[i].distance;
//	}
//	for (int i = 0; i < descriptors_1.rows; i++)//筛选好的匹配点
//	{
//		if (matches[i].distance < 6 * min_dist)
//		{
//			goodmatches.push_back(matches[i]);
//		}
//	}
//
//	for (int i = 0; i < (int)goodmatches.size(); i++)
//	{
//		list1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
//		list2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
//	}
//	//计算homography
//	cv::Mat h = cv::findHomography(list1, list2, CV_RANSAC, 3);
//	list1.clear();
//	list2.clear();
//	cv::Mat AdjustMat = (cv::Mat_<double>(3, 3) << 1.0, 0,41, 0, 1.0, 6, 0, 0, 1.0);//这个是调整位置（还没有解决BC两张图关于A的homography之后不能完全重叠的问题，按学长所说似乎本来就可能误差，只能这样抵消误差）
//	cv::Mat adhomo=AdjustMat*h;
//	//图像配准  
//	cv::Mat imageTransform1;
//	warpPerspective(srcimage1, imageTransform1, adhomo, cv::Size(srcimage1.cols*2.4, srcimage1.rows* 2.4));
//	//imshow("直接经过透视矩阵变换", imageTransform1);
//	cv::Mat dst(image123.rows, image123.cols, CV_8UC3);
//	imageTransform1.copyTo(dst(cv::Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
//	for (int i = 0; i < image123.rows; i++)
//		for (int j = 0; j < image123.cols; j++)
//		{
//		if (!image123.at<Vec3b>(i, j)[0])
//			continue;
//		dst.at<Vec3b>(i, j) = image123.at<Vec3b>(i, j);
//		}
//	/*image123.copyTo(dst(cv::Rect(0, 0, image123.cols, image123.rows)));
//	for (int i = 0; i <imageTransform1.rows; i++)
//		for (int j = 0; j <imageTransform1.cols-2; j++)
//	{
//	if (!imageTransform1.at<Vec3b>(i, j)[0])
//			continue;
//		dst.at<Vec3b>(i, j) = imageTransform1.at<Vec3b>(i, j);
//		}*/
//	//移动图像在幕布上的位置和调整大小，便于观察结果
//	dst = imagedisplacement(dst, 300, 50);
//	cv::Mat result(dst.rows - 200, dst.cols - 200, CV_8UC3);
//	for (int i = 100; i <dst.rows - 100; i++)
//		for (int j = 100; j <dst.cols - 100; j++)
//		{
//		result.at<Vec3b>(i - 100, j - 100) = dst.at<Vec3b>(i, j);
//		}
//	resize(result, result, cv::Size(), 0.5, 0.5);
//	cv::imwrite("dst.jpg", result);
//	cv::imshow("bst", result);
//	/*cv::Mat result(image123.rows-200, image123.cols-200, CV_8UC3);
//	for (int i = 100; i < dst.rows - 100; i++)
//		for (int j = 100; j < dst.cols - 100; j++)
//			result.at<Vec3b>(i-100, j-100) = dst.at<Vec3b>(i, j);
//			cv::imshow("b_dst", result);*/
//			return dst;
//}
cv::Mat Stitcher6pic(cv::Mat &srcimage1, cv::Mat &srcimage2)
{
	imshow("src1", srcimage1);
	imshow("src2", srcimage2);
	//SURF检测器检测特征点
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
	std::vector<cv::DMatch> matches, goodmatches;
	matcher.match(descriptors_1, descriptors_2, matches);
	std::vector<cv::Point2f>list1, list2;
	double min_dist = 100;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < min_dist) min_dist = matches[i].distance;
	}
	for (int i = 0; i < descriptors_1.rows; i++)//筛选好的匹配点
	{
		if (matches[i].distance < 4* min_dist)
		{
			goodmatches.push_back(matches[i]);
		}
	}
	cout << "find3" << endl;
	for (int i = 0; i < (int)goodmatches.size(); i++)
	{
		list1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
		list2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
	}
	//计算homography
	
	cout << "f" << list1.size() << list2.size() << endl;
	cv::Mat h = cv::findHomography(list1, list2, CV_RANSAC, 3);
	list1.clear();
	list2.clear();
	cv::Mat AdjustMat = (cv::Mat_<double>(3, 3) << 1.0, 0,-2, 0, 1.0, -1, 0, 0, 1.0);
	cv::Mat adhomo=AdjustMat*h;
	//图像配准  
	cv::Mat imageTransform1;
	warpPerspective(srcimage1, imageTransform1, adhomo, cv::Size(srcimage2.cols * 2, srcimage2.rows * 2));
	imshow("直接经过透视矩阵变换", imageTransform1);
	cv::Mat dst(srcimage2.rows + srcimage1.rows, srcimage2.cols + srcimage1.cols, CV_8UC3);
	dst.setTo(0);
	srcimage2.copyTo(dst(cv::Rect(0, 0, srcimage1.cols, srcimage1.rows)));
	for (int i = 0; i < imageTransform1.rows; i++)
		for (int j = 0; j < imageTransform1.cols; j++)
		{
		if (!imageTransform1.at<Vec3b>(i, j)[0])
			continue;
		dst.at<Vec3b>(i, j) = imageTransform1.at<Vec3b>(i, j);
		}
	
	cv::Mat result(dst.rows - 1300, dst.cols-2100, CV_8UC3);
	for (int i = 0; i <result.rows; i++)
		for (int j = 0; j <result.cols; j++)
		{
		
		result.at<Vec3b>(i, j) = dst.at<Vec3b>(i, j);
		}
	cv::imshow("b_dst1", result);
	imwrite("result.jpg", result);
	return dst;
}