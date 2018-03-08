// seam carving.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<vector>
#include<iostream>
#include<algorithm>
#define cutcols 20
#define cutrows 90
typedef struct PathSTRUCT
{
	std::vector<cv::Point2i> pathpoint;
	int value;
}Path;
typedef struct ValuePoint
{
cv::Point2i position;
	double value;
}ValuePoint;
inline bool Sortfiled(const ValuePoint &v1, const ValuePoint &v2)
{
	return v1.value<v2.value;//升序
}
inline cv::Point2i sortedfunction(std::vector<ValuePoint> &points);
inline void seamcarving_cols(cv::Mat &srcimage, cv::Mat& outputimage);
inline cv::Mat get_pointpathvalue(const cv::Mat &srcimage);
inline cv::Mat get_energy(const cv::Mat &srcimage);
inline std::vector<cv::Point2i> find_path(const cv::Mat &energypath);
inline void seamcarving_rows(cv::Mat &srcimage, cv::Mat& outputimage);
int main()
{
	register cv::Mat srcimage =cv:: imread("pic1.bmp");
	//cv::resize(srcimage, srcimage,cv::Size(),0.5,0.5);
	srcimage.convertTo(srcimage,CV_8UC3,1,0);
	register cv::Mat dstimage;
	imshow("原图",srcimage);
	if (cutcols>srcimage.cols-50 || cutrows > srcimage.rows-50)
	{
		std::cout << "裁剪过多像素，不可以完成，适度减小裁剪量,按回车退出程序！" << std::endl;
		getchar();
		return 1;
	}
	seamcarving_cols(srcimage, dstimage);
	seamcarving_rows(dstimage, dstimage);
	cv:: imshow("效果图", dstimage);
	cv:: imwrite("outputimage.bmp", dstimage);
	cv::waitKey(0);
	return 0;
}

void seamcarving_rows(cv::Mat &srcimage, cv::Mat& outputimage)
{
	cv::Mat tempsrc, tempimage;
	tempsrc=srcimage.t();
	//srcimage.copyTo(tempsrc);
	std::vector<cv::Point2i> pixellist;
	for (int n = 1; n <= cutrows; n++)
	{
		tempimage = cv::Mat(srcimage.cols, tempsrc.cols - 1, CV_8UC3);
		pixellist = find_path(get_pointpathvalue(get_energy(tempsrc)));
		for (int i = 0; i < srcimage.cols; i++)//第i行
		{
			for (int j1 = 0; j1 < pixellist[i].y; j1++)
				tempimage.at<cv::Vec3b>(i, j1) = tempsrc.at<cv::Vec3b>(i, j1);
			for (int j2 = pixellist[i].y; j2 < tempimage.cols; j2++)
				tempimage.at<cv::Vec3b>(i, j2) = tempsrc.at<cv::Vec3b>(i, j2 + 1);
			//imshow("lines", tempsrc);
		}
		pixellist.clear();
		tempimage.copyTo(tempsrc);
	}
	outputimage=tempsrc.t();

}
void seamcarving_cols(cv::Mat &srcimage, cv::Mat& outputimage)
{
	cv::Mat tempsrc, tempimage;
	srcimage.copyTo(tempsrc);
	std::vector<cv::Point2i> pixellist;	
	for (int n = 1; n <= cutcols; n++)
	{   tempimage = cv::Mat(srcimage.rows, tempsrc.cols - 1, CV_8UC3);
		pixellist = find_path(get_pointpathvalue(get_energy(tempsrc)));
		for (int i = 0; i < srcimage.rows; i++)//第i行
		{
			for (int j1 = 0; j1 < pixellist[i].y; j1++)
				tempimage.at<cv::Vec3b>(i, j1) = tempsrc.at<cv::Vec3b>(i, j1);
			for (int j2 = pixellist[i].y; j2 < tempimage.cols; j2++)
				tempimage.at<cv::Vec3b>(i, j2) = tempsrc.at<cv::Vec3b>(i, j2 + 1);
			//imshow("lines", tempsrc);
		}
	 pixellist.clear();
	tempimage.copyTo(tempsrc);
	}
tempsrc.copyTo(outputimage);

}
std::vector<cv::Point2i> find_path(const cv::Mat &energypath)//得到来自findpointpathvalue的 64FC1的 各像素点路径值矩阵 输出要操作的坐标点列
{

	Path path;//path容器
	std::vector<ValuePoint> sortedmachine;//排序机
	std::vector<cv::Point2i> result;
	ValuePoint temp;//临时点容器
	for (int i = 0; i < energypath.cols; i++)//first line
	{
		temp.position = cv::Point2i(0, i);
		temp.value = energypath.at<double>(0, i);
		sortedmachine.push_back(temp);
	}
	path.pathpoint.push_back(sortedfunction(sortedmachine));//path0
	sortedmachine.clear();
	for (int n = 0; n <energypath.rows - 1; n++)//path 1---energypath.rows
	{
		for (int i = 0; i <= 2; i++)
		{
			if (path.pathpoint[n].y - 1 + i >= energypath.cols || path.pathpoint[n].y - 1 + i < 0)
				continue;
			temp.position = cv::Point2i(path.pathpoint[n].x + 1, path.pathpoint[n].y - 1 + i);
			temp.value = energypath.at<double>(path.pathpoint[n].x + 1, path.pathpoint[n].y - 1 + i);
			sortedmachine.push_back(temp);
		}
		path.pathpoint.push_back(sortedfunction(sortedmachine));//path n+1
		sortedmachine.clear();
	}
	if (path.pathpoint.size() != energypath.rows)
	{
		std::cout << "path.pathpoint.size()=" << path.pathpoint.size() << "press to exit";
		getchar();
		exit(1);
	}
	for (int i = 0; i < energypath.rows; ++i)
		result.push_back(path.pathpoint[i]);
	return result;
} 
cv::Mat get_energy(const cv::Mat &srceimage)//src为三通道原彩图8UC3得到一个转化为1/255，double形单通道梯度能量图
{
	
	cv::Mat gradx, grady, abs_gradx, abs_grady, Energy, srcimage;
	cv::cvtColor(srceimage, srcimage, CV_BGR2GRAY);//非常关键
	cv::Sobel(srcimage, grady, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grady, abs_grady);
	cv::Sobel(srcimage, gradx, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gradx, abs_gradx);
	addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0, Energy);
	Energy.convertTo(Energy, CV_64FC1, 1.0/255.0, 0); 
	return Energy;
}
cv::Mat get_pointpathvalue(const cv::Mat &srcimage)//传入energy 单通道64F srcimage 
{

	cv::Mat pathvalue = cv::Mat::zeros(srcimage.size(), CV_64FC1);
	for (int i = 0; i < srcimage.cols; i++)//last row
		pathvalue.at<double>(srcimage.rows - 1, i) = (double)srcimage.at<double>(srcimage.rows - 1, i);
	for (int i = srcimage.rows - 2; i >= 0; --i)
		for (int j = 0; j < srcimage.cols; ++j)
		{
		if (j == 0)
		{
			pathvalue.at<double>(i, j) = std::min(pathvalue.at<double>(i + 1, j + 1), pathvalue.at<double>(i + 1, j)) + srcimage.at<double>(i, j);
			continue;
		}
		if (j == srcimage.cols - 1)
		{
			pathvalue.at<double>(i, j) = std::min(pathvalue.at<double>(i + 1, j - 1), pathvalue.at<double>(i + 1, j)) + srcimage.at<double>(i, j);
			continue;
		}
		pathvalue.at<double>(i, j) = std::min(std::min(pathvalue.at<double>(i + 1, j + 1), pathvalue.at<double>(i + 1, j)), pathvalue.at<double>(i + 1, j - 1)) + srcimage.at<double>(i, j);
		}
	return pathvalue;
}
cv::Point2i sortedfunction(std::vector<ValuePoint> &points)
{
	random_shuffle(points.begin(), points.end());
	sort(points.begin(), points.end(), Sortfiled);
	cv::Point2i position = points[0].position;
	return position;
}

