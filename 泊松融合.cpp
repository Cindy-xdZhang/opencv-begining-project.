// PossionImageEdting.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "engine.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<vector>
#include<iostream>
#include <Windows.h>
class Triplets {
public:
	int nmax; int m; int n; int* ri; int* ci;//ri ci 为第i个非零元素的行列坐标 nzval为对应的数值  nnz为非零元素总个数，应小于nmax，m，n为稀疏矩阵行列数
	double* nzval; int nnz;
public:
	Triplets() :
		nmax(0), m(-1), n(-1), ri(0), ci(0), nzval(0), nnz(0) {
	}
	~Triplets() {
		clear();
	}
	void clear() {
		if (nzval)
			delete[] nzval;
		if (ri)
			delete[] ri;
		if (ci)
			delete[] ci;

		m = n = -1;
		nnz = 0;
		nzval = 0;
		ri = ci = 0;
		nmax = 0;
	}
	void reserve(int rows, int cols, int num) {
		clear();
		m = rows;
		n = cols;
		nmax = num;
		nzval = new double[num];
		ri = new int[num];
		ci = new int[num];
	}
	void add(int r, int c, double val) {
		if (nnz >= nmax) {
			printf("Triplets::add - the maximun number of elements reached!");
			return;
		}
		ri[nnz] = r;
		ci[nnz] = c;
		nzval[nnz] = val;
		nnz++;
	}
public:
	//debug
	void print() {
		for (int i = 0; i < nnz; i++) {
			//			cout << "[" << i << "]: " << ri[i] << " " << ci[i] << " " << nzval[i] << std::endl;
		}
	}
};
void matlabSparseLinearSystem(Triplets& A, double b[], double x[]);//solve by matlab \ solver
#pragma comment(lib, "libeng.lib")
#pragma comment(lib, "libmx.lib")
#pragma comment(lib, "libmat.lib")
#define workingmode 2//1则截取矩形局域进行c++融合 2则截取矩形进行matlab求解
int sythesispositionx = 150;
int sythesispositiony = 320;
using namespace cv;
using namespace std;
string IMAGE_PATH_PREFIX = "image/";
string IMAGE_OUTPUT_PATH = "output/";
bool g_bLeftButtonDown = false;
bool g_brightButtonDown = false;
string MainWindowName = "Mouse operation";//主窗口（进行鼠标操作的原图窗口）
string MaskWindowName = "Mask Image";//附属窗口（记录鼠标操作的黑白窗口）
Mat g_mMaskImg;//记录划线的图
Point g_pPrevious; // previous point position
Point g_pCurrent; // current mouse position 这两句对点的声明如果放进了mousecallback函数里就会出问题
void on_MouseCallBack(int event, int x, int y, int flags, void* params);
Mat UIdesign(const Mat&  srcImg);
Mat GetMask(const Mat&  srcImg);
Mat GettheA(int rows, int cols);
Mat GettheB(const Mat &img, Mat& mod_diff);
void solve_possion(const Mat &backimg, Mat& patch_diff);
Mat Laplace(const Mat &img);
void Getposition(const Mat&  srcImg);
void FastSolvePossion(const Mat &A, Mat& B, Mat& X);
int main()
{
	
	Mat dstImg = imread(IMAGE_PATH_PREFIX + "beach.jpg");
	Mat srcImg  = imread(IMAGE_PATH_PREFIX + "shark.jpg");
	resize(dstImg, dstImg, Size(), 0.6, 0.8);
	//resize(srcImg, srcImg, Size(), 0.5, 0.5);
	//resize(srcImg, srcImg, Size(), 0.6, 0.6);
	resize(dstImg, dstImg, Size(), 0.5, 0.5);
	Mat srcmask = GetMask(srcImg);
	Getposition(dstImg);
	Mat DIF = Laplace(srcmask);
	solve_possion(dstImg, DIF);
	return 0;
}
void Getposition(const Mat&  srcImg){
	int x; int y;
	Mat markmat = UIdesign(srcImg);
	for (int i = 0; i < (int)markmat.rows; i++)
		for (int j = 0; j < (int)markmat.cols; j++)
		{
		if (markmat.at<Vec3b>(i, j)[2] == 255 && markmat.at<Vec3b>(i, j)[1] == 0 && markmat.at<Vec3b>(i, j)[0] == 0)
		{
			x = i;
			y = j;
			break;
		}
		}
	sythesispositionx = x;
	sythesispositiony = y;

}
Mat UIdesign(const Mat&  srcImg)
{

	Mat tempImg;//复制srcimage用于操作的
	Size srcSize = srcImg.size();
	g_mMaskImg = Mat(srcSize.height, srcSize.width, CV_8UC3);
	g_mMaskImg = Scalar::all(0);

	srcImg.copyTo(tempImg);

	namedWindow(MainWindowName);
	//namedWindow(MaskWindowName);

	// set mouse call back function
	setMouseCallback(MainWindowName, on_MouseCallBack, (void*)&tempImg);

	while (1)
	{
		imshow(MainWindowName, tempImg);
		//imshow(MaskWindowName, g_mMaskImg);

		if (waitKey(10) == 27)//每个循环等待10ms，有按ESC键则退出
		{
			break;
		}
	}

	//imwrite(IMAGE_OUTPUT_PATH + "mask1.jpg", tempImg);
	//imwrite(IMAGE_OUTPUT_PATH + "mask2.jpg", g_mMaskImg);
	//return  g_mMaskImg;
	return tempImg;

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

			line(img, g_pPrevious, g_pCurrent, Scalar(0, 0, 255), 4, 4); // draw line on the input image 前景用红色画在原图的副本上和空白纸上
			line(g_mMaskImg, g_pPrevious, g_pCurrent, Scalar(0, 0, 255), 4, 4); // draw line on the mask image

			g_pPrevious = g_pCurrent; // after drawing, current mouse position should be previous mouse position of next movement
		}
		if (g_brightButtonDown)
		{
			g_pCurrent = Point(x, y); // current mouse position
			//rightpLinePoints.push_back(g_pCurrent); // add current mouse position

			line(img, g_pPrevious, g_pCurrent, Scalar(255, 0, 0), 4, 4); // 后景用蓝色画在原图的副本上//b g r
			line(g_mMaskImg, g_pPrevious, g_pCurrent, Scalar(255, 0, 0), 4, 4); // 讲画的标记线画在空白纸上
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
Mat GetMask(const Mat&  srcImg){
	
		Mat markmat = UIdesign(srcImg);
		int minx = markmat.rows, miny = markmat.cols, maxx = 0, maxy = 0;
		for (int i = 0; i < (int)markmat.rows; i++)
			for (int j = 0; j < (int)markmat.cols; j++)
			{
			if (markmat.at<Vec3b>(i, j)[2] == 255 && markmat.at<Vec3b>(i, j)[1] == 0 && markmat.at<Vec3b>(i, j)[0] == 0)
			{
				if (i <minx)minx = i;
				if (i >maxx)maxx = i;
				if (j <miny)miny = j;
				if (j >maxy)maxy = j;
			}
			}
		Mat tp = srcImg(cv::Rect(miny, minx, maxy - miny + 1, maxx - minx + 1));//tp2和sampleimage的部分共享rect 四个参数指的是左上点的列数 行数，总列数，行数
		//imshow("mask3", tp);
		//waitKey(0);
		return tp;
}
void solve_possion(const Mat &backimg, Mat& patch_diff)//传入背景和ROI的laplace图8UC3和32fc3
{
	vector<Mat>channelsV, channelsL;
	split(backimg, channelsV);
	//提取蓝色通道的数据  
	Mat imageBlue = channelsV[0];
	Mat imageGreen = channelsV[1];
	Mat imageRed = channelsV[2];
	split(patch_diff, channelsL);
	//提取蓝色通道的数据  
	Mat imageBlueDIV = channelsL[0];//经验证为float型
	Mat imageGreenDIV = channelsL[1];
	Mat imageRedDIV = channelsL[2];
	Mat  A = GettheA(patch_diff.rows, patch_diff.cols);
	Mat  B1 = GettheB(imageBlue, imageBlueDIV);
	Mat  X1 = Mat(patch_diff.rows* patch_diff.cols, 1, CV_32F);
	Mat  B2 = GettheB(imageGreen, imageGreenDIV);
	Mat  X2 = Mat(patch_diff.rows* patch_diff.cols, 1, CV_32F);
	Mat  B3 = GettheB(imageRed, imageRedDIV);
	Mat  X3 = Mat(patch_diff.rows* patch_diff.cols, 1, CV_32F);
	cout << "solving...." << endl;
	if (workingmode == 1)
	{
		solve(A, B1, X1, CV_SVD);
		cout << "solve 33%...." << endl;
        solve(A, B2, X2, CV_SVD);
		cout << "solve 66%...." << endl;
		solve(A, B3, X3, CV_SVD);
		cout << "solve finish!" << endl;
	}
	if (workingmode == 2)
	{
		FastSolvePossion(A, B1, X1);
		cout << "solve 33%...." << endl;

		FastSolvePossion(A, B2, X2);
		cout << "solve 66%...." << endl;

		FastSolvePossion(A, B3, X3);
		cout << "solve finish!" << endl;
	}
	
	for (int i = 0; i < patch_diff.rows; ++i)
		for (int j = 0; j < patch_diff.cols; ++j)
			imageBlue.at<uchar>(i + sythesispositionx, sythesispositiony + j) = X1.at<float>(i*patch_diff.cols + j, 0);
	for (int i = 0; i < patch_diff.rows; ++i)
		for (int j = 0; j < patch_diff.cols; ++j)
			imageGreen.at<uchar>(i + sythesispositionx, sythesispositiony + j) = X2.at<float>(i*patch_diff.cols + j, 0);
	for (int i = 0; i < patch_diff.rows; ++i)
		for (int j = 0; j < patch_diff.cols; ++j)
			imageRed.at<uchar>(i + sythesispositionx, sythesispositiony + j) = X3.at<float>(i*patch_diff.cols + j, 0);
	Mat mergeImage;
	merge(channelsV, mergeImage);
	imshow("image", mergeImage);
	waitKey(0);
	cout << "If U want to save the result ,please press 'q',the result will save at the output directory." << endl;
	Sleep(5 * 1000);//
	char c = getchar();
	if (c == 113)
	{
		imwrite(IMAGE_OUTPUT_PATH + "result.jpg", mergeImage);
		cout << "Successfully saved! " << endl;
		Sleep(5 * 1000);
	}
	else cout << "Exit without saving the result! " << endl;
	Sleep(5 * 1000);

}
Mat GettheA(int rows, int cols)
{
	Mat A = Mat::zeros(rows* cols, rows* cols, CV_32F);
	//左上角
	{
		A.at<float>(0, 0) = -4;
		A.at<float>(0, 1) = 1;
		A.at<float>(0, cols) = 1;
	}
	//右上角
	{
		A.at<float>(cols - 1, cols - 1) = -4;
		A.at<float>(cols - 1, cols - 2) = 1;
		A.at<float>(cols - 1, cols + cols - 1) = 1;
	}
	//左下角
	{
		A.at<float>(cols* (rows - 1), cols* (rows - 1)) = -4;
		A.at<float>(cols* (rows - 1), cols* (rows - 1) + 1) = 1;
		A.at<float>(cols* (rows - 1), cols* (rows - 1) - cols) = 1;
	}
	//右下角
	{
		A.at<float>(cols *rows - 1, cols *rows - 1) = -4;
		A.at<float>(cols *rows - 1, cols *rows - 2) = 1;
		A.at<float>(cols *rows - 1, cols *rows - cols - 1) = 1;
	}
	//上边界
	for (int i = 0; i < cols - 2; ++i)
	{
		A.at<float>(i + 1, i + 1) = -4;
		A.at<float>(i + 1, i) = 1;
		A.at<float>(i + 1, i + 2) = 1;
		A.at<float>(i + 1, i + 1 + cols) = 1;
	}
	//下边界
	for (int i = 0; i < cols - 2; ++i)
	{
		A.at<float>(i + cols* (rows - 1) + 1, i + cols* (rows - 1) + 1) = -4;
		A.at<float>(i + cols* (rows - 1) + 1, i + cols* (rows - 1)) = 1;
		A.at<float>(i + cols* (rows - 1) + 1, i + cols* (rows - 1) + 2) = 1;
		A.at<float>(i + cols* (rows - 1) + 1, i + cols* (rows - 1) + 1 - cols) = 1;
	}
	//左边界
	for (int i = 0; i < rows - 2; ++i)
	{
		A.at<float>((i + 1)*cols, (i + 1)*cols) = -4;
		A.at<float>((i + 1)*cols, (i + 1)*cols - cols) = 1;
		A.at<float>((i + 1)*cols, (i + 1)*cols + cols) = 1;
		A.at<float>((i + 1)*cols, (i + 1)*cols + 1) = 1;
	}
	//右边界
	for (int i = 0; i < rows - 2; ++i)
	{
		A.at<float>((i + 1)*cols + cols - 1, (i + 1)*cols + cols - 1) = -4;
		A.at<float>((i + 1)*cols + cols - 1, (i + 1)*cols - cols + cols - 1) = 1;
		A.at<float>((i + 1)*cols + cols - 1, (i + 1)*cols + cols + cols - 1) = 1;
		A.at<float>((i + 1)*cols + cols - 1, (i + 1)*cols - 1 + cols - 1) = 1;
	}
	//内点
	int m = cols *rows - 4 - 2 * (rows - 2) - 2 * (cols - 2);
	int n = 0, k = 0;
	for (int i = 0; i < m; ++i)
	{
		A.at<float>(n *cols + k + cols + 1, n *cols + k + cols + 1) = -4;
		A.at<float>(n *cols + k + cols + 1, n *cols + k + cols + 2) = 1;
		A.at<float>(n *cols + k + cols + 1, n *cols + k + cols) = 1;
		A.at<float>(n *cols + k + cols + 1, n *cols + k + cols + 1 - cols) = 1;
		A.at<float>(n *cols + k + cols + 1, n *cols + k + cols + 1 + cols) = 1;
		k++;
		if (k == cols - 2)
		{
			k = 0;
			n++;
		}

	}
	return A;
}
Mat GettheB(const Mat &img, Mat& mod_diff)//传入单通道的背景图片和patch的散度矩阵 8UC1和32FC1 
{

	//int rows = mod_diff.rows;
	//int cols = mod_diff.cols;
	//Mat B = Mat::zeros(rows*cols, 1, CV_32F);
	////左上角
	//{
	//	B.at<float>(0, 0) = mod_diff.at<float>(0, 0) - img.at<uchar>(sythesispositionx - 1, sythesispositiony) - img.at<uchar>(sythesispositionx , sythesispositiony-1);
	// }
	////右上角
	//{
	//	B.at<float>(cols - 1, 0) = mod_diff.at<float>(0, cols - 1) - img.at<uchar>(sythesispositionx-1, sythesispositiony + cols - 1) - img.at<uchar>(sythesispositionx, sythesispositiony + cols );
	//}
	////左下角
	//{
	//	B.at<float>(cols* (rows - 1), 0) = mod_diff.at<float>(rows - 1, 0) - img.at<uchar>(sythesispositionx + rows , sythesispositiony) - img.at<uchar>(sythesispositionx + rows - 1, sythesispositiony-1);
	//}
	////右下角
	//{
	//	B.at<float>(cols *rows - 1, 0) = mod_diff.at<float>(rows - 1, cols - 1) - img.at<uchar>(sythesispositionx + rows, sythesispositiony + cols - 1) - img.at<uchar>(sythesispositionx + rows - 1, sythesispositiony + cols);
	//}
	////上边界
	//for (int i = 0; i < cols - 2; ++i)
	//{
	//	B.at<float>(i + 1, 0) = mod_diff.at<float>(0, i + 1) - img.at<uchar>(sythesispositionx-1, sythesispositiony+ i + 1);
	//}
	////下边界
	//for (int i = 0; i < cols - 2; ++i)
	//{
	//	B.at<float>(i + cols* (rows - 1) + 1, 0) = mod_diff.at<float>(rows - 1, i + 1) - img.at<uchar>(sythesispositionx + rows , sythesispositiony + i + 1);
	//}
	////左边界
	//for (int i = 0; i < rows - 2; ++i)
	//{
	//	B.at<float>((i + 1)*cols, 0) = mod_diff.at<float>(i + 1, 0) - img.at<uchar>(sythesispositionx + i + 1, sythesispositiony-1);
	//}
	////右边界
	//for (int i = 0; i < rows - 2; ++i)
	//{
	//	B.at<float>((i + 1)*cols + cols - 1, 0) = mod_diff.at<float>(i + 1, cols - 1) - img.at<uchar>(sythesispositionx + i + 1, sythesispositiony +cols );
	//}
	////内点
	//vector<float>divvalue;
	//for (int i = 0; i < rows - 2; ++i)
	//	for (int j = 0;j < cols - 2; ++j)
	//		divvalue.push_back(mod_diff.at<float>(i + 1, j + 1));
	////Mat dictp = Mat::zeros(rows,cols, CV_32F);
	////for (int i = 0; i < rows - 2; ++i)
	////	for (int j = 0; j < cols - 2; ++j)
	////dictp.at<float>(i, j) = divvalue[(cols-2)*i+j];
	////imshow("dictp",dictp);
	////waitKey(0);
	//int m = cols *rows - 4 - 2 * (rows - 2) - 2 * (cols - 2);
	//int n = 0, k = 0;
	//for (int i = 0; i < m; ++i)
	//{
	//	B.at<float>(n *cols + k + cols + 1, 0) = divvalue[i];
	//	k++;
	//	if (k == cols - 2)
	//	{
	//		k = 0;
	//		n++;
	//	}
	//}
	////cout << B; 

	int rows = mod_diff.rows;
	int cols = mod_diff.cols;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
		if (i == 0)
			mod_diff.at<float>(i, j) = mod_diff.at<float>(i, j) - img.at<uchar>(sythesispositionx + i - 1, sythesispositiony + j);
		if (i == rows - 1)
			mod_diff.at<float>(i, j) = mod_diff.at<float>(i, j) - img.at<uchar>(sythesispositionx + i + 1, sythesispositiony + j);
		if (j == 0)
			mod_diff.at<float>(i, j) = mod_diff.at<float>(i, j) - img.at<uchar>(sythesispositionx + i, sythesispositiony + j - 1);
		if (j == cols - 1)
			mod_diff.at<float>(i, j) = mod_diff.at<float>(i, j) - img.at<uchar>(sythesispositionx + i, sythesispositiony + j + 1);
		}
	Mat B = Mat::zeros(rows*cols, 1, CV_32F);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			B.at<float>(i*cols + j, 0) = mod_diff.at<float>(i, j);
	//Mat map(rows,cols, CV_32F);;
	//for (int i = 0; i < rows; i++)
	//	for (int j = 0; j < cols; j++)
	//		map.at<float>(i,j) = B.at<float>(i*cols + j, 0);
	//imshow("image3", map);
	//waitKey(0);
	return B;
}
Mat Laplace(const Mat &img)
{
	Mat kernel = Mat::zeros(3, 3, CV_8S);
	Mat result;
	kernel.at<char>(0, 1) = 1;
	kernel.at<char>(1, 1) = -4;
	kernel.at<char>(1, 0) = 1;
	kernel.at<char>(1, 2) = 1;
	kernel.at<char>(2, 1) = 1;
	filter2D(img, result, CV_32F, kernel);
	return result;

}
void FastSolvePossion(const Mat &A, Mat& B, Mat& X)
{
	if (A.rows != A.cols)
		cout << "Something wrong with the A.."<<endl;
	Triplets ATriplet;
	ATriplet.reserve(A.rows, A.cols, 5 * A.rows);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
		{
		if (A.at<float>(i, j) == 0)continue;
		ATriplet.add(i,j,A.at<float>(i, j));
		}
	double *b;
	double *x;
	b = new double[A.rows];
	x = new double[A.rows];
	for (int i = 0; i <A.rows; i++)
	{
		b[i] = (double)B.at<float>(i,0);
	}
	matlabSparseLinearSystem(ATriplet, b, x);
	for (int i = 0; i <A.rows; i++)
	{
		X.at<float>(i,0)= (float)x[i];
	}
}
void matlabSparseLinearSystem(Triplets& A, double b[], double x[]){
	Engine *ep;
	if (!(ep = engOpen("\0")))
	{
		fprintf(stderr, "\nCan't start MATLAB engine\n");
	}
	mxArray *inArray[4];
	inArray[0] = mxCreateDoubleMatrix(A.nnz, 1, mxREAL);//创建a行b列的二维double实数值矩阵（数组）
	inArray[1] = mxCreateDoubleMatrix(A.nnz, 1, mxREAL);
	inArray[2] = mxCreateDoubleMatrix(A.nnz, 1, mxREAL);
	inArray[3] = mxCreateDoubleMatrix(A.m, 1, mxREAL);

	double *dArow = new double[A.nnz];
	double *dAcol = new double[A.nnz];
	double *dVal = new double[A.nnz];
	for (int i = 0; i < A.nnz; i++)
	{
		dArow[i] = double(A.ri[i] + 1);
		dAcol[i] = double(A.ci[i] + 1);
		dVal[i] = double(A.nzval[i]);
	}
	memcpy((void*)mxGetPr(inArray[0]), (void*)dArow, (A.nnz)*sizeof(double));
	memcpy((void*)mxGetPr(inArray[1]), (void*)dAcol, (A.nnz)*sizeof(double));
	memcpy((void*)mxGetPr(inArray[2]), (void*)dVal, (A.nnz)*sizeof(double));
	memcpy((void*)mxGetPr(inArray[3]), (void*)b, (A.m)*sizeof(double));

	mxArray *outArray;
	engEvalString(ep, "clear all;");
	engPutVariable(ep, "ARows", inArray[0]);
	engPutVariable(ep, "ACols", inArray[1]);
	engPutVariable(ep, "AVals", inArray[2]);
	engPutVariable(ep, "b", inArray[3]);

	engEvalString(ep, "A = sparse(ARows, ACols, AVals);");
	engEvalString(ep, "AT = A';");
	engEvalString(ep, "ATA = AT*A;");
	engEvalString(ep, "ATb = AT*b;");
	engEvalString(ep, "X = ATA \\ ATb;");
	outArray = engGetVariable(ep, "X");
	double* X = (double*)mxGetData(outArray);

	for (int i = 0; i<A.n; i++){
		x[i] = X[i];
	}

	for (int i = 0; i < 4; i++)
	{
		mxDestroyArray(inArray[i]);
	}
	mxDestroyArray(outArray);
	delete[] dArow;
	delete[] dAcol;
	delete[] dVal;
}