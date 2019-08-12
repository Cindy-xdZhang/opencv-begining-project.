
//#include <opencv2/opencv.hpp>
//#include<opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<opencv2\imgproc\imgproc.hpp>
//
//#include<iostream>
////#include <stdlib.h> 
////#include <time.h>
////#include <math.h>
////#include<opencv2/legacy/legacy.hpp>
////#include<opencv2/nonfree/nonfree.hpp>
////#include<opencv2/features2d/features2d.hpp>
//
////	b = Mat::ones(3, 3, CV_8UC3);
////a.at<Vec3b>(0, 2) = { 14, 5, 6 };
//
//using namespace cv;
//using namespace  std;
//double CalDistance(Mat& a, Mat& b);
////int main()
////{
////	Mat a, b;
////
////
////	CalDistance(a, b);
////	return 0;
////}
//double CalDistance(Mat& a, Mat& b)
//{
//	if (a.size() != b.size())
//	{cout << "caldistance error!" << endl;
//	getchar(); getchar();
//	exit(1);
//   }
//	Mat tmp;
//	absdiff(a, b, tmp);
//return 1e4 - sum(sum(tmp))[0];
//}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <ctime>
namespace util {
	using namespace std;
	using namespace cv;
	float random_range(float min, float max) {

		if (min > max) { int t = min; min = max; max = t; }
		float f = (rand() % 65537) * 1.0f / 65537.0f;
		return f * (max - min) + min;

	}
	double sim_abs_diff(Mat& a, Mat& b, int _) {//计算绝对距离
		Mat tmp;
		absdiff(a, b, tmp);
		return 1e4 - sum(sum(tmp))[0];
		/*tmp = a - b;
		tmp=tmp.mul(tmp);
		tmp = 0.1*tmp;
		double c = sum(tmp)[0] + sum(tmp)[1] + sum(tmp)[2];
		c= sqrt(c);
		return (1e4 - c);*/

	}
	int argmax(float v[], int n) {
		int max_i = 0;
		for (int i = 1; i < n; i++)
			if (v[i] > v[max_i]) max_i = i;
		return max_i;

	}

	int argmin(float v[], int n) {
		int min_i = 0;
		for (int i = 1; i < n; i++)
			if (v[i] < v[min_i])min_i = i;
		return min_i;
	}

	int max(int a, int b) {
		return a > b ? a : b;
	}

	int min(int a, int b) { return a < b ? a : b; }

}
using namespace cv;
using namespace std;
double(*sim)(Mat&, Mat&, int) = util::sim_abs_diff;  //metric::ssim;
typedef vector < vector < vector<int> > >	Vector3i;
typedef vector < vector <int> >				Vector2i;
typedef struct Unfilled_point
{
	cv::Point2i position;
	double filedNeighbors;
}unfilledpoint;
typedef struct Matches
{
	cv::Point2i position;
	double err;
}matches;
void reconstruct(Vector3i& f, Mat& result, Mat& ref, int patch_size);
Mat pick_patch(Mat& mat, int r, int c, int r_offset, int c_offset, int patch_size) {//向右下取patch
	int rr = r + r_offset, rc = c + c_offset;
	return mat(Range(rr, rr + patch_size), Range(rc, rc + patch_size));
}
// 初始化 offset
void initialize(Vector3i& f, int n_rows, int n_cols, int patch_size) {
	cout << "initializing..." << endl;
	f.resize(n_rows);
	for (int i = 0; i < n_rows; i++) {
		f[i].resize(n_cols);
		for (int j = 0; j < n_cols; j++)
			f[i][j].resize(2);
	}
	for (int i = 0; i < n_rows; i++) {
		for (int j = 0; j < n_cols; j++) {
			f[i][j][0] = int(util::random_range(0, n_rows - patch_size)) - i;
			f[i][j][1] = int(util::random_range(0, n_cols - patch_size)) - j;
			if (i == 0 || j == 0 || i == n_rows - 1 || j == n_cols - 1)
				f[i][j][0] = f[i][j][1] = 0;
		}
	}
}
void patchmatch(Vector3i& f, Mat& img_dst, Mat& img_ref, int patch_size = 3, int n_iterations = 5) {
	const int n_rows = img_dst.rows, n_cols = img_dst.cols;
	/*  初始化 offset */
	initialize(f, n_rows, n_cols, patch_size);
	/* 迭代 */
	int row_start, row_end, col_start, col_end, step;
	Vector2i v;  // current similarity compared with current patch offset
	v.resize(n_rows); for (int i = 0; i < n_rows; i++) { v[i].resize(n_cols); }
	for (int i = 0; i < n_rows - patch_size; i++)
		for (int j = 0; j < n_cols - patch_size; j++)
			v[i][j] = sim(pick_patch(img_dst, i, j, 0, 0, patch_size),	pick_patch(img_ref, i, j, f[i][j][0], f[i][j][1], patch_size), 1);//vij代表像素图ij+offset与目标图ij间的patch距离
	bool reverse = false;//控制奇数偶数次扫描方向
	auto checkvalid = [patch_size, n_rows, n_cols](int r, int c, int ro, int co)
	{

		if (r + ro < 0) return false;
		if (c + co < 0) return false;
		if (r + ro + patch_size >= n_rows) return false;
		if (c + co + patch_size >= n_cols) return false;
		return true;
	};//函数内定义函数类
	for (int t = 0; t < n_iterations; t++) {//迭代（五次）
		Mat progress(img_dst.rows, img_dst.cols, img_dst.type());
		reconstruct(f, progress, img_ref, patch_size);
		//imshow("progress", progress);
		//cv::waitKey(0);


		/* propagate */
		cout << "iteration " << t + 1<<".." << endl;
		if (reverse) {
			row_start = n_rows - patch_size - 2;
			row_end = -1;
			col_start = n_cols - patch_size - 2;
			col_end = -1;
			step = -1;
		}
		else {
			row_start = 1;
			row_end = n_rows - patch_size;
			col_start = 1;
			col_end = n_cols - patch_size;
			step = 1;
		}
		for (int i = row_start; i != row_end; i += step) {
			for (int j = col_start; j != col_end; j += step) {
				float sm[3];
				Mat patch = pick_patch(img_dst, i, j, 0, 0, patch_size);
				// Mat ipatch = pick_patch(img_ref, i, j, f[i][j][0], f[i][j][1], patch_size);		// sim(patch, ipatch) == v[i][j]
				sm[0] = v[i][j];
				if (checkvalid(i, j, f[i - step][j][0], f[i - step][j][1])) {
					Mat xpatch = pick_patch(img_ref, i, j, f[i - step][j][0], f[i - step][j][1], patch_size);
					sm[1] = sim(patch, xpatch, 1);
				}
				else sm[1] = -1e6f;
				if (checkvalid(i, j, f[i][j - step][0], f[i][j - step][1])) {
					Mat ypatch = pick_patch(img_ref, i, j, f[i][j - step][0], f[i][j - step][1], patch_size);
					sm[2] = sim(patch, ypatch, 1);
				}   
				else sm[2] = -1e6f;
				int k = util::argmax(sm, 3);
				v[i][j] = sm[k];
				switch (k) {
				case 0: break;
				case 1: f[i][j][0] = f[i - step][j][0]; f[i][j][1] = f[i - step][j][1]; break;
				case 2: f[i][j][0] = f[i][j - step][0]; f[i][j][1] = f[i][j - step][1]; break;
				default: break; // error
				}
			}
		}
		reverse = !reverse;
		///* 随机搜索 */
		for (int i = row_start; i != row_end; i += step){
			for (int j = col_start; j != col_end; j += step) {
				int r_ws = n_rows, c_ws = n_cols;
				float alpha = 0.8f;
				while (r_ws*alpha > 1 && c_ws*alpha > 1) {
					int rmin = util::max(0, int(i + f[i][j][0] - r_ws*alpha));
					int rmax = util::min(int(i + f[i][j][0] + r_ws*alpha), n_rows - patch_size);
					int cmin = util::max(0, int(j + f[i][j][1] - c_ws*alpha));
					int cmax = util::min(int(j + f[i][j][1] + c_ws*alpha), n_cols - patch_size);

					if (rmin > rmax) rmin = rmax = f[i][j][0] + i;
					if (cmin > cmax) cmin = cmax = f[i][j][1] + j;

					int r_offset = int(util::random_range(rmin, rmax)) - i;
					int c_offset = int(util::random_range(cmin, cmax)) - j;

					Mat  patch = pick_patch(img_dst, i, j, 0, 0, patch_size);
					Mat cand= pick_patch(img_ref, i, j, r_offset, c_offset, patch_size);

					float similarity = sim(patch, cand, 1);
					if (similarity > v[i][j]) {
						v[i][j] = similarity;
						f[i][j][0] = r_offset; f[i][j][1] = c_offset;
					}
					alpha *= 0.8f;
				}
			}
		}
	}//结束一次迭代
}
void reconstruct(Vector3i& f, Mat& result, Mat& ref, int patch_size) {
/* 复制-粘贴，需要考虑到一个非边界的patch只有左上角点不会被后续patch覆盖 ，所以对于非边界的像素点，直接取match中的像素点即可 */
	int n_rows = result.rows, n_cols = result.cols;
	for (int i = 0; i < n_rows; i++) {
		for (int j = 0; j < n_cols; j++) {
			result.at< Vec3b>(i, j) = ref.at< Vec3b>(i + f[i][j][0], j + f[i][j][1]);
		}
	}
	///* 处理边界情况 */
	int r = n_rows - patch_size - 1;
	int c;
	for (c = 0; c < n_cols - patch_size - 1; c++) {
		Mat last_patch = pick_patch(ref, r, c, f[r][c][0], f[r][c][1], patch_size);
		for (int i = 0; i < patch_size; i++) {
			for (int j = 0; j < patch_size; j++) {
				result.at< Vec3b>(r + i, c + j) = last_patch.at< Vec3b>(i, j);
			}
		}
	}
	c = n_cols - patch_size - 1;
   for(r = 0; r < n_rows - patch_size - 1; r++) {
		Mat last_patch = pick_patch(ref, r, c, f[r][c][0], f[r][c][1], patch_size);
		for (int i = 0; i < patch_size; i++) {
			for (int j = 0; j < patch_size; j++) {
				result.at< Vec3b>(r + i, c + j) = last_patch.at< Vec3b>(i, j);
			}
		}
	}
}
void testImage(const char *img_dst_path, const char *img_ref_path) {
	srand((unsigned int)time(NULL));
	Mat dst = imread(img_dst_path);
	Mat ref = imread(img_ref_path);
	assert(dst.cols == ref.cols && dst.rows == ref.rows);
	assert(dst.type() == CV_8UC3);
	imshow("src.jpg", dst);
	imshow(" ref.jpg", ref);
	Vector3i offset;
	patchmatch(offset, dst, ref, 5, 5);
	reconstruct(offset, dst, ref, 5);
	imshow("result", dst);
	imwrite("result.png", dst);
	cvWaitKey(0);
}
int main() {

	char *dstpath = "dst.jpg";
	char *refpath = "src.jpg";
	testImage(dstpath, refpath);
	return 0;

}

