#include<opencv2/opencv.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/ximgproc/disparity_filter.hpp>
#include <iostream>
#include <string>
#include "Setting.h"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int leftCam = 1;
int rightCam = 2;
string LEFT_PATH = "img/left/";
string RIGHT_PATH = "img/right/";
string OUTPUT_CAB = "data/calibrate";

int showCamera()
{
	VideoCapture web1(leftCam);
	VideoCapture web2(rightCam);

	if (!web1.isOpened() || !web2.isOpened())
		return -1;
	Mat edges;
	namedWindow("left", WINDOW_AUTOSIZE);
	namedWindow("right", WINDOW_AUTOSIZE);

	for (;;)
	{
		Mat left;
		Mat right;
		web1.read(left);
		web2.read(right);
		
		rotate(right, right, ROTATE_180);
		line(left, Point(0,left.rows/2), Point(left.cols, left.rows / 2), Scalar(0,0,255), 2, 8, 0);
		line(right, Point(0, left.rows / 2), Point(left.cols, left.rows / 2), Scalar(0,0,255), 2, 8, 0);
		imshow("left", left);
		imshow("right", right);		
		if (waitKey(33) == '1') {
			imwrite("testL.jpg", left);
			imwrite("testR.jpg", right);
			break;
		}
	}
	return 0;
}


int captureImage()
{

	VideoCapture webLeft(leftCam);
	VideoCapture webRight(rightCam);
	if (!webLeft.isOpened() || !webRight.isOpened())
		return -1;

	int i = 0;
	for (;;)
	{
		Mat left;
		Mat right;
		webLeft >> left;
		webRight >> right;		
		cout << LEFT_PATH.append(i + ".jpg");
		imshow("left", left);
		imshow("right", right);
		imwrite(LEFT_PATH.append("1.jpg"), left);
		imwrite(RIGHT_PATH.append("1.jpg"), right);
		cout << i << "\n";
		i++;
		if (waitKey() == '1') break;
	}
	return 0;
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
	corners.resize(0);
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(float(j*squareSize),
				float(i*squareSize), 0));
}

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

int calibrate()
{
	
	Size boardSize(7, 6);
	vector<string> image({ "capture\\left\\000000.jpg", "capture\\left\\000001.jpg", "capture\\left\\000002.jpg", "capture\\left\\000003.jpg", "capture\\left\\000004.jpg", "capture\\left\\000005.jpg", "capture\\left\\000006.jpg", "capture\\left\\000007.jpg", "capture\\left\\000008.jpg", "capture\\left\\000009.jpg", "capture\\left\\000010.jpg", "capture\\left\\000011.jpg", "capture\\left\\000012.jpg", "capture\\left\\000013.jpg", "capture\\left\\000014.jpg", "capture\\left\\000015.jpg", "capture\\left\\000016.jpg", "capture\\left\\000017.jpg", "capture\\left\\000018.jpg", "capture\\left\\000019.jpg", "capture\\left\\000020.jpg", "capture\\left\\000021.jpg", "capture\\left\\000022.jpg", "capture\\left\\000023.jpg", "capture\\left\\000024.jpg", "capture\\left\\000025.jpg", "capture\\left\\000026.jpg", "capture\\left\\000027.jpg", "capture\\left\\000028.jpg", "capture\\left\\000029.jpg", "capture\\left\\000030.jpg", "capture\\left\\000031.jpg", "capture\\left\\000032.jpg", "capture\\left\\000033.jpg", "capture\\left\\000034.jpg", "capture\\left\\000035.jpg", "capture\\left\\000036.jpg", "capture\\left\\000037.jpg", "capture\\left\\000038.jpg", "capture\\left\\000039.jpg", "capture\\left\\000040.jpg", "capture\\left\\000041.jpg", "capture\\left\\000042.jpg", "capture\\left\\000043.jpg", "capture\\left\\000044.jpg", "capture\\left\\000045.jpg", "capture\\left\\000046.jpg", "capture\\left\\000047.jpg", "capture\\left\\000048.jpg", "capture\\left\\000049.jpg", "capture\\left\\000050.jpg", "capture\\left\\000051.jpg", "capture\\left\\000052.jpg", "capture\\left\\000053.jpg", "capture\\left\\000054.jpg", "capture\\left\\000055.jpg", "capture\\left\\000056.jpg", "capture\\left\\000057.jpg", "capture\\left\\000058.jpg", "capture\\left\\000059.jpg", "capture\\left\\000060.jpg", "capture\\left\\000061.jpg", "capture\\left\\000062.jpg", "capture\\left\\000063.jpg", "capture\\left\\000064.jpg", "capture\\left\\000065.jpg", "capture\\left\\000066.jpg", "capture\\left\\000067.jpg", "capture\\left\\000068.jpg", "capture\\left\\000069.jpg", "capture\\left\\000070.jpg", "capture\\left\\000071.jpg", "capture\\left\\000072.jpg", "capture\\left\\000073.jpg", "capture\\left\\000074.jpg", "capture\\left\\000075.jpg", "capture\\left\\000076.jpg", "capture\\left\\000077.jpg", "capture\\left\\000078.jpg", "capture\\left\\000079.jpg", "capture\\left\\000080.jpg", "capture\\left\\000081.jpg", "capture\\left\\000082.jpg", "capture\\left\\000083.jpg", "capture\\left\\000084.jpg", "capture\\left\\000085.jpg", "capture\\left\\000086.jpg", "capture\\left\\000087.jpg", "capture\\left\\000088.jpg", "capture\\left\\000089.jpg", "capture\\left\\000090.jpg", "capture\\left\\000091.jpg", "capture\\left\\000092.jpg", "capture\\left\\000093.jpg", "capture\\left\\000094.jpg", "capture\\left\\000095.jpg", "capture\\left\\000096.jpg", "capture\\left\\000097.jpg", "capture\\left\\000098.jpg", "capture\\left\\000099.jpg", "capture\\left\\000100.jpg", "capture\\left\\000101.jpg", "capture\\left\\000102.jpg", "capture\\left\\000103.jpg", "capture\\left\\000104.jpg", "capture\\left\\000105.jpg", "capture\\left\\000106.jpg", "capture\\left\\000107.jpg", "capture\\left\\000108.jpg", "capture\\left\\000109.jpg", "capture\\left\\000110.jpg", "capture\\left\\000111.jpg", "capture\\left\\000112.jpg", "capture\\left\\000113.jpg", "capture\\left\\000114.jpg", "capture\\left\\000115.jpg", "capture\\left\\000116.jpg", "capture\\left\\000117.jpg", "capture\\left\\000118.jpg", "capture\\left\\000119.jpg", "capture\\left\\000120.jpg", "capture\\left\\000121.jpg", "capture\\left\\000122.jpg", "capture\\left\\000123.jpg", "capture\\left\\000124.jpg", "capture\\left\\000125.jpg", "capture\\left\\000126.jpg", "capture\\left\\000127.jpg", "capture\\left\\000128.jpg" });
	
	int count = 0;
	double totalAvgErr = 0;
	vector<float> reprojErrs;
	vector<vector<Point2f> > imagePoints;
	for (int i = 0; i < image.size(); i++) 
	{
		vector<Mat> rvecs, tvecs;
		Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
		Mat cameraMatrix = Mat::zeros(8, 1, CV_64F);
		vector<vector<Point3f> > objectPoints(1);
		vector<Point2f> output;
		Mat img = imread(image.at(i));
		//cout << image.at(i) << endl;
		bool found = findChessboardCorners(img, boardSize, output, CALIB_CB_ADAPTIVE_THRESH);

		calcChessboardCorners(boardSize, 18.0, objectPoints[0]);
		objectPoints.resize(output.size(), objectPoints[0]);
		
		double rms = calibrateCamera(objectPoints,output,img.size(),cameraMatrix,distCoeffs,rvecs,tvecs, CALIB_FIX_K4);

		printf("RMS error reported by calibrateCamera: %g\n", rms);
		bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

		totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
			rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

		drawChessboardCorners(img, boardSize, output, found);
		imshow("left", img);
		waitKey(33);
		if (found)
		{
			count++;
		}		
	}
	
	cout << totalAvgErr << endl;
	cout << count << endl;
	getchar();
	return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
	int min_disparity = matcher_instance->getMinDisparity();
	int num_disparities = matcher_instance->getNumDisparities();
	int block_size = matcher_instance->getBlockSize();

	int bs2 = block_size / 2;
	int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

	int xmin = maxD + bs2;
	int xmax = src_sz.width + minD - bs2;
	int ymin = bs2;
	int ymax = src_sz.height - bs2;

	Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
	return r;
}


int dispartyCal()
{
	VideoCapture web1(leftCam);
	VideoCapture web2(rightCam);
	
	Mat lef = imread("testL.jpg",IMREAD_COLOR);
	Mat rih = imread("testR.jpg",IMREAD_COLOR);

	Mat left_disp, right_disp;
	Mat filtered_disp;
	Rect ROI;
	Mat left_for_matcher, right_for_matcher;

	int max_disp = 160;
	int wsize = 3;
	double lambda = 8000;
	double sigma = 1.5;
	double vis_mult = 1.0;


	Ptr<DisparityWLSFilter> wls_filter ;

	for (;;) {

		web1.read(lef);
		web2.read(rih);
		rotate(rih, rih, ROTATE_180);

		//clone for next use
		left_for_matcher = lef.clone();
		right_for_matcher = rih.clone();

		//set compute method
		Ptr<StereoSGBM> left_matcher = StereoSGBM::create(0, max_disp, wsize);
		left_matcher->setP1(24 * wsize*wsize);
		left_matcher->setP2(96 * wsize*wsize);
		//left_matcher->setP1(8 * wsize*wsize);
		//left_matcher->setP2(32 * wsize*wsize);
		left_matcher->setPreFilterCap(63);
		left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);

		ROI = computeROI(left_for_matcher.size(), left_matcher);
		wls_filter = createDisparityWLSFilterGeneric(false);
		wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));


		Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);


		cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
		cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

		left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
		right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

		//smoothering
		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		wls_filter->filter(left_disp, lef, filtered_disp, Mat(), ROI);


		namedWindow("left", WINDOW_AUTOSIZE);
		namedWindow("right", WINDOW_AUTOSIZE);


		imshow("left", lef);
		imshow("right", rih);

		Mat raw_disp_vis;
		getDisparityVis(left_disp, raw_disp_vis, vis_mult);
		namedWindow("raw disparity", WINDOW_AUTOSIZE);
		imshow("raw disparity", raw_disp_vis);

		Mat filtered_disp_vis;
		getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
		namedWindow("filtered disparity", WINDOW_AUTOSIZE);
		imshow("filtered disparity", filtered_disp_vis);

		short pixVal = raw_disp_vis.at<short>(300, 300);
		cout << (float)pixVal / 16;
		if (waitKey(33) == '1') break;
 	}
	


	return 0;
}






int main()
{
	
	
	return calibrate();


	
	return 0;
	//return dispartyCal();
	//return captureImage();
	//return showCamera();
}
