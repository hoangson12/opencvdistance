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
	float scale = 1;
	VideoCapture webLeft(leftCam);
	VideoCapture webRight(rightCam);
	if (!webLeft.isOpened() || !webRight.isOpened())
	{
		cout << "wrong camera setting";
		getchar();
		return -1;
	}
		
	webLeft.set(CV_CAP_PROP_FRAME_HEIGHT, 960 / scale);
	webLeft.set(CV_CAP_PROP_FRAME_WIDTH, 1280 / scale);
	webLeft.set(CV_CAP_PROP_FPS, 30);
	webRight.set(CV_CAP_PROP_FRAME_HEIGHT, 960 / scale);
	webRight.set(CV_CAP_PROP_FRAME_WIDTH, 1280 / scale);
	webRight.set(CV_CAP_PROP_FPS, 30);
	int i = 0;
	for (;;)
	{
		Mat left;
		Mat right;
		webLeft >> left;
		webRight >> right;
		flip(right, right, -1);
		imshow("left", left);
		imshow("right", right);		
		char c = (char)waitKey(33);
		if (c == ' ')
		{

			string s = to_string(i);
			cout << LEFT_PATH + s + ".jpg";
			imwrite(LEFT_PATH+s + ".jpg", left);
			imwrite(RIGHT_PATH+ s + ".jpg", right);
			cout << i << "\n";
			i++;
		}
		if (c == '1') break;
	}
	return 0;
}

static int StereoCalib(const vector<string> &imageList, Size broadSize, float squareSize, bool displayCorners = false, 
								bool useCalibrated = true)
{
	//copy from https://github.com/opencv/opencv/blob/3.4.1/samples/cpp/stereo_calib.cpp check this shit out

	if (imageList.size() % 2 != 0)
	{
		cout << "need to be even" << endl;
		return 0;
	}

	const int maxScale = 2;

	vector<vector<Point2f>> imagePoints[2];
	vector<vector<Point3f>> objectPoints;
	Size imageSize;
	
	int i, j, k, nimages = (int)imageList.size() / 2;
	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);

	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string &fileName = imageList[i*2+k];
			Mat img = imread(fileName,0);
			if (img.empty()) break;
			if (imageSize == Size()) imageSize = img.size();
			else if (imageSize!=img.size())
			{
				cout << "fucking wrong size, skipping " << fileName << endl;
				break;
			}

			bool found = true;
			vector<Point2f> &corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1) timg = img;
				else resize(img,timg,Size(),scale,scale,INTER_LINEAR_EXACT);
				//found = findChessboardCorners(timg, broadSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				found = findChessboardCorners(timg, broadSize, corners, CALIB_CB_ADAPTIVE_THRESH);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}

			}
			if (displayCorners)
			{
				cout << fileName << endl;
				Mat cimg, cimg1;
				cvtColor(img,cimg,COLOR_GRAY2BGR);
				drawChessboardCorners(cimg,broadSize,corners,found);
				double sf = 640. / MAX(img.rows, img.cols);
				resize(cimg,cimg1,Size(),sf,sf,INTER_LINEAR_EXACT);
				imshow("corner",cimg1);
				char c = (char)waitKey(33);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found) break;

			cornerSubPix(img,corners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imageList[i * 2]);
			goodImageList.push_back(imageList[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pair successfully detected" << endl;
	nimages = j;
	if (nimages < 2)
	{
		cout << "2 fucking little pair to run" << endl;
		return 0;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < broadSize.height; j++) 
			for (k = 0; k < broadSize.width; k++)
			{
				objectPoints[i].push_back(Point3f(k*squareSize,j*squareSize,0));
			}
	}
	cout << "Running image calib .." << endl;	
	Mat cameraMatrix[2], distCoeffs[2];

	cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

	Mat R, T, E, F;
	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_USE_INTRINSIC_GUESS +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;

	//recheck RMS value, should remove, idc
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];

	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);

			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err " << err / npoints << endl;

	//save to file to remap later
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
	{
		cout << "sum ting wrong" << endl;
	}

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinics.yml",FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else cout << "Error: can not save the extrinsic parameters\n";
	fs.open("validRoi.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "V1" << validRoi[0] << "V2" << validRoi[1];
		fs.release();
	}
	else cout << "SML";

	
	return 0;
}


static int remapInput(string intrinsics, string extrinics, Mat input1, Mat input2,Size imageSize, bool showRectified = true)
{
	Mat R1, R2, P1, P2, Q;
	Mat cameraMatrix[2], distCoeffs[2];
	Rect validRoi[2];
	FileStorage fs;
	fs.open("extrinics.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["R1"] >> R1;
		fs["R2"] >> R2;
		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;
		fs.release();
	}
	else
	{
		cout << "NO calibrate file, rerun the fukcing calib" << endl;
		return -1;
	}

	fs.open("intrinsics.yml", FileStorage::READ);

	if (fs.isOpened())
	{
		fs["M1"] >> cameraMatrix[0];
		fs["M2"] >> cameraMatrix[1];
		fs["D1"] >> distCoeffs[0];
		fs["D2"] >> distCoeffs[1];
		fs.release();
	}
	else
	{
		cout << "NO calibrate file, rerun the fukcing calib" << endl;
		return -1;
	}
	fs.open("validRoi.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["V1"] >> validRoi[0];
		fs["V2"] >> validRoi[1];
		fs.release();
	}
	else
	{
		cout << "NO calibrate file, rerun the fukcing calib" << endl;
		return -1;
	}


	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
	
	if (!showRectified) return 0;
	Mat rmap[2][2];
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	// number  300  for horizontal stereo
	sf = 300. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h * 2, w, CV_8UC3);
	//testing purpose only lul
	Mat img = imread("img/left/1.jpg");
	Mat rimg, cimg;
	remap(img,rimg,rmap[0][0],rmap[0][1],INTER_LINEAR);
	cvtColor(rimg, cimg, COLOR_GRAY2BGR);
	Mat canvasPart = canvas(Rect(w*0, 0, w, h));
	resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);

	Rect vroi(cvRound(validRoi[0].x*sf),cvRound(validRoi[0].y*sf),cvRound(validRoi[0].width*sf),cvRound(validRoi[0].height*sf));
	rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
	for (int j = 0; j < canvas.cols; j += 16)
	{
		line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
	}

	imshow("rectified", canvas);
	char c = (char)waitKey();
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
	Size broadSize(7, 6);
	//vector<string> imageLeft ({ "capture\\left\\000000.jpg", "capture\\left\\000001.jpg", "capture\\left\\000002.jpg", "capture\\left\\000003.jpg", "capture\\left\\000004.jpg", "capture\\left\\000005.jpg", "capture\\left\\000006.jpg", "capture\\left\\000007.jpg", "capture\\left\\000008.jpg", "capture\\left\\000009.jpg", "capture\\left\\000010.jpg", "capture\\left\\000011.jpg", "capture\\left\\000012.jpg", "capture\\left\\000013.jpg", "capture\\left\\000014.jpg", "capture\\left\\000015.jpg", "capture\\left\\000016.jpg", "capture\\left\\000017.jpg", "capture\\left\\000018.jpg", "capture\\left\\000019.jpg", "capture\\left\\000020.jpg", "capture\\left\\000021.jpg", "capture\\left\\000022.jpg", "capture\\left\\000023.jpg", "capture\\left\\000024.jpg", "capture\\left\\000025.jpg", "capture\\left\\000026.jpg", "capture\\left\\000027.jpg", "capture\\left\\000028.jpg", "capture\\left\\000029.jpg", "capture\\left\\000030.jpg", "capture\\left\\000031.jpg", "capture\\left\\000032.jpg", "capture\\left\\000033.jpg", "capture\\left\\000034.jpg", "capture\\left\\000035.jpg", "capture\\left\\000036.jpg", "capture\\left\\000037.jpg", "capture\\left\\000038.jpg", "capture\\left\\000039.jpg", "capture\\left\\000040.jpg", "capture\\left\\000041.jpg", "capture\\left\\000042.jpg", "capture\\left\\000043.jpg", "capture\\left\\000044.jpg", "capture\\left\\000045.jpg", "capture\\left\\000046.jpg", "capture\\left\\000047.jpg", "capture\\left\\000048.jpg", "capture\\left\\000049.jpg", "capture\\left\\000050.jpg", "capture\\left\\000051.jpg", "capture\\left\\000052.jpg", "capture\\left\\000053.jpg", "capture\\left\\000054.jpg", "capture\\left\\000055.jpg", "capture\\left\\000056.jpg", "capture\\left\\000057.jpg", "capture\\left\\000058.jpg", "capture\\left\\000059.jpg", "capture\\left\\000060.jpg", "capture\\left\\000061.jpg", "capture\\left\\000062.jpg", "capture\\left\\000063.jpg", "capture\\left\\000064.jpg", "capture\\left\\000065.jpg", "capture\\left\\000066.jpg", "capture\\left\\000067.jpg", "capture\\left\\000068.jpg", "capture\\left\\000069.jpg", "capture\\left\\000070.jpg", "capture\\left\\000071.jpg", "capture\\left\\000072.jpg", "capture\\left\\000073.jpg", "capture\\left\\000074.jpg", "capture\\left\\000075.jpg", "capture\\left\\000076.jpg", "capture\\left\\000077.jpg", "capture\\left\\000078.jpg", "capture\\left\\000079.jpg", "capture\\left\\000080.jpg", "capture\\left\\000081.jpg", "capture\\left\\000082.jpg", "capture\\left\\000083.jpg", "capture\\left\\000084.jpg", "capture\\left\\000085.jpg", "capture\\left\\000086.jpg", "capture\\left\\000087.jpg", "capture\\left\\000088.jpg", "capture\\left\\000089.jpg", "capture\\left\\000090.jpg", "capture\\left\\000091.jpg", "capture\\left\\000092.jpg", "capture\\left\\000093.jpg", "capture\\left\\000094.jpg", "capture\\left\\000095.jpg", "capture\\left\\000096.jpg", "capture\\left\\000097.jpg", "capture\\left\\000098.jpg", "capture\\left\\000099.jpg", "capture\\left\\000100.jpg", "capture\\left\\000101.jpg", "capture\\left\\000102.jpg", "capture\\left\\000103.jpg", "capture\\left\\000104.jpg", "capture\\left\\000105.jpg", "capture\\left\\000106.jpg", "capture\\left\\000107.jpg", "capture\\left\\000108.jpg", "capture\\left\\000109.jpg", "capture\\left\\000110.jpg", "capture\\left\\000111.jpg", "capture\\left\\000112.jpg", "capture\\left\\000113.jpg", "capture\\left\\000114.jpg", "capture\\left\\000115.jpg", "capture\\left\\000116.jpg", "capture\\left\\000117.jpg", "capture\\left\\000118.jpg", "capture\\left\\000119.jpg", "capture\\left\\000120.jpg", "capture\\left\\000121.jpg", "capture\\left\\000122.jpg", "capture\\left\\000123.jpg", "capture\\left\\000124.jpg", "capture\\left\\000125.jpg", "capture\\left\\000126.jpg", "capture\\left\\000127.jpg", "capture\\left\\000128.jpg" });
	vector<string> imageLeft ({ "img\\left\\0.jpg","img\\left\\1.jpg","img\\left\\10.jpg","img\\left\\11.jpg","img\\left\\12.jpg","img\\left\\13.jpg","img\\left\\14.jpg","img\\left\\15.jpg","img\\left\\16.jpg","img\\left\\17.jpg","img\\left\\18.jpg","img\\left\\19.jpg","img\\left\\2.jpg","img\\left\\3.jpg","img\\left\\4.jpg","img\\left\\5.jpg","img\\left\\6.jpg","img\\left\\7.jpg","img\\left\\8.jpg","img\\left\\9.jpg" });

	vector<string> imageListAll;

	for (int i = 0; i < imageLeft.size(); i++)
	{
		imageListAll.push_back(imageLeft[i]);
		imageListAll.push_back(imageLeft[i].replace(4,4,"right"));
		
	}
	for (int i=0;i<imageListAll.size();i++) cout << imageListAll[i] << endl;

	StereoCalib(imageListAll, broadSize, 18, true, true);
	//getchar();
	return 0;
	
	//return calibrate();
	//return dispartyCal();
	//return captureImage();
	//return showCamera();

}
