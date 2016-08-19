#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/optflow.hpp"
#include "sparse_matching_gpc.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>

/* This tool finds correspondences between two images using Global Patch Collider
 * and calculates error using provided ground truth flow.
 *
 * It will look for the file named "forest.dump" with a learned forest.
 * You can obtain the "forest.dump" either by manually training it using another tool with *_train suffix
 * or by downloading one of the files trained on some publicly available dataset from here:
 *
 * https://drive.google.com/open?id=0B7Hb8cfuzrIIZDFscXVYd0NBNFU
 */

using namespace cv;

const String keys = "{help h ?     |           | print this message}"
"{@image1      |<none>     | image1}"
"{@image2      |<none>     | image2}"
"{@groundtruth |<none>     | path to the .flo file}"
"{@output      |           | output to a file instead of displaying, output image path}"
"{f forest     |forest.dump| path to the forest.dump}";

const int nTrees = 5;
const bool useOpenCL = false;

static double normL2(const Point2f &v) { return sqrt(v.x * v.x + v.y * v.y); }

static Vec3d getFlowColor(const Point2f &f, const bool logScale = true, const double scaleDown = 5)
{
	if (f.x == 0 && f.y == 0)
		return Vec3d(0, 0, 1);

	double radius = normL2(f);
	if (logScale)
		radius = log(radius + 1);
	radius /= scaleDown;
	radius = std::min(1.0, radius);

	double angle = (atan2(-f.y, -f.x) + CV_PI) * 180 / CV_PI;
	return Vec3d(angle, radius, 1);
}

static void displayFlow(InputArray _flow, OutputArray _img)
{
	const Size sz = _flow.size();
	Mat flow = _flow.getMat();
	_img.create(sz, CV_32FC3);
	Mat img = _img.getMat();

	for (int i = 0; i < sz.height; ++i)
		for (int j = 0; j < sz.width; ++j)
			img.at< Vec3f >(i, j) = getFlowColor(flow.at< Point2f >(i, j));

	cvtColor(img, img, COLOR_HSV2BGR);
}

static bool fileProbe(const char *name) { return std::ifstream(name).good(); }

int main(int argc, const char **argv)
{
	String fromPath = String(argv[2]);
	String toPath = String(argv[3]);
	String forestDumpPath = String(argv[1]);
	String outShowPath;
	if (argc > 4)
		outShowPath = String(argv[4]);


	ocl::setUseOpenCL(useOpenCL);

	std::cout << "load GPC model:" << forestDumpPath << "......";
	Ptr< optflow::GPCForest< nTrees > > forest = Algorithm::load< optflow::GPCForest< nTrees > >(forestDumpPath);
	std::cout << "done" << std::endl;
	Mat from = imread(fromPath);
	Mat to = imread(toPath);

	std::vector< std::pair< Point2i, Point2i > > corr;
	forest->findCorrespondences(from, to, corr, optflow::GPCMatchingParams(useOpenCL));
	std::cout << "Found " << corr.size() << " matches." << std::endl;

	Mat disp = Mat::zeros(from.size(), CV_32FC3);
	disp = Scalar(0, 0, 1);

//	for (size_t i = 0; i < corr.size(); ++i)
//	{
//		const Point2f a = corr[i].first;
//		const Point2f b = corr[i].second;
//		circle(disp, a, 3, getFlowColor(b - a), -1);
//	}

	int ind = 0;
	
	cv::Mat show0 = cv::Mat::zeros(from.rows, to.cols * 2, CV_8UC3);
	cv::Rect rect(0, 0, from.cols, from.rows);
	from.copyTo(show0(rect));
	rect.x = rect.x + from.cols;
	to.copyTo(show0(rect));

	RNG rng(12345);
	srand(time(NULL));
	int step = corr.size() / 100 * 2;
	int goout = 1;
	Mat show;

	namedWindow("Correspondences", WINDOW_AUTOSIZE);
	std::cout << "Only about 100 points are shown, press N to show others, press other keys to exit!" << std::endl;
	while (goout == 1) {
		show = show0.clone();
		while (ind < corr.size()) {
			Point2f a = corr[ind].first;
			Point2f b = corr[ind].second;
			b.x += from.cols;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::line(show, a, b, color, 1, LINE_8, 0);
			ind = ind + (int)rand() % step;
		}
		imshow("Correspondences", show);
		int c = waitKey();
		switch (c) {
			case 'n':
			case 'N': goout = 1; break;
			default: goout = 0;
		}
		ind = 0;
	}
	if (argc > 4)
		imwrite(outShowPath, show);
//	cvtColor(disp, disp, COLOR_HSV2BGR);
	return 0;
}
