/*
 * @file stereo_match.cpp
 *
 * Copyright (c) 2024 VoXel Interaction Design
 *
 * VoXel.at <office@voxel.at>
 * @author Simon Vogl
 *
 * @brief Computes a stereo match algorithm from a stereo calibrated and
 * rectified image pair. It creates a 3D point cloud from the dispatity image
 * generated.
 *
 *
 * Usage: ./stereo_pflasterl -cf config_file.xml -l left.jpg -r right.jpg
 */

#include <stdio.h>
#include <stdlib.h> // getenv

#include <iostream>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#if PCL_FOUND
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#if PCL_VIZ
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#endif
#endif

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

using namespace cv;
using namespace boost;
namespace fs = boost::filesystem;

using namespace std;

string intrinsic_filename,
    extrinsic_filename,
    args_filename,
    capture_folder,
    dst_folder,
    calib_folder;

float camera_z_offset = 0.0, scale = 1.f;

enum {
    STEREO_BM = 0,
    STEREO_SGBM = 1,
    STEREO_HH = 2,
    STEREO_VAR = 3
};
int alg = STEREO_SGBM,
    rgb = 0,
    minDisparity, numberOfDisparities,
    SADWindowSize, preFilterCap, uniquenessRatio,
    speckleWindowSize, speckleRange, disp12MaxDiff;

bool show_disparity = false,
     show_rectified = false,
     show_cloud = false;

float max_z, min_z,
    max_x, min_x,
    max_y, min_y;

// extrinsic and intrinsic values
Mat Q,
    M1, D1, M2, D2,
    R, T, R1, P1, R2, P2;

/** input manipulation */
bool rot180 = true;
bool flipChannels = false;
double tooDarkPct = 0.8;
double tooBrightPct = 0.8;

/*
 * intrinsics manipulation
 */
Mat mulLeft(Size(3, 3), CV_64FC1);
Mat mulRight(Size(3, 3), CV_64FC1);
static double mulInc = .002;
Mat posLeft(Size(3, 3), CV_64FC1);
Mat posRight(Size(3, 3), CV_64FC1);
static double posInc = 1.;
bool chgCenterLeft = false;
int mulDisp = 1;

string title; ///< prefix for imshow

/*
 * extrinsics manipulation
 */
Mat rotVec(Size(1, 3), CV_64FC1);

/** one pixel rotation in rad */
static double rotAngle = 60.f / 2952.f / 180.f * M_PI * 10;

// debugging - xz-plot
Mat xzPlot(800, 800, CV_8UC3);
bool laplaceOverlay = false;
bool saveRectified = false;

static void usage()
{
    printf("\n Stereo matching converting L and R images into disparity and point clouds\n");
    printf("\n Usage: stereoPflasterl -cf <config.yaml> -l <left.jpg> -r <right.jpg> \n");
}

/**
 * Parses the configuration file and initialices the cofig variables.
 *
 **/
static int parseConfig(string param_filename);

/**
 * Rectify the images using the calibration parameter and creates the disparity images.
 *
 */
static int rectifyAndReproject(const Mat& img1, const Mat img2, Mat& disp, Mat& rectified);

/** print 3d point information from disparity map
 *
 */
static void dispTo3D(Mat Q, Mat img1, Mat disp, int x, int y);

#if !PCL_FOUND
namespace pcl {
class PointXYZ {
public:
    double x, y, z;
};
class PointXYZRGB {
public:
    double x, y, z;
    float rgb;
};
};
#endif
static pcl::PointXYZRGB dispToPtRGB(const Mat& Q, const Mat& img1, const Mat& disp, int x, int y);

/** check if the majority of points are between low and high thresholds.
 * Uses a scaled-down (1:8) version of img
 * @return true if ok, false if outliers
 */
static bool checkImgHistogram(const Mat& img, int thresholdLow, int thresholdHigh);

/*
 * Creates the pointClouds from disparity image
 *
 */
static void createCloud(Mat Q, Mat img1, Mat disp, string cloud_filename);

/*
 * Deals with system enviroment variables.
 *
 */
static std::string expand_environment_variables(std::string s);

/** plots an x/z-stripe to a pre-allocated Mat; 1px == 1cm
 *
 */
static void plotHStripe(const Mat& disp, const Mat& img, Mat& plotTo, Scalar color, int y);

static void plotVStripe(const Mat& disp, const Mat& img, Mat& plotTo, Scalar color, int x);

static void dumpParams();

// Time count tests,
const clock_t begin_time = clock();

int main(int argc, char** argv)
{
    //// - Parameters parser section
    if (argc < 6) {
        usage();
        cout << "[Error] Too few arguments" << endl;
        exit(1);
    }
    
    // intrinsics
    mulLeft = Mat::eye(3, 3, CV_64FC1);
    mulLeft.at<double>(0, 2) = 1.;
    mulLeft.at<double>(1, 2) = 1.;

    mulRight = Mat::eye(3, 3, CV_64FC1);
    mulRight.at<double>(0, 2) = 1.;
    mulRight.at<double>(1, 2) = 1.;

    posLeft = 0;
    posRight = 0;

    // extrinsics:
    rotVec = 0.; // set to no rotation by default

    string left, right;

    for (int i = 1; i < argc; i++) {
        string arg = string(argv[i]);
        // --- Get and parse configuration file.
        if (arg == "-cf") {
            if (parseConfig(argv[++i]) == -1) {
                cout << "[Error] Could not read xml config file" << argv[i] << endl;
                exit(106);
            }
        } else if (arg == "-l") {
            left = argv[++i];
        } else if (arg == "-r") {
            right = argv[++i];
        } else if (arg == "-src") {
            capture_folder = argv[++i];
            // --- Target foder for generated clouds
        } else if (arg == "-dst") {
            dst_folder = argv[++i];
        } else {
            cout << "[Error] Command-line parameter error: unknown option " << argv[i] << endl;
            exit(105);
        }
    }
    //// - Parameters parser END

    //// - Loading calibration parameters section

    // --- Loading intrinsic parameters
    intrinsic_filename = expand_environment_variables(intrinsic_filename);
    FileStorage fs(intrinsic_filename, FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "[Error] Failed to open intrinsic file" << intrinsic_filename << std::endl;
        exit(106);
    }

    fs["M1"] >> M1; // cameraMatrix1
    fs["D1"] >> D1; // distCoeffs1
    fs["M2"] >> M2; // cameraMatrix2
    fs["D2"] >> D2; // distCoeffs2

    // --- Scale images
    M1 *= scale;
    M2 *= scale;
    fs.release();

    // --- Loading extrinsic parameters
    extrinsic_filename = expand_environment_variables(extrinsic_filename);
    fs.open(extrinsic_filename, FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "[Error] Failed to open extrinsic file" << extrinsic_filename << std::endl;
        exit(106);
    }

    fs["R"] >> R; // Rotation matrix
    fs["T"] >> T; // Translation vector

    fs.release();
    //// - Loading calibration parameters END

    //// - Multiple image folders
    string img1_filename,
        img2_filename,
        disparity_filename,
        cloud_filename;

    /* CV_LOAD_IMAGE_UNCHANGED (<0) loads the image as is (including the
     * alpha channel if present)
     * CV_LOAD_IMAGE_GRAYSCALE ( 0) loads the image as an intensity one
     * CV_LOAD_IMAGE_COLOR (>0) loads the image in the RGB format
     */
    int color_mode = alg == STEREO_BM ? 0 : -1;

    Mat img1, img2, disp;

    title = "img";

    //// - Loading images
    img1 = cv::imread(left, color_mode);
    if (!img1.data) {
	std::cout << "[Warning] No frame in: " << img1_filename << std::endl
		  << "[Error] No image pair available." << std::endl;
	exit(107);
    }

    img2 = cv::imread(right, color_mode);
    if (!img2.data) {
	std::cout << "[Warning] No frame in: " << img2_filename << std::endl
		  << "[Error] No image pair available." << std::endl;
	exit(107);
    }
    //// - Loading images END
	
    if (scale != 1.f) {
	int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
	resize(img1, img1, Size(), scale, scale, method);
	resize(img2, img2, Size(), scale, scale, method);
    }

    // check histograms for validity:
    int thresholdLow = 50, thresholdHigh = 256 - 50;
    cout << "L" << endl;
    bool validLeft = checkImgHistogram(img1, thresholdLow, thresholdHigh);
    cout << "R" << endl;
    bool validRight = checkImgHistogram(img2, thresholdLow, thresholdHigh);
    cout << "VAL " << validLeft << " " << validRight << endl;

    cloud_filename = dst_folder + "/cloud.pcd";

    disparity_filename = dst_folder + "/out_disp.jpg";

    cout << "img1_filename: " << img1_filename << endl
	 << "img2_filename: " << img2_filename << endl;

    char res = 'x';
    cout << "HEIHA!" << endl;
    cout << " control :     w <--y-axis      e                                " << endl;
    cout << " control :     |                |      s.. reset to 0            " << endl;
    cout << " control : a --+-- d <--x-axis  |      p.. export point cloud    " << endl;
    cout << " control :     |      z-axis--> |      r.. rotAngle /= 2.        " << endl;
    cout << " control :     x                q      t.. rotAngle *= 2.        " << endl;
    cout << " control :                             m.. save rectified images" << endl;
    cout << " control :s to reset to zero, q to quit   " << endl;
    cout << " cam :    g .. left fov*1.01     h .. r   ctr:      i        1..left cam   " << endl;
    cout << " cam :    b .. left fov/=1.0.1   n .. r          j<-k->l     2..right cam  " << endl;
    cout << " cam :    Tx .. baseline 3 -, 4 + 1.0 mm  " << endl;

    bool doRecompute = true;
    Mat rectified;
    do {
	Mat i1, i2;

	if (doRecompute) {
	    cout << endl;
	    img1.copyTo(i1); // make sure we start from the input data...
	    img2.copyTo(i2);

	    rectifyAndReproject(i1, i2, disp, rectified);
	    dispTo3D(Q, i1, disp, i1.cols / 2, img1.rows / 2);

	    xzPlot = Scalar(128, 128, 128);
	    plotHStripe(disp, i1, xzPlot, Scalar(128, 255, 0), disp.rows / 2 - 200);
	    plotHStripe(disp, i1, xzPlot, Scalar(128, 255, 0), disp.rows / 2 - 100);
	    plotHStripe(disp, i1, xzPlot, Scalar(0, 255, 0), disp.rows / 2);
	    plotHStripe(disp, i1, xzPlot, Scalar(0, 255, 128), disp.rows / 2 + 100);
	    plotHStripe(disp, i1, xzPlot, Scalar(0, 255, 128), disp.rows / 2 + 200);

	    plotVStripe(disp, i1, xzPlot, Scalar(0, 128, 255), disp.cols / 2);
	    plotVStripe(disp, i1, xzPlot, Scalar(0, 128, 255), disp.cols / 2 + 200);
	    plotVStripe(disp, i1, xzPlot, Scalar(0, 128, 255), disp.cols / 2 + 400);
	    imshow(title + "xz", xzPlot);
	}

	Mat& pos = posRight;
	if (chgCenterLeft) { // set camera to control
	    pos = posLeft;
	    cout << "INTRINSIC CAM CONTROL SET TO left" << endl;
	} else {
	    cout << "INTRINSIC CAM CONTROL SET TO right" << endl;
	}

	res = waitKey(0);
	cout << "KEY " << res << endl;
	doRecompute = true;
	Mat m1Mod, m2Mod; // intermed results
	switch (res) {
	case 'a': // left
	    rotVec.at<double>(1) += rotAngle;
	    cout << "rot a " << rotVec << endl;
	    break;
	case 'd': // right
	    rotVec.at<double>(1) -= rotAngle;
	    cout << "rot d " << rotVec << endl;
	    break;
	case 'w': // up
	    rotVec.at<double>(0) -= rotAngle;
	    cout << "rot w " << rotVec << endl;
	    break;
	case 'x': // down
	    rotVec.at<double>(0) += rotAngle;
	    cout << "rot x " << rotVec << endl;
	    break;
	case 's': // right
	    rotVec = 0.f;
	    cout << "rot s " << rotVec << endl;
	    break;
	case 'e': // right
	    rotVec.at<double>(2) += rotAngle;
	    cout << "rot e " << rotVec << endl;
	    break;
	case 'q': // right
	    rotVec.at<double>(2) -= rotAngle;
	    cout << "rot c " << rotVec << endl;
	    break;
	case 'r':
	    rotAngle /= 2.0;
	    mulInc /= 2.0;
	    cout << "angl dec " << rotAngle << " mulInc dec " << mulInc << endl;
	    doRecompute = false;
	    break;
	case 't':
	    rotAngle *= 2.0;
	    mulInc *= 2.0;
	    cout << "angl inc " << rotAngle << " mulInc inc " << mulInc << endl;
	    doRecompute = false;
	    break;
	case 'm':
	    saveRectified = !saveRectified;
	    break;
	case 'g':
	    mulLeft.at<double>(0, 0) *= 1.0 + mulInc;
	    mulLeft.at<double>(1, 1) *= 1.0 + mulInc;
	    cout << " mulInc " << mulInc << " m00 " << mulLeft.at<double>(0, 0) << endl;
	    cout << "m1Mod" << (mulLeft * M1 + posLeft) << endl;
	    break;
	case 'b':
	    mulLeft.at<double>(0, 0) /= 1.0 + mulInc;
	    mulLeft.at<double>(1, 1) /= 1.0 + mulInc;
	    cout << " mulInc " << mulInc << " m00 " << mulLeft.at<double>(0, 0) << endl;
	    cout << "m1Mod" << (mulLeft * M1 + posLeft) << endl;
	    break;
	case 'h':
	    mulRight.at<double>(0, 0) *= 1.0 + mulInc;
	    mulRight.at<double>(1, 1) *= 1.0 + mulInc;
	    cout << " mulInc " << mulInc << " m00 " << mulRight.at<double>(0, 0) << endl;
	    cout << "m2Mod" << (mulRight * M2 + posRight) << endl;
	    break;
	case 'n':
	    mulRight.at<double>(0, 0) /= 1.0 + mulInc;
	    mulRight.at<double>(1, 1) /= 1.0 + mulInc;
	    cout << " mulInc " << mulInc << " m00 " << mulRight.at<double>(0, 0) << endl;
	    cout << "m2Mod" << (mulRight * M2 + posRight) << endl;
	    break;
	case '1':
	    chgCenterLeft = true;
	    break;
	case '2':
	    chgCenterLeft = false;
	    break;
	case '3':
	    T.at<double>(0, 0) -= 0.1;
	    cout << "T " << T << endl;
	    break;
	case '4':
	    T.at<double>(0, 0) += 0.1;
	    cout << "T " << T << endl;
	    break;
	case 'j':
	    pos.at<double>(0, 2) -= posInc;
	    cout << "POS " << pos << endl;
	    break;
	case 'l':
	    pos.at<double>(0, 2) += posInc;
	    cout << "POS " << pos << endl;
	    break;
	case 'i':
	    pos.at<double>(1, 2) -= posInc;
	    cout << "POS " << pos << endl;
	    break;
	case 'k':
	    pos.at<double>(1, 2) += posInc;
	    cout << "POS " << pos << endl;
	    break;
	case 'z':
	    mulDisp *= 2;
	    cout << "mulDisp set to " << mulDisp << endl;
	    break;
	case '<':
	    mulDisp /= 2;
	    cout << "mulDisp set to " << mulDisp << endl;
	    break;
	case 'p':
	    createCloud(Q, rectified, disp, cloud_filename);
	    doRecompute = false;
	    break;
	case '-': {
	    cout << "M1 BEF " << M1 << endl;
	    cout << "M2 BEF " << M2 << endl;
	    Mat m, d;

	    m = M1;
	    M1 = M2;
	    M2 = m;
	    cout << "M1 AFT " << M1 << endl;
	    cout << "M2 AFT " << M2 << endl;

	    d = D1;
	    D1 = D2;
	    D2 = d;
	} break;
	case '=':
	    laplaceOverlay = !laplaceOverlay;
	    break;
	case '0':
	    dumpParams();
	    doRecompute = false;
	    break;
	}
	cout << endl;
	disp = 0;
    } while (res != 27);

    return 0;
}

/**
 * Parses the configuration file and initialices the cofig variables.
 *
 **/
int parseConfig(string param_filename)
{
    cout << "initialize parameters" << endl;

    // intrinsics
    mulLeft = Mat::eye(3, 3, CV_64FC1);
    mulLeft.at<double>(0, 2) = 1.;
    mulLeft.at<double>(1, 2) = 1.;

    mulRight = Mat::eye(3, 3, CV_64FC1);
    mulRight.at<double>(0, 2) = 1.;
    mulRight.at<double>(1, 2) = 1.;


    FileStorage fparam(param_filename, FileStorage::READ);
    if (!fparam.isOpened()) {
        printf("[Error] Failed to open file %s\n", args_filename.c_str());
        return -1;
    }

    // --- Get algorithm
    string _alg = (string)fparam["algorithm"];
    if (_alg == "bm")
        alg = STEREO_BM;
    else if (_alg == "sgbm")
        alg = STEREO_SGBM;
    else if (_alg == "hh")
        alg = STEREO_HH;
    else if (_alg == "var")
        alg = STEREO_VAR;
    else {
        printf("Unknown stereo algorithm\n\n");
        return -1;
    }

    fparam["calib_intrinsic"] >> intrinsic_filename;
    fparam["calib_extrinsic"] >> extrinsic_filename;

    camera_z_offset = (float)fparam["offset"];

    FileNode n = fparam["display_options"];
    n["dispatity"] >> show_disparity;
    n["rectified"] >> show_rectified;
    n["cloud"] >> show_cloud;

    rgb = (int)fparam["rgb"];

    n = fparam["bounding_box"];
    n["max_z"] >> max_z;
    n["min_z"] >> min_z;
    n["max_x"] >> max_x;
    n["min_x"] >> min_x;
    n["max_y"] >> max_y;
    n["min_y"] >> min_y;

    // INPUT

    fparam["rotate_180"] >> rot180;
    fparam["flip_channels"] >> flipChannels;

    // INTRINSICS

    if (fparam["mul_left"].isReal()) {
        double mul;
        fparam["mul_left"] >> mul;

        mulLeft.at<double>(0, 0) = 1. + mul;
        mulLeft.at<double>(1, 1) = 1. + mul;
    }
    if (fparam["pos_left_x"].isReal()) {
        double x, y;
        fparam["pos_left_x"] >> x;
        fparam["pos_left_y"] >> y;
        posLeft.at<double>(0, 2) = x;
        posLeft.at<double>(1, 2) = y;
    }
    if (fparam["mul_right"].isReal()) {
        double mul;
        fparam["mul_right"] >> mul;

        mulRight.at<double>(0, 0) = 1. + mul;
        mulRight.at<double>(1, 1) = 1. + mul;
    }
    if (fparam["pos_right_x"].isReal()) {
        double x, y;
        fparam["pos_right_x"] >> x;
        fparam["pos_right_y"] >> y;
        posRight.at<double>(0, 2) = x;
        posRight.at<double>(1, 2) = y;
    }

    if (fparam["too_bright_pct"].isReal()) {
        fparam["too_bright_pct"] >> tooBrightPct;
    }
    if (fparam["too_dark_pct"].isReal()) {
        fparam["too_dark_pct"] >> tooBrightPct;
    }

    // EXTRINSICS
    // rodrigues input:
    double x, y, z;
    fparam["rot_x"] >> x;
    fparam["rot_y"] >> y;
    fparam["rot_z"] >> z;
    rotVec.at<double>(0, 0) = x;
    rotVec.at<double>(0, 1) = y;
    rotVec.at<double>(0, 2) = z;

    minDisparity = (int)fparam["minDisparity"];
    numberOfDisparities = (int)fparam["numDisparities"];
    SADWindowSize = (int)fparam["blockSize"];
    preFilterCap = (int)fparam["preFilterCap"];
    uniquenessRatio = (int)fparam["uniquenessRatio"];
    speckleWindowSize = (int)fparam["speckleWindowSize"];
    speckleRange = (int)fparam["speckleRange"];
    disp12MaxDiff = (int)fparam["disp12MaxDiff"];
    scale = (float)fparam["scale"];

    cout << "Config file: " << param_filename << " readed." << endl
         << "intrinsic_filename: " << intrinsic_filename << endl
         << "extrinsic_filename: " << extrinsic_filename << endl
         << "camera_z_offset: " << camera_z_offset << endl
         << "alg: " << alg << endl;

    cout << "rotVec " << rotVec.t() << endl;
    cout << "rot180? " << rot180 << endl;
    cout << "flip? " << flipChannels << endl;

    return 1;
}

// borrowed from https://learnopencv.com/rotation-matrix-to-euler-angles/
static Mat rotationMatrixToEulerAngles(const Mat& R, Mat& v)
{

    // assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    Mat vec(Size(1, 3), CV_64FC1);
    vec.at<double>(0) = x;
    vec.at<double>(1) = y;
    vec.at<double>(2) = z;
    v = vec;
    return vec;
}

static int rectifyAndReproject(const Mat& img1, const Mat img2, Mat& disp, Mat& rectified)
{
    Rect roi1, roi2;

    Size img_size = img1.size();

    // Computes the undistortion and rectification transformation map.
    // P1 and P2 -> new camera matrix
    cout << float(clock() - begin_time) / CLOCKS_PER_SEC
         << " time before rectify" << endl;

    //
    // modify intrinsics
    //
    Mat m1Mod = mulLeft * M1 + posLeft;
    Mat m2Mod = mulRight * M2 + posRight;
    Mat d1Mod = D1;
    Mat d2Mod = D2;
    //
    // modify extrinsics
    //

    // optionally apply additional rotations on the secondary cam lens
    Mat rMod, rodrigues;

    Rodrigues(rotVec, rodrigues);
    cout << "ROT VEC " << rotVec.t() << endl;

    if (flipChannels) {
        Mat rr;
        Mat rvec;

        // channels are flipped - use the intrinsics that was used for capturing
        m1Mod = mulLeft * M2 + posLeft;
        m2Mod = mulRight * M1 + posRight;
        d1Mod = D2;
        d2Mod = D1;

        rotationMatrixToEulerAngles(R, rvec);

        rvec.at<double>(0) *= 1.0f;
        rvec.at<double>(1) *= 1.0f;
        rvec.at<double>(2) *= -1.0f;

        cout << "ROT VEC x " << rvec << endl;
        Rodrigues(rvec, rr);
        // cout << "ROT VEC x rr " << rr << endl;

        rMod = rodrigues * rr;
    } else {
        rMod = rodrigues * R;
        // cout << "R " << R << endl;
        // cout << "rotVec " << rotVec << endl;
        // cout << "rodr " << rodrigues << endl;
    }

    cout << "Rmod " << rMod << endl;

    // init output variables
    R1 = 0;
    R2 = 0;
    P1 = 0;
    P2 = 0;
    Q = 0;
    /*
     * Stereorectify
     *
     * - Flag: CALIB_ZERO_DISPARITY . If set, the function makes the principal
     * points of each camera have the same pixel coordinates in the rectified views.
     * - Alpha: Free scaling parameter. If it is -1 or absent, the function performs
     * the default scaling.
     * - NewImageSize: New image resolution after rectification. The same size
     * should be passed to initUndistortRectifyMap(). When (0,0) is passed
     * (default), it is set to the original imageSize . Setting it to larger value
     * can help you preserve details in the original image, especially when there is
     * a big radial distortion.
     * - ValidPixROI1–2: Optional output rectangles inside the rectified images
     * where all the pixels are valid.
     */
    stereoRectify(m1Mod, d1Mod, m2Mod, d2Mod,
        img_size, rMod, T, R1, R2, P1, P2, Q,
        CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2);

    cout << float(clock() - begin_time) / CLOCKS_PER_SEC
         << " time after stereorectify" << endl;

    // cout << P1 << endl;
    Mat map11, map12, map21, map22; // Output maps
    initUndistortRectifyMap(m1Mod, d1Mod, R1, P1, img_size, CV_32F, map11, map12);
    initUndistortRectifyMap(m2Mod, d2Mod, R2, P2, img_size, CV_32F, map21, map22);

    Mat img1r, img2r;
    if (rot180) {
        Mat i1, i2;
        rotate(img1, i2, ROTATE_180);
        rotate(img2, i1, ROTATE_180);
        cout << "ROT180!!!" << endl;
        // n.b.:
        // as the cameras have been calibrated with rotation set to 180,
        // we need to apply the rotation and flip the left/right images
        // to be in the same calibration domain again.
        remap(i1, img1r, map11, map12, INTER_CUBIC);
        remap(i2, img2r, map21, map22, INTER_CUBIC);
    } else {
        cout << "ROT180 NOT!!! FLIP CHAN? " << flipChannels << endl;
        if (flipChannels) {

            remap(img2, img1r, map11, map12, INTER_CUBIC);
            remap(img1, img2r, map21, map22, INTER_CUBIC);
        } else {
            remap(img1, img1r, map11, map12, INTER_CUBIC);
            remap(img2, img2r, map21, map22, INTER_CUBIC);
        }
    }
    rectified = img1r;

    if (false) {
        // just for the fun of it, you might try..
        Sobel(img1r, img1r, 0, 1, 0, 5);
        Sobel(img2r, img2r, 0, 1, 0, 5);
    }

    if (saveRectified) {
        // Write and show rectified images
        string left_filename = dst_folder + "_rect_left.jpg";
        string rite_filename = dst_folder + "_rect_right.jpg";
        imwrite(left_filename, img1r);
        imwrite(rite_filename, img2r);
    }

    cout << float(clock() - begin_time) / CLOCKS_PER_SEC
         << " time after rectify" << endl;

    if (show_rectified) {
        // Write and show rectified images
        // imwrite( "r1_left.jpg", img1r );
        // imwrite( "r1_right.jpg", img2r );

        Mat canvas;
        double sf;
        int w, h;

        sf = 1024. / MAX(img_size.width, img_size.height);
        w = cvRound(img_size.width * sf);
        h = cvRound(img_size.height * sf);
        canvas.create(h, w * 2, CV_8UC3);

        {
            Mat canvasPart = canvas(Rect(0, 0, w, h));
            resize(img1r, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            Rect vroi(cvRound(roi1.x * sf),
                cvRound(roi1.y * sf), cvRound(roi1.width * sf),
                cvRound(roi1.height * sf));

            rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
        }
        {
            Mat canvasPart2 = canvas(Rect(w, 0, w, h));
            resize(img2r, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
            Rect vroi2(cvRound(roi2.x * sf),
                cvRound(roi2.y * sf), cvRound(roi2.width * sf),
                cvRound(roi2.height * sf));
            rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);
        }
        //
        // indicator lines
        //
        for (int j = 0; j < canvas.rows; j += 16) {
            line(canvas, Point(0, j),
                Point(canvas.cols, j), Scalar(256 - (j * 4) & 0xff, 255, (j * 4) & 0xff), 1, 8);
        }
        namedWindow(title + "rectified", 0);
        imshow(title + "rectified", canvas);

        Mat diff;
        absdiff(img1r, img2r, diff);
        imshow(title + "R-L", diff * 4);
    }

    cout << "Creating disparity image..." << endl;

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities
                                                  : ((img_size.width / 8) + 15) & -16;

    if (alg == STEREO_BM) {

        /* Class for computing stereo correspondence using the block matching algorithm
         * Parameters:
         * 	ndisparities – the disparity search range. For each pixel algorithm will
         *		find the best disparity from 0 (default minimum disparity) to
         * 		ndisparities. The search range can then be shifted by changing the
         * 		minimum disparity.
         *	SADWindowSize – the linear size of the blocks compared by the algorithm.
         *		The size should be odd (as the block is centered at the current pixel).
         *		Larger block size implies smoother, though less accurate disparity map.
         *		Smaller block size gives more detailed disparity map, but there is
         *		higher chance for algorithm to find a wrong correspondence.
         */
#if CV_VERSION_MAJOR >= 3
        Ptr<StereoBM> bm = StereoBM::create(16, 9);
#else
        Ptr<StereoBM> bm = createStereoBM(16, 9);
#endif
        bm->setROI1(roi1);
        bm->setROI2(roi2);
        bm->setPreFilterCap(preFilterCap);
        bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
        bm->setMinDisparity(minDisparity);
        bm->setNumDisparities(numberOfDisparities);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(uniquenessRatio);
        bm->setSpeckleWindowSize(speckleWindowSize);
        bm->setSpeckleRange(speckleRange);
        bm->setDisp12MaxDiff(disp12MaxDiff);
        bm->compute(img1r, img2r, disp);

        /*	} else if( alg == STEREO_VAR ) {
            Class for computing stereo correspondence using the variational
            * matching algorithm

            StereoVar var;
            // ignored with USE_AUTO_PARAMS
            var.levels = 3;
            // ignored with USE_AUTO_PARAMS
            var.pyrScale = 0.5;
            var.nIt = 25;
            var.minDisp = -numberOfDisparities;
            var.maxDisp = 0;
            var.poly_n = 3;
            var.poly_sigma = 0.0;
            var.fi = 15.0f;
            var.lambda = 0.03f;
            // ignored with USE_AUTO_PARAMS
            var.penalization = var.PENALIZATION_TICHONOV;
            // ignored with USE_AUTO_PARAMS
            var.cycle = var.CYCLE_V;
            var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS
            | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;
            var(img1, img2, disp);
        */
    } else if (alg == STEREO_SGBM || alg == STEREO_HH) {

        /*
         * Parameters:
         *	minDisparity – Minimum possible disparity value. Normally, it is zero but
         * 		sometimes rectification algorithms can shift images, so this parameter
         *		needs to be adjusted accordingly.
         *	numDisparities – Maximum disparity minus minimum disparity. The value is
         *		always greater than zero. In the current implementation, this parameter
         *		must be divisible by 16.
         *	blockSize – Matched block size. It must be an odd number >=1 . Normally,
         *		it should be somewhere in the 3..11 range.
         */
#if CV_VERSION_MAJOR >= 3
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
#else
        Ptr<StereoSGBM> sgbm = createStereoSGBM(0, 16, 3);
#endif
        sgbm->setPreFilterCap(preFilterCap);
        int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
        sgbm->setBlockSize(sgbmWinSize);

        int cn = img1r.channels();

        sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
        sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
        sgbm->setMinDisparity(minDisparity);
        sgbm->setNumDisparities(numberOfDisparities);
        sgbm->setUniquenessRatio(uniquenessRatio);
        sgbm->setSpeckleWindowSize(speckleWindowSize);
        sgbm->setSpeckleRange(speckleRange);
        sgbm->setDisp12MaxDiff(disp12MaxDiff);
        sgbm->setMode(alg == STEREO_HH ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);

        cout << float(clock() - begin_time) / CLOCKS_PER_SEC
             << " time before creating disparity" << endl;
        sgbm->compute(img1r, img2r, disp);
        cout << float(clock() - begin_time) / CLOCKS_PER_SEC
             << " time after creating disparity" << endl;
    }

    if (rot180) {
        rotate(disp, disp, ROTATE_180);
    }

    if (show_disparity) {
        // Showing and saving disparty map
        Mat disp8, lapla;
        disp.convertTo(disp8, CV_8U, 1. / 16.);

        if (laplaceOverlay) {
            Laplacian(disp8, lapla, CV_16S, 3);
            Mat l;
            lapla.convertTo(l, CV_8U, 1., 16);
            imshow(title + "disparity", disp8 * mulDisp + l);
        } else {

            imshow(title + "disparity", disp8 * mulDisp);
        }
    }
    {
        int x = disp.cols / 2;
        int y = disp.rows / 2;
        cout << "DISP[" << y << "," << x << "]=" << disp.at<short>(y, x) / 16. << endl;
    }

    return 1;
}

static pcl::PointXYZRGB dispToPtRGB(const Mat& Q, const Mat& img1, const Mat& disp, int x, int y)
{
    pcl::PointXYZRGB point;
    double Q03, Q13, Q23, Q32, Q33;
    // Get the interesting parameters from Q
    Q03 = Q.at<double>(0, 3);
    Q13 = Q.at<double>(1, 3);
    Q23 = Q.at<double>(2, 3);
    Q32 = Q.at<double>(3, 2);
    Q33 = Q.at<double>(3, 3);

    double px, py, pz;
    uchar pr, pg, pb;
    int i = y;
    int j = x;

    const uchar* rgb_ptr = img1.ptr<uchar>(i);
    const short* disp_ptr = disp.ptr<short>(i);

    double d = disp_ptr[j] / 16.;

    if (d == -1) {
        point.z = -1;
        return point; // Discard bad pixels
    }
    double pw = static_cast<double>(d) * Q32 + Q33;
    px = static_cast<double>(j) + Q03;
    py = static_cast<double>(i) + Q13;
    pz = Q23;

    px = px / pw;
    py = py / pw;
    pz = pz / pw;

    // Get RGB info
    pb = rgb_ptr[3 * j];
    pg = rgb_ptr[3 * j + 1];
    pr = rgb_ptr[3 * j + 2];

    // Insert info into point cloud structure
    point.x = px / 100.;
    point.y = py / 100.;
    point.z = (pz + camera_z_offset) / 100.;

    uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
    void* prgb = &rgb; // explicit pointer to overcome compiler warning..
    point.rgb = *reinterpret_cast<float*>(prgb);

    return point;
}

/** print 3d point information given a disparity image & iamge coordinates.*/
static void dispTo3D(Mat Q, Mat img1, Mat disp, int x, int y)
{
    double Q03, Q13, Q23, Q32, Q33;
    // Get the interesting parameters from Q
    Q03 = Q.at<double>(0, 3);
    Q13 = Q.at<double>(1, 3);
    Q23 = Q.at<double>(2, 3);
    Q32 = Q.at<double>(3, 2);
    Q33 = Q.at<double>(3, 3);

    double px, py, pz;

    if (rgb) {
        pcl::PointXYZRGB point = dispToPtRGB(Q, img1, disp, x, y);
#if PCL_FOUND
        cout << "CLOUD rgb center " << point << endl;
#endif
    } else {
        pcl::PointXYZ point;
        int i = y;
        int j = x;

        short* disp_ptr = disp.ptr<short>(i);

        // Get 3D coordinates
        double d = disp_ptr[j] / 16.;

        if (d == -1) {
            cout << "CLOUD center DISPARITY INVALID!" << endl;

            return; // Discard bad pixels
        }

        double pw = static_cast<double>(d) * Q32 + Q33;
        px = static_cast<double>(j) + Q03;
        py = static_cast<double>(i) + Q13;
        pz = Q23;

        px = px / pw;
        py = py / pw;
        pz = pz / pw;

        // Insert info into point cloud structure
        point.x = px / 100.;
        point.y = py / 100.;
        point.z = (pz + camera_z_offset) / 100.;
#if PCL_FOUND
        cout << "CLOUD center " << point << endl;
#endif
    }
}

static void createCloud(Mat Q, Mat img1, Mat disp, string cloud_filename)
{
#if PCL_FOUND
    //// - point cloud construction
    cout << "********* Creating Point Cloud..." << endl;
    Mat colors;
    if (rot180) {
        rotate(img1, colors, ROTATE_180);
    } else {
        colors = img1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr
        pointCloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        pointCloud_ptr_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    double Q03, Q13, Q23, Q32, Q33;
    // Get the interesting parameters from Q
    Q03 = Q.at<double>(0, 3);
    Q13 = Q.at<double>(1, 3);
    Q23 = Q.at<double>(2, 3);
    Q32 = Q.at<double>(3, 2);
    Q33 = Q.at<double>(3, 3);

    double px, py, pz;
    cout << float(clock() - begin_time) / CLOCKS_PER_SEC
         << " time before cloud created" << endl;
    if (rgb) {

        uchar pr, pg, pb;

        for (int i = 0; i < img1.rows; i++) {
            uchar* rgb_ptr = colors.ptr<uchar>(i);
            short* disp_ptr = disp.ptr<short>(i);
            for (int j = 0; j < img1.cols; j++) {
                // Get 3D coordinates

                double d = disp_ptr[j] / 16.;

                if (d == -1)
                    continue; // Discard bad pixels

                double pw = static_cast<double>(d) * Q32 + Q33;
                px = static_cast<double>(j) + Q03;
                py = static_cast<double>(i) + Q13;
                pz = Q23;

                px = px / pw;
                py = py / pw;
                pz = pz / pw;

                // Get RGB info
                pb = rgb_ptr[3 * j];
                pg = rgb_ptr[3 * j + 1];
                pr = rgb_ptr[3 * j + 2];

                // Insert info into point cloud structure
                pcl::PointXYZRGB point;
                point.x = px / 100.;
                point.y = py / 100.;
                point.z = (pz + camera_z_offset) / 100.;

                // bounding box filtering
                if (point.z > max_z || point.z < min_z)
                    continue;
                if (point.x < min_x || point.x > max_x)
                    continue;
                if (point.y < min_y || point.y > max_y)
                    continue;

                uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
                void* prgb = &rgb; // explicit pointer to overcome compiler warning..
                point.rgb = *reinterpret_cast<float*>(prgb);

                pointCloud_ptr_rgb->points.push_back(point);

                if (i == img1.rows / 2 && j == img1.cols / 2) {
                    // center point - print it
                    cout << "CLOUD rgb center " << point << endl;
                }
            }
        }
        pointCloud_ptr_rgb->width = (int)pointCloud_ptr_rgb->points.size();
        pointCloud_ptr_rgb->height = 1;
        cout << "cloud pts " << pointCloud_ptr_rgb->width << endl;
        if (pcl::io::savePCDFile(cloud_filename, *pointCloud_ptr_rgb, true) != 0) {
            cout << "[Error] Could not create cloud, empty." << endl;
            exit(108);
        }
        cout << "PCD file saved" << endl;
    } else {

        pcl::PointXYZ point;

        for (int i = 0; i < img1.rows; i++) {
            short* disp_ptr = disp.ptr<short>(i);
            for (int j = 0; j < img1.cols; j++) {

                // Get 3D coordinates
                double d = disp_ptr[j] / 16.;

                if (d == -1)
                    continue; // Discard bad pixels

                double pw = static_cast<double>(d) * Q32 + Q33;
                px = static_cast<double>(j) + Q03;
                py = static_cast<double>(i) + Q13;
                pz = Q23;

                px = px / pw;
                py = py / pw;
                pz = pz / pw;

                // Insert info into point cloud structure
                point.x = px / 100.;
                point.y = py / 100.;
                point.z = (pz + camera_z_offset) / 100.;

                if (i == img1.rows / 2 && j == img1.cols / 2) {
                    // center point - print it
                    cout << "CLOUD center " << point << endl;
                }

                // bounding box filtering
                if (point.z > max_z || point.z < min_z)
                    continue;
                if (point.x < min_x || point.x > max_x)
                    continue;
                if (point.y < min_y || point.y > max_y)
                    continue;

                pointCloud_ptr->points.push_back(point);
            }
        }
        pointCloud_ptr->width = (int)pointCloud_ptr->points.size();
        pointCloud_ptr->height = 1;
        if (pcl::io::savePCDFileBinary(cloud_filename, *pointCloud_ptr) != 0) {
            cout << "[Error] Could not create cloud, empty." << endl;
            exit(108);
        }
        cout << "PCD file saved" << endl;
    }

    cout << float(clock() - begin_time) / CLOCKS_PER_SEC
         << " time after cloud created" << endl;

#if PCL_VIZ
    // Helps for debuging
    if (show_cloud) {
        // Create visualizer
        pcl::visualization::PCLVisualizer::Ptr
            viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0); // black
        if (rgb) {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
                rgb(pointCloud_ptr_rgb);
            viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud_ptr_rgb,
                rgb, "reconstruction");
        } else {
            viewer->addPointCloud<pcl::PointXYZ>(pointCloud_ptr,
                "reconstruction");
        }
        // viewer->addCoordinateSystem ( 1.0 ,0);
        viewer->initCameraParameters();

        // Main loop
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }
#endif
#else // PCL_FOUND
    cout << "PCL not compiled in - ignoring createCloud" << endl;
#endif
}

static const bool dbgHist = true;

static bool checkImgHistogram(const Mat& img, int thresholdLow, int thresholdHigh)
{
    Mat m;
    resize(img, m, m.size(), 1. / 8., 1. / 8., INTER_LINEAR);

    vector<Mat> bgr_planes;
    split(m, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 }; // the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

    int bMax = b_hist.at<float>(0), bMaxIdx = 0;
    int gMax = g_hist.at<float>(0), gMaxIdx = 0;
    int rMax = r_hist.at<float>(0), rMaxIdx = 0;
    int numHi = 0, numLo = 0;
    float pctHi = 0.f, pctLo = 0.f;

    if (dbgHist)
        cout << "B;"; //////////////////////////////////////////////////
    for (int i = 0; i < histSize; i++) {
        if (dbgHist)
            cout << b_hist.at<float>(i) << ";";
        if (b_hist.at<float>(i) > bMax) {
            bMax = b_hist.at<float>(i);
            bMaxIdx = i;
        }
    }
    if (dbgHist)
        cout << ";;MAX;" << bMax << ";" << bMaxIdx;

    numLo = 0;
    for (int i = 0; i < thresholdLow; i++) {
        numLo += b_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMLO;" << numLo << "LO%;" << (numLo / (float)(m.rows * m.cols)) << endl;

    numHi = 0;
    for (int i = thresholdHigh; i < histSize; i++) {
        numHi += b_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMHI;" << numHi << "HI%;" << (numHi / (float)(m.rows * m.cols)) << endl;

    if (dbgHist)
        cout << "G;"; //////////////////////////////////////////////////
    for (int i = 0; i < histSize; i++) {
        if (dbgHist)
            cout << b_hist.at<float>(i) << ";";
        if (g_hist.at<float>(i) > gMax) {
            gMax = g_hist.at<float>(i);
            gMaxIdx = i;
        }
    }
    if (dbgHist)
        cout << ";;MAX;" << gMax << ";" << gMaxIdx;

    numLo = 0;
    for (int i = 0; i < thresholdLow; i++) {
        numLo += g_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMLO;" << numLo << "LO%;" << (numLo / (float)(m.rows * m.cols)) << endl;
    pctLo = (numLo / (float)(m.rows * m.cols));

    numHi = 0;
    for (int i = thresholdHigh; i < histSize; i++) {
        numHi += b_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMHI;" << numHi << "HI%;" << (numHi / (float)(m.rows * m.cols)) << endl;
    pctHi = (numHi / (float)(m.rows * m.cols));

    if (dbgHist)
        cout << "R;"; //////////////////////////////////////////////////
    for (int i = 0; i < histSize; i++) {
        if (dbgHist)
            cout << b_hist.at<float>(i) << ";";
        if (r_hist.at<float>(i) > rMax) {
            rMax = r_hist.at<float>(i);
            rMaxIdx = i;
        }
    }
    if (dbgHist)
        cout << ";;MAX;" << rMax << ";" << rMaxIdx;

    numLo = 0;
    for (int i = 0; i < thresholdLow; i++) {
        numLo += g_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMLO;" << numLo << "LO%;" << (numLo / (float)(m.rows * m.cols)) << endl;

    numHi = 0;
    for (int i = thresholdHigh; i < histSize; i++) {
        numHi += b_hist.at<float>(i);
    }
    if (dbgHist)
        cout << ";NUMHI;" << numHi << "HI%;" << (numHi / (float)(m.rows * m.cols)) << endl;
    if (dbgHist)
        cout << endl;

    // EVAL //////////////////////////////////////////////////

    if (pctHi > tooBrightPct) {
        cout << "TOO BRIGHT! %= " << pctHi * 100 << endl;
        return false;
    }
    if (pctLo > tooDarkPct) {
        cout << "TOO DARK! %= " << pctLo * 100 << endl;
        return false;
    }

    return true;
}

static std::string expand_environment_variables(std::string s)
{
    if (s.find("${") == std::string::npos)
        return s;

    std::string pre = s.substr(0, s.find("${"));
    std::string post = s.substr(s.find("${") + 2);

    if (post.find('}') == std::string::npos)
        return s;

    std::string variable = post.substr(0, post.find('}'));
    std::string value = "";

    post = post.substr(post.find('}') + 1);

    if (getenv(variable.c_str()) != NULL)
        value = std::string(getenv(variable.c_str()));

    return expand_environment_variables(pre + value + post);
}

static void plotHStripe(const Mat& disp, const Mat& img, Mat& plotTo, Scalar color, int y)
{
    int w = plotTo.cols;
    int h = plotTo.rows;

    int camX = w / 2;
    int camZ = 10;

    circle(plotTo, Point(camX, camZ), 5, Scalar(0, 255, 255));

    for (int i = 0; i < w; i++) {
        pcl::PointXYZRGB pt = dispToPtRGB(Q, img, disp, i, y);
        int xCm = pt.x * 100;
        int zCm = pt.z * 100;
        circle(plotTo, Point(xCm + camX, zCm + camZ), 1, color);
    }
}
static void plotVStripe(const Mat& disp, const Mat& img, Mat& plotTo, Scalar color, int x)
{
    int w = plotTo.cols;
    int h = plotTo.rows;

    int camX = 10; // w / 2;
    int camZ = h / 2; // 10;

    circle(plotTo, Point(camX, camZ), 5, Scalar(0, 255, 255));

    for (int i = 0; i < h; i++) {
        pcl::PointXYZRGB pt = dispToPtRGB(Q, img, disp, x, i);
        int yCm = pt.y * 100;
        int zCm = pt.z * 100;
        circle(plotTo, Point(zCm + camX, yCm + camZ), 1, color);
    }
}

static void dumpParams()
{
    cout << " INTRINSICS --------------- M1 " << endl;
    cout << "M1 " << M1 << endl;
    cout << "D1 " << D1 << endl;
    cout << " ---- " << endl;
    cout << " mulLeft " << mulLeft << " mul_left " << (1. - mulLeft.at<double>(0, 0)) << endl;
    cout << " posLeft " << posLeft << endl;
    cout << " m1Mod " << (mulLeft * M1 + posLeft) << endl;
    cout << " INTRINSICS --------------- M2 " << endl;
    cout << "M2 " << M1 << endl;
    cout << "D2 " << D1 << endl;
    cout << " ---- " << endl;
    cout << " mulRight " << mulRight << " mul_right " << (1. - mulRight.at<double>(0, 0)) << endl;
    cout << " posRight " << posRight << endl;
    cout << " m2Mod " << (mulRight * M2 + posRight) << endl;
    cout << endl;
    cout << " EXTRINSICS --------------- " << endl;
    cout << " rotVec " << rotVec << endl;
    cout << " T " << T << endl;
    cout << " -------------------------- " << endl;
    cout << " Q " << Q << endl;
    cout << " -------------------------- " << endl;
    cout << " <!-- match.conf --> " << endl;
    cout << " <pos_left_x></pos_left_x> " << endl;
}
