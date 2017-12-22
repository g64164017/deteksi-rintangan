#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

Mat ProcessHoughLines(Mat src){
    Mat dst, cdst;

    // preprocess
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
//    threshold( src, dst, 120, 255, 1);
    GaussianBlur(src, dst, Size(7,7), 1.5,1.5);
    imshow("blur",dst);
    //    Canny(dst, dst, 10, 40, 3);
    Canny(dst, dst, 50, 50, 3); ///EDGE DETECTION
//    bitwise_not ( dst, dst );
    imshow("canny",dst);
    cvtColor(src, cdst, CV_GRAY2BGR);

    // hough transform
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 75, 30, 25 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }

//    vector<Vec2f> lines2;
//    HoughLines( dst, lines2, 1, CV_PI/180, 120 );
//
//    for( size_t i = 0; i < lines2.size(); i++ )
//    {
//        float rho = lines2[i][0];
//        float theta = lines2[i][1];
//        double a = cos(theta), b = sin(theta);
//        double x0 = a*rho, y0 = b*rho;
//        Point pt1(cvRound(x0 + 1000*(-b)),
//                  cvRound(y0 + 1000*(a)));
//        Point pt2(cvRound(x0 - 1000*(-b)),
//                  cvRound(y0 - 1000*(a)));
//        line( cdst, pt1, pt2, Scalar(255,0,0), 3, 8 );
//    }

    return cdst;
}

//Mat ProcessHoughCircle(Mat src){
//  Mat dst, cdst;
//  int sum = 0;
//
//  ///MENGURANGI NOISE
//  GaussianBlur( src, dst, Size(3, 3), 2, 2 );
//  //medianBlur(src, dst, 3);
//  //cvtColor(dst, cdst, CV_GRAY2BGR);
//  //cdst = dst.clone();
//
//
//  vector<Vec3f> circles;
//
//  /// LIBRARY HOUGH UNTUK MENDETEKSI LINGKARAN
//  HoughCircles( dst, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 50, 0, 0 );
//
//  /// ALGORITMA MENGGAMBAR LINGKARAN
//  for( size_t i = 0; i < circles.size(); i++ )
//  {
//      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//      int radius = cvRound(circles[i][2]);
//      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );// circle center
//      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );// circle outline
//   }
//   /// PUT TEXT
////    putText(src,
////        to_string(circles.size()),
////        Point(80,200), FONT_HERSHEY_SIMPLEX,1,Scalar(200,0,200),4
////    );
////    cout << "Count = " << circles.size() << endl;
//   return src;
//}

int main()
{
    Mat resultHoughLine, resultHoughCircle;

////    VideoCapture cap(1);
////    VideoCapture cap("http://nasrul_hamid:16017@172.20.33.156:4747/mjpegfeed?640x480");
//    VideoCapture cap("../../videos/pedestrian2.mp4");
//    Mat src;
//
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
//
//    while (true)
//    {
//        cap >> src;
//        cvtColor(src,src,CV_BGR2GRAY);
//        resize(src,src,cv::Size(640, 480));
//        imshow("Original", src);
//        resultHoughLine = ProcessHoughLines(src);
//        imshow("Hough Line", resultHoughLine);
//
//        if (waitKey(10) == 'q')
//            break;
//    }


//    Mat src = imread("/home/nasrulhamid/G64164017/S3/projects/ppcd/gambar/busway/busway05.png",0);
//    if(!src.data){
//        return -1;
//    }

    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    String folder = "/home/nasrulhamid/G64164017/S3/projects/ppcd/gambar/busway/"; // again we are using the Opencv's embedded "String" class

    glob(folder, filenames); // new function that does the job ;-)

    for(size_t i = 0; i < filenames.size(); ++i)
    {
        Mat src = imread(filenames[i],0);

        if(!src.data)
            cerr << "Problem loading image!!!" << endl;

        resize(src,src,cv::Size(400, 300));
        imshow("Original", src);
        resultHoughLine = ProcessHoughLines(src);
        imshow("Hough Line", resultHoughLine);

        waitKey();
    }

    return 0;
}

///REFERENSI LAIN: http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
///KATA KUNCI:
///OPEN CV SHAPE DETECTION ATAU SHAPE DETECTION
///CARI TAHU SOAL BLOB, CONTOUR, TEMPLATE MATCHING
///HAAR CLASSIFIER -> UNTUK MENDAPATKAN POLA SUATU OBJEK TERTENTU (SELAIN DARI OBJEK DEFAULT SEPERTI CIRCLE, LINE, ECLIPSE)


