#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

Mat ProcessHoughLines(Mat src){
    Mat dst, cdst;

    Canny(src, dst, 50, 200, 3); ///EDGE DETECTION
    cvtColor(dst, cdst, CV_GRAY2BGR);
    //cdst = dst.clone();
#if 0
    vector<Vec2f> lines, lines2;///VARIABLE UNTUK MENAMPUNG GAMBAR GARIS

    /// LIBRARY HOUGH UNTUK MENDETEKSI GARIS
    HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 );

    /// ALGORITMA MENGGAMBAR GARIS
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }
#else
    vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
#endif
    return cdst;
}

Mat ProcessHoughCircle(Mat src){
  Mat dst, cdst;
  int sum = 0;

  ///MENGURANGI NOISE
  GaussianBlur( src, dst, Size(3, 3), 2, 2 );
  //medianBlur(src, dst, 3);
  //cvtColor(dst, cdst, CV_GRAY2BGR);
  //cdst = dst.clone();


  vector<Vec3f> circles;

  /// LIBRARY HOUGH UNTUK MENDETEKSI LINGKARAN
  HoughCircles( dst, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 50, 0, 0 );

  /// ALGORITMA MENGGAMBAR LINGKARAN
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );// circle center
      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );// circle outline
   }
   /// PUT TEXT
//    putText(src,
//        to_string(circles.size()),
//        Point(80,200), FONT_HERSHEY_SIMPLEX,1,Scalar(200,0,200),4
//    );
//    cout << "Count = " << circles.size() << endl;
   return src;
}

int main()
{
//    VideoCapture cap(1);
//    VideoCapture cap("http://nasrul_hamid:16017@172.20.33.156:4747/mjpegfeed?640x480");
    VideoCapture cap("../videos/lab1.mp4");

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    Mat src;

//    Mat src = imread("upload/coin2.jpg", 0);

//    if(!src.data){
//        return -1;
//    }

    cout << "Pilihan Transform:\n"
         << "1. Hough Transform Line\n"
         << "2. Hough Transform Circle\n";

    int pilihan;
    cin >> pilihan;

    Mat resultHoughLine, resultHoughCircle;

    while (true)
    {
        cap >> src;
        cvtColor(src,src,CV_BGR2GRAY);
//        GaussianBlur(src, src, Size(7,7), 1.5, 1.5);
//        Canny(src, src, 50, 200, 3);
        resize(src,src,cv::Size(640, 480));
        switch(pilihan){
            case 1:

                imshow("Original", src);
                resultHoughLine = ProcessHoughLines(src);
                imshow("Hough Line", resultHoughLine);
                break;

            case 2:

                imshow("Original", src);
                resultHoughCircle = ProcessHoughCircle(src);
                imshow("Hough Circle", resultHoughCircle);
                break;
        }

        if (waitKey(10) == 'q')
            break;
    }
    return 0;
}

///REFERENSI LAIN: http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
///KATA KUNCI:
///OPEN CV SHAPE DETECTION ATAU SHAPE DETECTION
///CARI TAHU SOAL BLOB, CONTOUR, TEMPLATE MATCHING
///HAAR CLASSIFIER -> UNTUK MENDAPATKAN POLA SUATU OBJEK TERTENTU (SELAIN DARI OBJEK DEFAULT SEPERTI CIRCLE, LINE, ECLIPSE)


