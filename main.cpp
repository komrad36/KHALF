/*******************************************************************
*   main.cpp
*   KHALF
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Dec 30, 2016
*******************************************************************/
// 
// Optimized special-case bilinear interpolation
// in which the width is halved and the height is unchanged,
// for example for split-screening two frames or streams
// for visual odometry in computer vision.
//
// KHALF is written partly in AVX2
// so an AVX2-ready CPU is required.
//
// For the more general case see KLERP and CUDALERP:
// https://github.com/komrad36/KLERP.
// https://github.com/komrad36/CUDALERP.
//
// All functionality is contained in the header 'KHALF.h'
// and has no external dependencies at all.
//
// Note that these are intended for computer vision use
// (hence the speed) and are designed for color (24-bit) images.
//
// The file 'main.cpp' is an example and speed test driver.
// It uses OpenCV for display and result comparison.
// 

#include <chrono>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "KHALF.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 200;
	constexpr int runs = 500;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------

	const int oldw = image.cols;
	const int h = image.rows;
	const int neww = oldw / 2;

	auto CPU_out = new uint8_t[3 * neww * h];

	cv::Mat dst;

	// OpenCV
	for (int i = 0; i < warmups; ++i) {
		cv::resize(image, dst, cv::Size(neww, h));
	}

	auto start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) {
		cv::resize(image, dst, cv::Size(neww, h));
	}
	auto end = high_resolution_clock::now();
	auto sum = (end - start) / runs;

	std::cout << "OpenCV took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;

	// KHALF
	for (int i = 0; i < warmups; ++i) {
		KHALF(image.data, oldw, h, CPU_out, neww);
	}

	start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) {
		KHALF(image.data, oldw, h, CPU_out, neww);
	}
	end = high_resolution_clock::now();
	sum = (end - start) / runs;

	std::cout << "KHALF took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;

	std::cout << "Input stats: " << h << " rows, " << oldw << " cols." << std::endl;
	std::cout << "Output stats: " << h << " rows, " << neww << " cols." << std::endl;

	cv::namedWindow("OpenCV", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	cv::imshow("OpenCV", dst);

	cv::namedWindow("KHALF", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	cv::imshow("KHALF", cv::Mat(h, neww, CV_8UC3, CPU_out, 3*neww));

	cv::waitKey(0);

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < neww; ++x) {
			if (abs(static_cast<int>(dst.data[y*neww + x]) - static_cast<int>(CPU_out[y*neww + x])) > 1) {
				std::cout << "MISMATCH at (" << x << ", " << y << "). OpenCV " << +dst.data[y*neww + x] << ", me " << +CPU_out[y*neww + x] << '.' << std::endl;
			}
		}
	}
}