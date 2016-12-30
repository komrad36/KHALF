/*******************************************************************
*   KHALF.h
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

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <thread>

constexpr uint8_t testsrc[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64 };

void _KHALF(uint8_t* __restrict src, uint8_t* __restrict dst, const int32_t src_w, const int32_t src_h, int32_t start, int32_t stride, int32_t dst_step) {
	const int32_t dst_w = src_w / 2;
	src += 3 * start * src_w;
	dst += 3 * start * dst_step;
	for (int32_t y = start; y < start + stride - 1; ++y, src += 3 * src_w, dst += 3 * dst_step) {
		for (int32_t x = 0; x < 3 * dst_w; x += 15) {
			__m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2*x));
			__m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2*x + 3));

			__m256i v = _mm256_avg_epu8(va, vb);
			__m256i s = _mm256_shuffle_epi8(v, _mm256_setr_epi8(0, 1, 2, 6, 7, 8, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 19, 20, 24, 25, 26, -1));
			__m128i res = _mm_or_si128(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), res);
		}
	}

	int32_t x = 0;
	for (; x <= 3 * dst_w - 30; x += 15) {
		__m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x));
		__m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x + 3));

		__m256i v = _mm256_avg_epu8(va, vb);
		__m256i s = _mm256_shuffle_epi8(v, _mm256_setr_epi8(0, 1, 2, 6, 7, 8, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 19, 20, 24, 25, 26, -1));
		__m128i res = _mm_or_si128(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), res);
	}

	__m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x));
	__m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x + 3));

	__m256i v = _mm256_avg_epu8(va, vb);
	__m256i s = _mm256_shuffle_epi8(v, _mm256_setr_epi8(0, 1, 2, 6, 7, 8, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 19, 20, 24, 25, 26, -1));
	__m128i res = _mm_or_si128(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));

	_mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), res);
	*reinterpret_cast<uint32_t*>(dst + x + 8) = _mm_extract_epi32(res, 2);
	*reinterpret_cast<uint16_t*>(dst + x + 12) = static_cast<uint16_t>(_mm_extract_epi16(res, 6));
	*reinterpret_cast<uint8_t*>(dst + x + 14) = static_cast<uint8_t>(_mm_extract_epi8(res, 14));
	x += 15;

	for (; x < 3 * dst_w; x += 3) {
		dst[x] = (static_cast<int32_t>(src[2 * x]) + static_cast<int32_t>(src[2 * x + 3]) + 1) / 2;
		dst[x + 1] = (static_cast<int32_t>(src[2 * x + 1]) + static_cast<int32_t>(src[2 * x + 4]) + 1) / 2;
		dst[x + 2] = (static_cast<int32_t>(src[2 * x + 2]) + static_cast<int32_t>(src[2 * x + 5]) + 1) / 2;
	}
}

void KHALF(uint8_t* __restrict src, int32_t src_w, int32_t src_h, uint8_t* __restrict dst, uint32_t dst_step) {
	static const int32_t hw_concur = static_cast<int32_t>(std::thread::hardware_concurrency());
	static std::future<void>* const __restrict fut = new std::future<void>[hw_concur];

	const int32_t stride = (src_h - 1) / hw_concur + 1;
	int32_t i = 0;
	int32_t start = 0;
	for (; i < std::min(src_h - 1, hw_concur - 1); ++i, start += stride) {
		fut[i] = std::async(std::launch::async, _KHALF, src, dst, src_w, src_h, start, stride, dst_step);
	}
	fut[i] = std::async(std::launch::async, _KHALF, src, dst, src_w, src_h, start, src_h - start, dst_step);
	for (int32_t j = 0; j <= i; ++j) fut[j].wait();
}