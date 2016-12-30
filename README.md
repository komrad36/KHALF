
Optimized special-case bilinear interpolation
in which the width is halved and the height is unchanged,
for example for split-screening two frames or streams
for visual odometry in computer vision.

KHALF is written partly in AVX2
so an AVX2-ready CPU is required.

For the more general case see KLERP and CUDALERP:
https://github.com/komrad36/KLERP.
https://github.com/komrad36/CUDALERP.

All functionality is contained in the header 'KHALF.h'
and has no external dependencies at all.

Note that these are intended for computer vision use
(hence the speed) and are designed for color (24-bit) images.

The file 'main.cpp' is an example and speed test driver.
It uses OpenCV for display and result comparison.
