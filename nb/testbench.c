#include "nb.h"

/**
* The main function compares results between Python implementation (sklearn library)
* and C implementation.
**/
int main(){
	unsigned i, correct = 0;

	FILE* fp;
  	char buf[2];
  	unsigned golden_label, count, count_errors;

	double x_train[][NB_FEATURES] =
	{{2.0,125.0,60.0,20.0,140.0,33.8,0.088,31.0},{3.0,130.0,64.0,0.0,0.0,23.1,0.314,22.0},{6.0,107.0,88.0,0.0,0.0,36.8,0.727,31.0},{4.0,96.0,56.0,17.0,49.0,20.8,0.34,26.0},{0.0,100.0,70.0,26.0,50.0,30.8,0.597,21.0},{8.0,120.0,78.0,0.0,0.0,25.0,0.409,64.0},{1.0,120.0,80.0,48.0,200.0,38.9,1.162,41.0},{13.0,106.0,70.0,0.0,0.0,34.2,0.251,52.0},{2.0,105.0,75.0,0.0,0.0,23.3,0.56,53.0},{9.0,165.0,88.0,0.0,0.0,30.4,0.302,49.0},{1.0,139.0,62.0,41.0,480.0,40.7,0.536,21.0},{6.0,109.0,60.0,27.0,0.0,25.0,0.206,27.0},{2.0,108.0,64.0,0.0,0.0,30.8,0.158,21.0},{1.0,71.0,78.0,50.0,45.0,33.2,0.422,21.0},{0.0,119.0,66.0,27.0,0.0,38.8,0.259,22.0},{8.0,179.0,72.0,42.0,130.0,32.7,0.719,36.0},{1.0,124.0,74.0,36.0,0.0,27.8,0.1,30.0},{2.0,101.0,58.0,35.0,90.0,21.8,0.155,22.0},{6.0,165.0,68.0,26.0,168.0,33.6,0.631,49.0},{2.0,108.0,62.0,32.0,56.0,25.2,0.128,21.0},{3.0,123.0,100.0,35.0,240.0,57.3,0.88,22.0},{5.0,158.0,70.0,0.0,0.0,29.8,0.207,63.0},{3.0,111.0,56.0,39.0,0.0,30.1,0.557,30.0},{1.0,86.0,66.0,52.0,65.0,41.3,0.917,29.0},{5.0,109.0,62.0,41.0,129.0,35.8,0.514,25.0},{3.0,169.0,74.0,19.0,125.0,29.9,0.268,31.0},{1.0,109.0,38.0,18.0,120.0,23.1,0.407,26.0},{1.0,93.0,70.0,31.0,0.0,30.4,0.315,23.0},{0.0,102.0,75.0,23.0,0.0,0.0,0.572,21.0},{2.0,100.0,70.0,52.0,57.0,40.5,0.677,25.0},{1.0,112.0,72.0,30.0,176.0,34.4,0.528,25.0},{1.0,100.0,74.0,12.0,46.0,19.5,0.149,28.0},{9.0,145.0,80.0,46.0,130.0,37.9,0.637,40.0},{3.0,129.0,64.0,29.0,115.0,26.4,0.219,28.0},{3.0,107.0,62.0,13.0,48.0,22.9,0.678,23.0},{0.0,146.0,82.0,0.0,0.0,40.5,1.781,44.0},{4.0,144.0,58.0,28.0,140.0,29.5,0.287,37.0},{13.0,126.0,90.0,0.0,0.0,43.4,0.583,42.0},{6.0,119.0,50.0,22.0,176.0,27.1,1.318,33.0},{1.0,173.0,74.0,0.0,0.0,36.8,0.088,38.0},{4.0,136.0,70.0,0.0,0.0,31.2,1.182,22.0},{1.0,180.0,0.0,0.0,0.0,43.3,0.282,41.0},{4.0,92.0,80.0,0.0,0.0,42.2,0.237,29.0},{6.0,117.0,96.0,0.0,0.0,28.7,0.157,30.0},{1.0,96.0,122.0,0.0,0.0,22.4,0.207,27.0},{1.0,102.0,74.0,0.0,0.0,39.5,0.293,42.0},{0.0,95.0,64.0,39.0,105.0,44.6,0.366,22.0},{11.0,120.0,80.0,37.0,150.0,42.3,0.785,48.0},{9.0,156.0,86.0,0.0,0.0,24.8,0.23,53.0},{0.0,189.0,104.0,25.0,0.0,34.3,0.435,41.0},{6.0,137.0,61.0,0.0,0.0,24.2,0.151,55.0},{3.0,100.0,68.0,23.0,81.0,31.6,0.949,28.0},{1.0,93.0,56.0,11.0,0.0,22.5,0.417,22.0},{1.0,147.0,94.0,41.0,0.0,49.3,0.358,27.0},{5.0,166.0,72.0,19.0,175.0,25.8,0.587,51.0},{1.0,90.0,62.0,12.0,43.0,27.2,0.58,24.0},{4.0,184.0,78.0,39.0,277.0,37.0,0.264,31.0},{9.0,154.0,78.0,30.0,100.0,30.9,0.164,45.0},{7.0,159.0,66.0,0.0,0.0,30.4,0.383,36.0},{3.0,171.0,72.0,33.0,135.0,33.3,0.199,24.0},{4.0,171.0,72.0,0.0,0.0,43.6,0.479,26.0},{9.0,106.0,52.0,0.0,0.0,31.2,0.38,42.0},{1.0,153.0,82.0,42.0,485.0,40.6,0.687,23.0},{3.0,180.0,64.0,25.0,70.0,34.0,0.271,26.0},{2.0,109.0,92.0,0.0,0.0,42.7,0.845,54.0},{1.0,80.0,55.0,0.0,0.0,19.1,0.258,21.0},{3.0,174.0,58.0,22.0,194.0,32.9,0.593,36.0},{8.0,151.0,78.0,32.0,210.0,42.9,0.516,36.0},{3.0,80.0,0.0,0.0,0.0,0.0,0.174,22.0},{4.0,129.0,60.0,12.0,231.0,27.5,0.527,31.0},{1.0,181.0,78.0,42.0,293.0,40.0,1.258,22.0},{13.0,152.0,90.0,33.0,29.0,26.8,0.731,43.0},{3.0,158.0,76.0,36.0,245.0,31.6,0.851,28.0},{4.0,146.0,85.0,27.0,100.0,28.9,0.189,27.0},{2.0,81.0,60.0,22.0,0.0,27.7,0.29,25.0},{2.0,129.0,74.0,26.0,205.0,33.2,0.591,25.0},{4.0,117.0,62.0,12.0,0.0,29.7,0.38,30.0},{4.0,144.0,82.0,32.0,0.0,38.5,0.554,37.0},{2.0,92.0,52.0,0.0,0.0,30.1,0.141,22.0},{1.0,79.0,60.0,42.0,48.0,43.5,0.678,23.0},{1.0,71.0,62.0,0.0,0.0,21.8,0.416,26.0},{1.0,114.0,66.0,36.0,200.0,38.1,0.289,21.0},{0.0,167.0,0.0,0.0,0.0,32.3,0.839,30.0},{10.0,125.0,70.0,26.0,115.0,31.1,0.205,41.0},{3.0,103.0,72.0,30.0,152.0,27.6,0.73,27.0},{7.0,195.0,70.0,33.0,145.0,25.1,0.163,55.0},{7.0,150.0,66.0,42.0,342.0,34.7,0.718,42.0},{1.0,130.0,70.0,13.0,105.0,25.9,0.472,22.0},{2.0,100.0,66.0,20.0,90.0,32.9,0.867,28.0},{11.0,103.0,68.0,40.0,0.0,46.2,0.126,42.0},{3.0,125.0,58.0,0.0,0.0,31.6,0.151,24.0},{3.0,173.0,84.0,33.0,474.0,35.7,0.258,22.0},{0.0,124.0,70.0,20.0,0.0,27.4,0.254,36.0},{1.0,97.0,70.0,40.0,0.0,38.1,0.218,30.0},{0.0,84.0,64.0,22.0,66.0,35.8,0.545,21.0},{8.0,118.0,72.0,19.0,0.0,23.1,1.476,46.0},{2.0,93.0,64.0,32.0,160.0,38.0,0.674,23.0},{7.0,100.0,0.0,0.0,0.0,30.0,0.484,32.0},{4.0,146.0,78.0,0.0,0.0,38.5,0.52,67.0},{5.0,99.0,54.0,28.0,83.0,34.0,0.499,30.0},{0.0,132.0,78.0,0.0,0.0,32.4,0.393,21.0},{8.0,126.0,74.0,38.0,75.0,25.9,0.162,39.0},{1.0,95.0,82.0,25.0,180.0,35.0,0.233,43.0},{0.0,84.0,82.0,31.0,125.0,38.2,0.233,23.0},{6.0,80.0,80.0,36.0,0.0,39.8,0.177,28.0},{0.0,93.0,60.0,25.0,92.0,28.7,0.532,22.0},{0.0,137.0,84.0,27.0,0.0,27.3,0.231,59.0},{7.0,150.0,78.0,29.0,126.0,35.2,0.692,54.0},{1.0,101.0,50.0,15.0,36.0,24.2,0.526,26.0},{4.0,134.0,72.0,0.0,0.0,23.8,0.277,60.0},{12.0,106.0,80.0,0.0,0.0,23.6,0.137,44.0},{1.0,119.0,54.0,13.0,50.0,22.3,0.205,24.0},{6.0,96.0,0.0,0.0,0.0,23.7,0.19,28.0},{1.0,111.0,62.0,13.0,182.0,24.0,0.138,23.0},{1.0,140.0,74.0,26.0,180.0,24.1,0.828,23.0},{3.0,122.0,78.0,0.0,0.0,23.0,0.254,40.0},{1.0,121.0,78.0,39.0,74.0,39.0,0.261,28.0},{1.0,116.0,78.0,29.0,180.0,36.1,0.496,25.0},{2.0,129.0,0.0,0.0,0.0,38.5,0.304,41.0},{6.0,115.0,60.0,39.0,0.0,33.7,0.245,40.0},{7.0,119.0,0.0,0.0,0.0,25.2,0.209,37.0},{0.0,98.0,82.0,15.0,84.0,25.2,0.299,22.0},{2.0,102.0,86.0,36.0,120.0,45.5,0.127,23.0},{1.0,107.0,68.0,19.0,0.0,26.5,0.165,24.0},{0.0,94.0,70.0,27.0,115.0,43.5,0.347,21.0},{4.0,125.0,70.0,18.0,122.0,28.9,1.144,45.0},{0.0,140.0,65.0,26.0,130.0,42.6,0.431,24.0},{9.0,89.0,62.0,0.0,0.0,22.5,0.142,33.0},{7.0,125.0,86.0,0.0,0.0,37.6,0.304,51.0},{1.0,112.0,80.0,45.0,132.0,34.8,0.217,24.0},{0.0,145.0,0.0,0.0,0.0,44.2,0.63,31.0},{3.0,88.0,58.0,11.0,54.0,24.8,0.267,22.0},{6.0,154.0,78.0,41.0,140.0,46.1,0.571,27.0},{7.0,102.0,74.0,40.0,105.0,37.2,0.204,45.0},{0.0,104.0,76.0,0.0,0.0,18.4,0.582,27.0},{8.0,110.0,76.0,0.0,0.0,27.8,0.237,58.0},{0.0,173.0,78.0,32.0,265.0,46.5,1.159,58.0},{5.0,78.0,48.0,0.0,0.0,33.7,0.654,25.0},{10.0,133.0,68.0,0.0,0.0,27.0,0.245,36.0},{4.0,131.0,68.0,21.0,166.0,33.1,0.16,28.0},{12.0,92.0,62.0,7.0,258.0,27.6,0.926,44.0},{2.0,142.0,82.0,18.0,64.0,24.7,0.761,21.0},{10.0,162.0,84.0,0.0,0.0,27.7,0.182,54.0},{6.0,85.0,78.0,0.0,0.0,31.2,0.382,42.0},{2.0,106.0,56.0,27.0,165.0,29.0,0.426,22.0},{3.0,115.0,66.0,39.0,140.0,38.1,0.15,28.0},{6.0,102.0,82.0,0.0,0.0,30.8,0.18,36.0},{3.0,191.0,68.0,15.0,130.0,30.9,0.299,34.0},{8.0,105.0,100.0,36.0,0.0,43.3,0.239,45.0},{2.0,96.0,68.0,13.0,49.0,21.1,0.647,26.0},{4.0,110.0,66.0,0.0,0.0,31.9,0.471,29.0},{13.0,76.0,60.0,0.0,0.0,32.8,0.18,41.0},{4.0,123.0,80.0,15.0,176.0,32.0,0.443,34.0},{8.0,107.0,80.0,0.0,0.0,24.6,0.856,34.0},{2.0,157.0,74.0,35.0,440.0,39.4,0.134,30.0},{1.0,143.0,86.0,30.0,330.0,30.1,0.892,23.0},{7.0,94.0,64.0,25.0,79.0,33.3,0.738,41.0},{2.0,94.0,68.0,18.0,76.0,26.0,0.561,21.0},{7.0,109.0,80.0,31.0,0.0,35.9,1.127,43.0},{1.0,88.0,78.0,29.0,76.0,32.0,0.365,29.0},{0.0,97.0,64.0,36.0,100.0,36.8,0.6,25.0},{0.0,131.0,66.0,40.0,0.0,34.3,0.196,22.0},{1.0,131.0,64.0,14.0,415.0,23.7,0.389,21.0},{0.0,123.0,88.0,37.0,0.0,35.2,0.197,29.0},{4.0,125.0,80.0,0.0,0.0,32.3,0.536,27.0},{1.0,115.0,70.0,30.0,96.0,34.6,0.529,32.0},{10.0,68.0,106.0,23.0,49.0,35.5,0.285,47.0},{7.0,114.0,66.0,0.0,0.0,32.8,0.258,42.0},{10.0,115.0,98.0,0.0,0.0,24.0,1.022,34.0},{5.0,86.0,68.0,28.0,71.0,30.2,0.364,24.0},{11.0,111.0,84.0,40.0,0.0,46.8,0.925,45.0},{3.0,142.0,80.0,15.0,0.0,32.4,0.2,63.0},{8.0,181.0,68.0,36.0,495.0,30.1,0.615,60.0},{7.0,187.0,68.0,39.0,304.0,37.7,0.254,41.0},{5.0,95.0,72.0,33.0,0.0,37.7,0.37,27.0},{0.0,127.0,80.0,37.0,210.0,36.3,0.804,23.0},{1.0,100.0,66.0,29.0,196.0,32.0,0.444,42.0},{6.0,93.0,50.0,30.0,64.0,28.7,0.356,23.0},{5.0,111.0,72.0,28.0,0.0,23.9,0.407,27.0},{10.0,168.0,74.0,0.0,0.0,38.0,0.537,34.0},{3.0,106.0,54.0,21.0,158.0,30.9,0.292,24.0},{0.0,74.0,52.0,10.0,36.0,27.8,0.269,22.0},{1.0,103.0,80.0,11.0,82.0,19.4,0.491,22.0},{2.0,158.0,90.0,0.0,0.0,31.6,0.805,66.0},{5.0,143.0,78.0,0.0,0.0,45.0,0.19,47.0},{1.0,116.0,70.0,28.0,0.0,27.4,0.204,21.0},{7.0,133.0,88.0,15.0,155.0,32.4,0.262,37.0},{0.0,135.0,68.0,42.0,250.0,42.3,0.365,24.0},{3.0,89.0,74.0,16.0,85.0,30.4,0.551,38.0},{3.0,150.0,76.0,0.0,0.0,21.0,0.207,37.0},{2.0,146.0,76.0,35.0,194.0,38.2,0.329,29.0},{6.0,80.0,66.0,30.0,0.0,26.2,0.313,41.0},{1.0,96.0,64.0,27.0,87.0,33.2,0.289,21.0},{7.0,106.0,92.0,18.0,0.0,22.7,0.235,48.0},{2.0,115.0,64.0,22.0,0.0,30.8,0.421,21.0},{10.0,129.0,62.0,36.0,0.0,41.2,0.441,38.0},{9.0,112.0,82.0,24.0,0.0,28.2,1.282,50.0},{2.0,146.0,70.0,38.0,360.0,28.0,0.337,29.0},{3.0,116.0,74.0,15.0,105.0,26.3,0.107,24.0},{0.0,141.0,84.0,26.0,0.0,32.4,0.433,22.0},{5.0,126.0,78.0,27.0,22.0,29.6,0.439,40.0},{3.0,128.0,78.0,0.0,0.0,21.1,0.268,55.0},{9.0,57.0,80.0,37.0,0.0,32.8,0.096,41.0},{2.0,100.0,68.0,25.0,71.0,38.5,0.324,26.0},{8.0,197.0,74.0,0.0,0.0,25.9,1.191,39.0},{10.0,101.0,76.0,48.0,180.0,32.9,0.171,63.0},{0.0,134.0,58.0,20.0,291.0,26.4,0.352,21.0},{8.0,194.0,80.0,0.0,0.0,26.1,0.551,67.0},{13.0,106.0,72.0,54.0,0.0,36.6,0.178,45.0},{9.0,184.0,85.0,15.0,0.0,30.0,1.213,49.0},{2.0,122.0,70.0,27.0,0.0,36.8,0.34,27.0},{1.0,143.0,74.0,22.0,61.0,26.2,0.256,21.0},{3.0,90.0,78.0,0.0,0.0,42.7,0.559,21.0},{11.0,136.0,84.0,35.0,130.0,28.3,0.26,42.0},{4.0,129.0,86.0,20.0,270.0,35.1,0.231,23.0},{7.0,160.0,54.0,32.0,175.0,30.5,0.588,39.0},{5.0,166.0,76.0,0.0,0.0,45.7,0.34,27.0},{1.0,88.0,30.0,42.0,99.0,55.0,0.496,26.0},{0.0,129.0,80.0,0.0,0.0,31.2,0.703,29.0},{2.0,105.0,58.0,40.0,94.0,34.9,0.225,25.0},{8.0,183.0,64.0,0.0,0.0,23.3,0.672,32.0},{2.0,82.0,52.0,22.0,115.0,28.5,1.699,25.0},{8.0,155.0,62.0,26.0,495.0,34.0,0.543,46.0},{12.0,151.0,70.0,40.0,271.0,41.8,0.742,38.0},{1.0,91.0,54.0,25.0,100.0,25.2,0.234,23.0},{0.0,101.0,76.0,0.0,0.0,35.7,0.198,26.0},{2.0,112.0,78.0,50.0,140.0,39.4,0.175,24.0},{1.0,91.0,64.0,24.0,0.0,29.2,0.192,21.0},{1.0,164.0,82.0,43.0,67.0,32.8,0.341,50.0},{9.0,152.0,78.0,34.0,171.0,34.2,0.893,33.0},{9.0,130.0,70.0,0.0,0.0,34.2,0.652,45.0},{2.0,122.0,60.0,18.0,106.0,29.8,0.717,22.0},{9.0,140.0,94.0,0.0,0.0,32.7,0.734,45.0},{1.0,95.0,74.0,21.0,73.0,25.9,0.673,36.0},{0.0,118.0,84.0,47.0,230.0,45.8,0.551,31.0},{0.0,108.0,68.0,20.0,0.0,27.3,0.787,32.0},{1.0,95.0,66.0,13.0,38.0,19.6,0.334,25.0},{2.0,90.0,60.0,0.0,0.0,23.5,0.191,25.0},{5.0,73.0,60.0,0.0,0.0,26.8,0.268,27.0},{5.0,77.0,82.0,41.0,42.0,35.8,0.156,35.0},{2.0,111.0,60.0,0.0,0.0,26.2,0.343,23.0},{5.0,147.0,75.0,0.0,0.0,29.9,0.434,28.0},{10.0,115.0,0.0,0.0,0.0,0.0,0.261,30.0},{5.0,139.0,64.0,35.0,140.0,28.6,0.411,26.0},{10.0,122.0,68.0,0.0,0.0,31.2,0.258,41.0},{1.0,144.0,82.0,46.0,180.0,46.1,0.335,46.0},{3.0,130.0,78.0,23.0,79.0,28.4,0.323,34.0},{5.0,130.0,82.0,0.0,0.0,39.1,0.956,37.0},{0.0,57.0,60.0,0.0,0.0,21.7,0.735,67.0},{3.0,141.0,0.0,0.0,0.0,30.0,0.761,27.0},{2.0,112.0,86.0,42.0,160.0,38.4,0.246,28.0},{9.0,156.0,86.0,28.0,155.0,34.3,1.189,42.0},{1.0,113.0,64.0,35.0,0.0,33.6,0.543,21.0},{2.0,94.0,76.0,18.0,66.0,31.6,0.649,23.0},{1.0,109.0,58.0,18.0,116.0,28.5,0.219,22.0},{0.0,165.0,90.0,33.0,680.0,52.3,0.427,23.0},{6.0,129.0,90.0,7.0,326.0,19.6,0.582,60.0},{2.0,99.0,70.0,16.0,44.0,20.4,0.235,27.0},{0.0,101.0,64.0,17.0,0.0,21.0,0.252,21.0},{4.0,142.0,86.0,0.0,0.0,44.0,0.645,22.0},{1.0,109.0,56.0,21.0,135.0,25.2,0.833,23.0},{0.0,121.0,66.0,30.0,165.0,34.3,0.203,33.0},{0.0,129.0,110.0,46.0,130.0,67.1,0.319,26.0},{7.0,107.0,74.0,0.0,0.0,29.6,0.254,31.0},{5.0,85.0,74.0,22.0,0.0,29.0,1.224,32.0},{1.0,111.0,94.0,0.0,0.0,32.8,0.265,45.0},{8.0,108.0,70.0,0.0,0.0,30.5,0.955,33.0},{8.0,109.0,76.0,39.0,114.0,27.9,0.64,31.0},{1.0,119.0,86.0,39.0,220.0,45.6,0.808,29.0},{2.0,89.0,90.0,30.0,0.0,33.5,0.292,42.0},{7.0,136.0,74.0,26.0,135.0,26.0,0.647,51.0},{2.0,117.0,90.0,19.0,71.0,25.2,0.313,21.0},{0.0,94.0,0.0,0.0,0.0,0.0,0.256,25.0},{5.0,122.0,86.0,0.0,0.0,34.7,0.29,33.0},{3.0,99.0,80.0,11.0,64.0,19.3,0.284,30.0},{4.0,97.0,60.0,23.0,0.0,28.2,0.443,22.0},{0.0,95.0,85.0,25.0,36.0,37.4,0.247,24.0},{1.0,157.0,72.0,21.0,168.0,25.6,0.123,24.0},{7.0,103.0,66.0,32.0,0.0,39.1,0.344,31.0},{1.0,143.0,84.0,23.0,310.0,42.4,1.076,22.0},{1.0,103.0,30.0,38.0,83.0,43.3,0.183,33.0},{2.0,122.0,76.0,27.0,200.0,35.9,0.483,26.0},{4.0,120.0,68.0,0.0,0.0,29.6,0.709,34.0},{0.0,107.0,62.0,30.0,74.0,36.6,0.757,25.0},{6.0,148.0,72.0,35.0,0.0,33.6,0.627,50.0},{2.0,108.0,52.0,26.0,63.0,32.5,0.318,22.0},{7.0,62.0,78.0,0.0,0.0,32.6,0.391,41.0},{11.0,143.0,94.0,33.0,146.0,36.6,0.254,51.0},{1.0,99.0,72.0,30.0,18.0,38.6,0.412,21.0},{3.0,74.0,68.0,28.0,45.0,29.7,0.293,23.0},{4.0,173.0,70.0,14.0,168.0,29.7,0.361,33.0},{4.0,99.0,76.0,15.0,51.0,23.2,0.223,21.0},{14.0,175.0,62.0,30.0,0.0,33.6,0.212,38.0},{0.0,124.0,56.0,13.0,105.0,21.8,0.452,21.0},{4.0,156.0,75.0,0.0,0.0,48.3,0.238,32.0},{2.0,197.0,70.0,45.0,543.0,30.5,0.158,53.0},{7.0,105.0,0.0,0.0,0.0,0.0,0.305,24.0},{4.0,90.0,88.0,47.0,54.0,37.7,0.362,29.0},{1.0,81.0,74.0,41.0,57.0,46.3,1.096,32.0},{2.0,123.0,48.0,32.0,165.0,42.1,0.52,26.0},{4.0,115.0,72.0,0.0,0.0,28.9,0.376,46.0},{5.0,137.0,108.0,0.0,0.0,48.8,0.227,37.0},{1.0,97.0,68.0,21.0,0.0,27.2,1.095,22.0},{7.0,159.0,64.0,0.0,0.0,27.4,0.294,40.0},{2.0,88.0,74.0,19.0,53.0,29.0,0.229,22.0},{1.0,125.0,50.0,40.0,167.0,33.3,0.962,28.0},{6.0,105.0,80.0,28.0,0.0,32.5,0.878,26.0},{2.0,197.0,70.0,99.0,0.0,34.7,0.575,62.0},{4.0,84.0,90.0,23.0,56.0,39.5,0.159,25.0},{6.0,134.0,80.0,37.0,370.0,46.2,0.238,46.0},{0.0,102.0,64.0,46.0,78.0,40.6,0.496,21.0},{11.0,85.0,74.0,0.0,0.0,30.1,0.3,35.0},{0.0,181.0,88.0,44.0,510.0,43.3,0.222,26.0},{0.0,161.0,50.0,0.0,0.0,21.9,0.254,65.0},{1.0,139.0,46.0,19.0,83.0,28.7,0.654,22.0},{3.0,112.0,74.0,30.0,0.0,31.6,0.197,25.0},{3.0,106.0,72.0,0.0,0.0,25.8,0.207,27.0},{10.0,94.0,72.0,18.0,0.0,23.1,0.595,56.0},{5.0,44.0,62.0,0.0,0.0,25.0,0.587,36.0},{10.0,108.0,66.0,0.0,0.0,32.4,0.272,42.0},{8.0,196.0,76.0,29.0,280.0,37.5,0.605,57.0},{5.0,155.0,84.0,44.0,545.0,38.7,0.619,34.0},{1.0,126.0,56.0,29.0,152.0,28.7,0.801,21.0},{5.0,104.0,74.0,0.0,0.0,28.8,0.153,48.0},{1.0,126.0,60.0,0.0,0.0,30.1,0.349,47.0},{5.0,162.0,104.0,0.0,0.0,37.7,0.151,52.0},{6.0,104.0,74.0,18.0,156.0,29.9,0.722,41.0},{1.0,84.0,64.0,23.0,115.0,36.9,0.471,28.0},{3.0,96.0,56.0,34.0,115.0,24.7,0.944,39.0},{1.0,124.0,60.0,32.0,0.0,35.8,0.514,21.0},{1.0,100.0,72.0,12.0,70.0,25.3,0.658,28.0},{3.0,82.0,70.0,0.0,0.0,21.1,0.389,25.0},{4.0,128.0,70.0,0.0,0.0,34.3,0.303,24.0},{6.0,102.0,90.0,39.0,0.0,35.7,0.674,28.0},{1.0,189.0,60.0,23.0,846.0,30.1,0.398,59.0},{3.0,129.0,92.0,49.0,155.0,36.4,0.968,32.0},{8.0,126.0,88.0,36.0,108.0,38.5,0.349,49.0},{1.0,97.0,66.0,15.0,140.0,23.2,0.487,22.0},{0.0,177.0,60.0,29.0,478.0,34.6,1.072,21.0},{3.0,111.0,58.0,31.0,44.0,29.5,0.43,22.0},{6.0,144.0,72.0,27.0,228.0,33.9,0.255,40.0},{4.0,90.0,0.0,0.0,0.0,28.0,0.61,31.0},{0.0,117.0,80.0,31.0,53.0,45.2,0.089,24.0},{2.0,175.0,88.0,0.0,0.0,22.9,0.326,22.0},{1.0,130.0,60.0,23.0,170.0,28.6,0.692,21.0},{5.0,187.0,76.0,27.0,207.0,43.6,1.034,53.0},{4.0,91.0,70.0,32.0,88.0,33.1,0.446,22.0},{1.0,128.0,82.0,17.0,183.0,27.5,0.115,22.0},{0.0,91.0,68.0,32.0,210.0,39.9,0.381,25.0},{3.0,84.0,68.0,30.0,106.0,31.9,0.591,25.0},{2.0,92.0,62.0,28.0,0.0,31.6,0.13,24.0},{2.0,88.0,58.0,26.0,16.0,28.4,0.766,22.0},{7.0,168.0,88.0,42.0,321.0,38.2,0.787,40.0},{3.0,116.0,0.0,0.0,0.0,23.5,0.187,23.0},{1.0,97.0,70.0,15.0,0.0,18.2,0.147,21.0},{4.0,145.0,82.0,18.0,0.0,32.5,0.235,70.0},{7.0,196.0,90.0,0.0,0.0,39.8,0.451,41.0},{9.0,120.0,72.0,22.0,56.0,20.8,0.733,48.0},{7.0,137.0,90.0,41.0,0.0,32.0,0.391,39.0},{2.0,90.0,70.0,17.0,0.0,27.3,0.085,22.0},{1.0,90.0,62.0,18.0,59.0,25.1,1.268,25.0},{6.0,98.0,58.0,33.0,190.0,34.0,0.43,43.0},{6.0,108.0,44.0,20.0,130.0,24.0,0.813,35.0},{9.0,164.0,84.0,21.0,0.0,30.8,0.831,32.0},{1.0,0.0,48.0,20.0,0.0,24.7,0.14,22.0},{1.0,136.0,74.0,50.0,204.0,37.4,0.399,24.0},{4.0,122.0,68.0,0.0,0.0,35.0,0.394,29.0},{6.0,87.0,80.0,0.0,0.0,23.2,0.084,32.0},{3.0,158.0,64.0,13.0,387.0,31.2,0.295,24.0},{5.0,115.0,76.0,0.0,0.0,31.2,0.343,44.0},{1.0,106.0,76.0,0.0,0.0,37.5,0.197,26.0},{1.0,122.0,64.0,32.0,156.0,35.1,0.692,30.0},{10.0,92.0,62.0,0.0,0.0,25.9,0.167,31.0},{6.0,195.0,70.0,0.0,0.0,30.9,0.328,31.0},{2.0,92.0,76.0,20.0,0.0,24.2,1.698,28.0},{5.0,112.0,66.0,0.0,0.0,37.8,0.261,41.0},{8.0,74.0,70.0,40.0,49.0,35.3,0.705,39.0},{0.0,120.0,74.0,18.0,63.0,30.5,0.285,26.0},{10.0,122.0,78.0,31.0,0.0,27.6,0.512,45.0},{0.0,78.0,88.0,29.0,40.0,36.9,0.434,21.0},{2.0,155.0,52.0,27.0,540.0,38.7,0.24,25.0},{8.0,84.0,74.0,31.0,0.0,38.3,0.457,39.0},{4.0,141.0,74.0,0.0,0.0,27.6,0.244,40.0},{3.0,99.0,62.0,19.0,74.0,21.8,0.279,26.0},{1.0,83.0,68.0,0.0,0.0,18.2,0.624,27.0},{10.0,161.0,68.0,23.0,132.0,25.5,0.326,47.0},{5.0,97.0,76.0,27.0,0.0,35.6,0.378,52.0},{4.0,99.0,68.0,38.0,0.0,32.8,0.145,33.0},{2.0,108.0,62.0,10.0,278.0,25.3,0.881,22.0},{7.0,179.0,95.0,31.0,0.0,34.2,0.164,60.0},{1.0,128.0,88.0,39.0,110.0,36.5,1.057,37.0},{0.0,137.0,40.0,35.0,168.0,43.1,2.288,33.0},{13.0,158.0,114.0,0.0,0.0,42.3,0.257,44.0},{0.0,180.0,78.0,63.0,14.0,59.4,2.42,25.0},{5.0,109.0,75.0,26.0,0.0,36.0,0.546,60.0},{6.0,103.0,72.0,32.0,190.0,37.7,0.324,55.0},{0.0,165.0,76.0,43.0,255.0,47.9,0.259,26.0},{10.0,179.0,70.0,0.0,0.0,35.1,0.2,37.0},{8.0,120.0,86.0,0.0,0.0,28.4,0.259,22.0},{7.0,97.0,76.0,32.0,91.0,40.9,0.871,32.0},{0.0,138.0,60.0,35.0,167.0,34.6,0.534,21.0},{0.0,117.0,0.0,0.0,0.0,33.8,0.932,44.0},{3.0,124.0,80.0,33.0,130.0,33.2,0.305,26.0},{0.0,91.0,80.0,0.0,0.0,32.4,0.601,27.0},{9.0,119.0,80.0,35.0,0.0,29.0,0.263,29.0},{6.0,154.0,74.0,32.0,193.0,29.3,0.839,39.0},{10.0,90.0,85.0,32.0,0.0,34.9,0.825,56.0},{13.0,153.0,88.0,37.0,140.0,40.6,1.174,39.0},{0.0,126.0,84.0,29.0,215.0,30.7,0.52,24.0},{5.0,158.0,84.0,41.0,210.0,39.4,0.395,29.0},{5.0,121.0,72.0,23.0,112.0,26.2,0.245,30.0},{5.0,116.0,74.0,29.0,0.0,32.3,0.66,35.0},{7.0,83.0,78.0,26.0,71.0,29.3,0.767,36.0},{10.0,148.0,84.0,48.0,237.0,37.6,1.001,51.0},{5.0,117.0,86.0,30.0,105.0,39.1,0.251,42.0},{0.0,141.0,0.0,0.0,0.0,42.4,0.205,29.0},{1.0,87.0,60.0,37.0,75.0,37.2,0.509,22.0},{4.0,85.0,58.0,22.0,49.0,27.8,0.306,28.0},{10.0,139.0,80.0,0.0,0.0,27.1,1.441,57.0},{1.0,99.0,58.0,10.0,0.0,25.4,0.551,21.0},{0.0,137.0,68.0,14.0,148.0,24.8,0.143,21.0},{1.0,181.0,64.0,30.0,180.0,34.1,0.328,38.0},{7.0,129.0,68.0,49.0,125.0,38.5,0.439,43.0},{1.0,172.0,68.0,49.0,579.0,42.4,0.702,28.0},{3.0,78.0,70.0,0.0,0.0,32.5,0.27,39.0},{6.0,194.0,78.0,0.0,0.0,23.5,0.129,59.0},{0.0,179.0,90.0,27.0,0.0,44.1,0.686,23.0},{8.0,112.0,72.0,0.0,0.0,23.6,0.84,58.0},{2.0,112.0,66.0,22.0,0.0,25.0,0.307,24.0},{6.0,134.0,70.0,23.0,130.0,35.4,0.542,29.0},{9.0,134.0,74.0,33.0,60.0,25.9,0.46,81.0},{1.0,89.0,66.0,23.0,94.0,28.1,0.167,21.0},{3.0,193.0,70.0,31.0,0.0,34.9,0.241,25.0},{13.0,129.0,0.0,30.0,0.0,39.9,0.569,44.0},{1.0,106.0,70.0,28.0,135.0,34.2,0.142,22.0},{5.0,168.0,64.0,0.0,0.0,32.9,0.135,41.0},{0.0,109.0,88.0,30.0,0.0,32.5,0.855,38.0},{2.0,99.0,0.0,0.0,0.0,22.2,0.108,23.0},{0.0,105.0,90.0,0.0,0.0,29.6,0.197,46.0},{8.0,133.0,72.0,0.0,0.0,32.9,0.27,39.0},{1.0,95.0,60.0,18.0,58.0,23.9,0.26,22.0},{2.0,56.0,56.0,28.0,45.0,24.2,0.332,22.0},{8.0,125.0,96.0,0.0,0.0,0.0,0.232,54.0},{1.0,73.0,50.0,10.0,0.0,23.0,0.248,21.0},{0.0,95.0,80.0,45.0,92.0,36.5,0.33,26.0},{9.0,112.0,82.0,32.0,175.0,34.2,0.26,36.0},{4.0,95.0,60.0,32.0,0.0,35.4,0.284,28.0},{12.0,84.0,72.0,31.0,0.0,29.7,0.297,46.0},{0.0,162.0,76.0,56.0,100.0,53.2,0.759,25.0},{0.0,99.0,0.0,0.0,0.0,25.0,0.253,22.0},{10.0,129.0,76.0,28.0,122.0,35.9,0.28,39.0},{11.0,127.0,106.0,0.0,0.0,39.0,0.19,51.0},{3.0,139.0,54.0,0.0,0.0,25.6,0.402,22.0},{3.0,163.0,70.0,18.0,105.0,31.6,0.268,28.0},{2.0,81.0,72.0,15.0,76.0,30.1,0.547,25.0},{0.0,107.0,60.0,25.0,0.0,26.4,0.133,23.0},{9.0,164.0,78.0,0.0,0.0,32.8,0.148,45.0},{5.0,117.0,92.0,0.0,0.0,34.1,0.337,38.0},{4.0,116.0,72.0,12.0,87.0,22.1,0.463,37.0},{0.0,104.0,64.0,23.0,116.0,27.8,0.454,23.0},{1.0,85.0,66.0,29.0,0.0,26.6,0.351,31.0},{3.0,80.0,82.0,31.0,70.0,34.2,1.292,27.0},{0.0,73.0,0.0,0.0,0.0,21.1,0.342,25.0},{8.0,188.0,78.0,0.0,0.0,47.9,0.137,43.0},{5.0,106.0,82.0,30.0,0.0,39.5,0.286,38.0},{8.0,65.0,72.0,23.0,0.0,32.0,0.6,42.0},{6.0,166.0,74.0,0.0,0.0,26.6,0.304,66.0},{2.0,90.0,68.0,42.0,0.0,38.2,0.503,27.0},{2.0,106.0,64.0,35.0,119.0,30.5,1.4,34.0},{9.0,122.0,56.0,0.0,0.0,33.3,1.114,33.0},{3.0,84.0,72.0,32.0,0.0,37.2,0.267,28.0},{12.0,140.0,82.0,43.0,325.0,39.2,0.528,58.0},{14.0,100.0,78.0,25.0,184.0,36.6,0.412,46.0},{1.0,163.0,72.0,0.0,0.0,39.0,1.222,33.0},{3.0,111.0,90.0,12.0,78.0,28.4,0.495,29.0},{4.0,123.0,62.0,0.0,0.0,32.0,0.226,35.0},{4.0,95.0,70.0,32.0,0.0,32.1,0.612,24.0},{4.0,117.0,64.0,27.0,120.0,33.2,0.23,24.0},{5.0,136.0,84.0,41.0,88.0,35.0,0.286,35.0},{1.0,111.0,86.0,19.0,0.0,30.1,0.143,23.0},{0.0,114.0,80.0,34.0,285.0,44.2,0.167,27.0},{2.0,74.0,0.0,0.0,0.0,0.0,0.102,22.0},{1.0,128.0,98.0,41.0,58.0,32.0,1.321,33.0},{1.0,77.0,56.0,30.0,56.0,33.3,1.251,24.0},{1.0,82.0,64.0,13.0,95.0,21.2,0.415,23.0},{8.0,100.0,74.0,40.0,215.0,39.4,0.661,43.0},{3.0,173.0,82.0,48.0,465.0,38.4,2.137,25.0},{2.0,101.0,58.0,17.0,265.0,24.2,0.614,23.0},{6.0,183.0,94.0,0.0,0.0,40.8,1.461,45.0},{7.0,136.0,90.0,0.0,0.0,29.9,0.21,50.0},{5.0,96.0,74.0,18.0,67.0,33.6,0.997,43.0},{3.0,96.0,78.0,39.0,0.0,37.3,0.238,40.0},{0.0,128.0,68.0,19.0,180.0,30.5,1.391,25.0},{1.0,97.0,64.0,19.0,82.0,18.2,0.299,21.0},{0.0,102.0,52.0,0.0,0.0,25.1,0.078,21.0},{2.0,84.0,0.0,0.0,0.0,0.0,0.304,21.0},{4.0,151.0,90.0,38.0,0.0,29.7,0.294,36.0},{12.0,140.0,85.0,33.0,0.0,37.4,0.244,41.0},{2.0,110.0,74.0,29.0,125.0,32.4,0.698,27.0},{2.0,129.0,84.0,0.0,0.0,28.0,0.284,27.0},{13.0,145.0,82.0,19.0,110.0,22.2,0.245,57.0},{4.0,103.0,60.0,33.0,192.0,24.0,0.966,33.0},{1.0,149.0,68.0,29.0,127.0,29.3,0.349,42.0},{0.0,113.0,76.0,0.0,0.0,33.3,0.278,23.0},{4.0,197.0,70.0,39.0,744.0,36.7,2.329,31.0},{3.0,121.0,52.0,0.0,0.0,36.0,0.127,25.0},{0.0,101.0,62.0,0.0,0.0,21.9,0.336,25.0},{0.0,67.0,76.0,0.0,0.0,45.3,0.194,46.0},{1.0,79.0,80.0,25.0,37.0,25.4,0.583,22.0},{3.0,61.0,82.0,28.0,0.0,34.4,0.243,46.0},{1.0,168.0,88.0,29.0,0.0,35.0,0.905,52.0},{1.0,88.0,62.0,24.0,44.0,29.9,0.422,23.0},{1.0,80.0,74.0,11.0,60.0,30.0,0.527,22.0},{3.0,148.0,66.0,25.0,0.0,32.5,0.256,22.0}};
	unsigned y_train[] =
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0};
	double x_test[][NB_FEATURES] =
	{{5.0,116.0,74.0,0.0,0.0,25.6,0.201,30.0},{3.0,78.0,50.0,32.0,88.0,31.0,0.248,26.0},{10.0,115.0,0.0,0.0,0.0,35.3,0.134,29.0},{4.0,110.0,92.0,0.0,0.0,37.6,0.191,30.0},{3.0,126.0,88.0,41.0,235.0,39.3,0.704,27.0},{8.0,99.0,84.0,0.0,0.0,35.4,0.388,50.0},{7.0,147.0,76.0,0.0,0.0,39.4,0.257,43.0},{6.0,92.0,92.0,0.0,0.0,19.9,0.188,28.0},{11.0,138.0,76.0,0.0,0.0,33.2,0.42,35.0},{9.0,102.0,76.0,37.0,0.0,32.9,0.665,46.0},{4.0,111.0,72.0,47.0,207.0,37.1,1.39,56.0},{7.0,133.0,84.0,0.0,0.0,40.2,0.696,37.0},{9.0,171.0,110.0,24.0,240.0,45.4,0.721,54.0},{0.0,180.0,66.0,39.0,0.0,42.0,1.893,25.0},{1.0,146.0,56.0,0.0,0.0,29.7,0.564,29.0},{2.0,71.0,70.0,27.0,0.0,28.0,0.586,22.0},{5.0,88.0,66.0,21.0,23.0,24.4,0.342,30.0},{8.0,176.0,90.0,34.0,300.0,33.7,0.467,58.0},{0.0,100.0,88.0,60.0,110.0,46.8,0.962,31.0},{0.0,105.0,64.0,41.0,142.0,41.5,0.173,22.0},{2.0,141.0,58.0,34.0,128.0,25.4,0.699,24.0},{5.0,99.0,74.0,27.0,0.0,29.0,0.203,32.0},{1.0,79.0,75.0,30.0,0.0,32.0,0.396,22.0},{0.0,131.0,0.0,0.0,0.0,43.2,0.27,26.0},{3.0,113.0,44.0,13.0,0.0,22.4,0.14,22.0},{0.0,101.0,65.0,28.0,0.0,24.6,0.237,22.0},{15.0,136.0,70.0,32.0,110.0,37.1,0.153,43.0},{7.0,81.0,78.0,40.0,48.0,46.7,0.261,42.0},{1.0,71.0,48.0,18.0,76.0,20.4,0.323,22.0},{1.0,122.0,90.0,51.0,220.0,49.7,0.325,31.0},{1.0,151.0,60.0,0.0,0.0,26.1,0.179,22.0},{0.0,125.0,96.0,0.0,0.0,22.5,0.262,21.0},{1.0,81.0,72.0,18.0,40.0,26.6,0.283,24.0},{2.0,85.0,65.0,0.0,0.0,39.6,0.93,27.0},{3.0,83.0,58.0,31.0,18.0,34.3,0.336,25.0},{1.0,89.0,76.0,34.0,37.0,31.2,0.192,23.0},{4.0,76.0,62.0,0.0,0.0,34.0,0.391,25.0},{4.0,146.0,92.0,0.0,0.0,31.2,0.539,61.0},{5.0,124.0,74.0,0.0,0.0,34.0,0.22,38.0},{6.0,111.0,64.0,39.0,0.0,34.2,0.26,24.0},{2.0,107.0,74.0,30.0,100.0,33.6,0.404,23.0},{5.0,132.0,80.0,0.0,0.0,26.8,0.186,69.0},{3.0,120.0,70.0,30.0,135.0,42.9,0.452,30.0},{1.0,118.0,58.0,36.0,94.0,33.3,0.261,23.0},{1.0,117.0,88.0,24.0,145.0,34.5,0.403,40.0},{0.0,105.0,84.0,0.0,0.0,27.9,0.741,62.0},{3.0,170.0,64.0,37.0,225.0,34.5,0.356,30.0},{5.0,105.0,72.0,29.0,325.0,36.9,0.159,28.0},{4.0,154.0,62.0,31.0,284.0,32.8,0.237,23.0},{5.0,147.0,78.0,0.0,0.0,33.7,0.218,65.0},{4.0,114.0,65.0,0.0,0.0,21.9,0.432,37.0},{7.0,152.0,88.0,44.0,0.0,50.0,0.337,36.0},{2.0,99.0,52.0,15.0,94.0,24.6,0.637,21.0},{17.0,163.0,72.0,41.0,114.0,40.9,0.817,47.0},{2.0,100.0,64.0,23.0,0.0,29.7,0.368,21.0},{0.0,131.0,88.0,0.0,0.0,31.6,0.743,32.0},{2.0,87.0,0.0,23.0,0.0,28.9,0.773,25.0},{2.0,75.0,64.0,24.0,55.0,29.7,0.37,33.0},{0.0,119.0,64.0,18.0,92.0,34.9,0.725,23.0},{1.0,0.0,74.0,20.0,23.0,27.7,0.299,21.0},{7.0,194.0,68.0,28.0,0.0,35.9,0.745,41.0},{5.0,139.0,80.0,35.0,160.0,31.6,0.361,25.0},{3.0,111.0,62.0,0.0,0.0,22.6,0.142,21.0},{9.0,123.0,70.0,44.0,94.0,33.1,0.374,40.0},{11.0,135.0,0.0,0.0,0.0,52.3,0.578,40.0},{8.0,85.0,55.0,20.0,0.0,24.4,0.136,42.0},{1.0,105.0,58.0,0.0,0.0,24.3,0.187,21.0},{4.0,109.0,64.0,44.0,99.0,34.8,0.905,26.0},{4.0,148.0,60.0,27.0,318.0,30.9,0.15,29.0},{0.0,113.0,80.0,16.0,0.0,31.0,0.874,21.0},{1.0,138.0,82.0,0.0,0.0,40.1,0.236,28.0},{7.0,184.0,84.0,33.0,0.0,35.5,0.355,41.0},{0.0,147.0,85.0,54.0,0.0,42.8,0.375,24.0},{6.0,125.0,68.0,30.0,120.0,30.0,0.464,32.0},{7.0,142.0,60.0,33.0,190.0,28.8,0.687,61.0},{1.0,100.0,66.0,15.0,56.0,23.6,0.666,26.0},{1.0,87.0,78.0,27.0,32.0,34.6,0.101,22.0},{3.0,162.0,52.0,38.0,0.0,37.2,0.652,24.0},{7.0,181.0,84.0,21.0,192.0,35.9,0.586,51.0},{9.0,124.0,70.0,33.0,402.0,35.4,0.282,34.0},{2.0,90.0,80.0,14.0,55.0,24.4,0.249,24.0},{0.0,86.0,68.0,32.0,0.0,35.8,0.238,25.0},{2.0,114.0,68.0,22.0,0.0,28.7,0.092,25.0},{1.0,193.0,50.0,16.0,375.0,25.9,0.655,24.0},{11.0,155.0,76.0,28.0,150.0,33.3,1.353,51.0},{0.0,138.0,0.0,0.0,0.0,36.3,0.933,25.0},{2.0,128.0,64.0,42.0,0.0,40.0,1.101,24.0},{2.0,146.0,0.0,0.0,0.0,27.5,0.24,28.0},{10.0,101.0,86.0,37.0,0.0,45.6,1.136,38.0},{7.0,106.0,60.0,24.0,0.0,26.5,0.296,29.0},{5.0,114.0,74.0,0.0,0.0,24.9,0.744,57.0},{0.0,146.0,70.0,0.0,0.0,37.9,0.334,28.0},{7.0,161.0,86.0,0.0,0.0,30.4,0.165,47.0},{2.0,108.0,80.0,0.0,0.0,27.0,0.259,52.0},{5.0,108.0,72.0,43.0,75.0,36.1,0.263,33.0},{2.0,128.0,78.0,37.0,182.0,43.3,1.224,31.0},{1.0,128.0,48.0,45.0,194.0,40.5,0.613,24.0},{6.0,151.0,62.0,31.0,120.0,35.5,0.692,28.0},{2.0,144.0,58.0,33.0,135.0,31.6,0.422,25.0},{5.0,115.0,98.0,0.0,0.0,52.9,0.209,28.0},{2.0,120.0,76.0,37.0,105.0,39.7,0.215,29.0},{2.0,124.0,68.0,28.0,205.0,32.9,0.875,30.0},{0.0,106.0,70.0,37.0,148.0,39.4,0.605,22.0},{2.0,155.0,74.0,17.0,96.0,26.6,0.433,27.0},{3.0,113.0,50.0,10.0,85.0,29.5,0.626,25.0},{2.0,112.0,68.0,22.0,94.0,34.1,0.315,26.0},{3.0,182.0,74.0,0.0,0.0,30.5,0.345,29.0},{2.0,112.0,75.0,32.0,0.0,35.7,0.148,21.0},{6.0,105.0,70.0,32.0,68.0,30.8,0.122,37.0},{2.0,87.0,58.0,16.0,52.0,32.7,0.166,25.0},{7.0,178.0,84.0,0.0,0.0,39.9,0.331,41.0},{1.0,0.0,68.0,35.0,0.0,32.0,0.389,22.0},{8.0,95.0,72.0,0.0,0.0,36.8,0.485,57.0},{5.0,0.0,80.0,32.0,0.0,41.0,0.346,37.0},{4.0,137.0,84.0,0.0,0.0,31.2,0.252,30.0},{12.0,88.0,74.0,40.0,54.0,35.3,0.378,48.0},{1.0,196.0,76.0,36.0,249.0,36.5,0.875,29.0},{5.0,189.0,64.0,33.0,325.0,31.2,0.583,29.0},{5.0,103.0,108.0,37.0,0.0,39.2,0.305,65.0},{4.0,147.0,74.0,25.0,293.0,34.9,0.385,30.0},{6.0,124.0,72.0,0.0,0.0,27.6,0.368,29.0},{3.0,81.0,86.0,16.0,66.0,27.5,0.306,22.0},{1.0,133.0,102.0,28.0,140.0,32.8,0.234,45.0},{0.0,118.0,64.0,23.0,89.0,0.0,1.731,21.0},{2.0,122.0,52.0,43.0,158.0,36.2,0.816,28.0},{0.0,93.0,100.0,39.0,72.0,43.4,1.021,35.0},{1.0,107.0,72.0,30.0,82.0,30.8,0.821,24.0},{0.0,105.0,68.0,22.0,0.0,20.0,0.236,22.0},{1.0,109.0,60.0,8.0,182.0,25.4,0.947,21.0},{1.0,125.0,70.0,24.0,110.0,24.3,0.221,25.0},{5.0,144.0,82.0,26.0,285.0,32.0,0.452,58.0},{4.0,158.0,78.0,0.0,0.0,32.9,0.803,31.0},{2.0,127.0,58.0,24.0,275.0,27.7,1.6,25.0},{4.0,95.0,64.0,0.0,0.0,32.0,0.161,31.0},{9.0,72.0,78.0,25.0,0.0,31.6,0.28,38.0},{1.0,119.0,88.0,41.0,170.0,45.3,0.507,26.0},{0.0,135.0,94.0,46.0,145.0,40.6,0.284,26.0},{2.0,139.0,75.0,0.0,0.0,25.6,0.167,29.0},{1.0,90.0,68.0,8.0,0.0,24.5,1.138,36.0},{2.0,83.0,66.0,23.0,50.0,32.2,0.497,22.0},{0.0,104.0,64.0,37.0,64.0,33.6,0.51,22.0},{2.0,134.0,70.0,0.0,0.0,28.9,0.542,23.0},{2.0,119.0,0.0,0.0,0.0,19.6,0.832,72.0},{2.0,100.0,54.0,28.0,105.0,37.8,0.498,24.0},{1.0,135.0,54.0,0.0,0.0,26.7,0.687,62.0},{5.0,88.0,78.0,30.0,0.0,27.6,0.258,37.0},{8.0,120.0,0.0,0.0,0.0,30.0,0.183,38.0},{1.0,144.0,82.0,40.0,0.0,41.3,0.607,28.0},{0.0,137.0,70.0,38.0,0.0,33.2,0.17,22.0},{4.0,114.0,64.0,0.0,0.0,28.9,0.126,24.0},{2.0,105.0,80.0,45.0,191.0,33.7,0.711,29.0},{7.0,114.0,76.0,17.0,110.0,23.8,0.466,31.0},{4.0,132.0,86.0,31.0,0.0,28.0,0.419,63.0},{3.0,158.0,70.0,30.0,328.0,35.5,0.344,35.0},{4.0,99.0,72.0,17.0,0.0,25.6,0.294,28.0},{2.0,83.0,65.0,28.0,66.0,36.8,0.629,24.0},{5.0,110.0,68.0,0.0,0.0,26.0,0.292,30.0},{6.0,0.0,68.0,41.0,0.0,39.0,0.727,41.0},{10.0,75.0,82.0,0.0,0.0,33.3,0.263,38.0},{0.0,180.0,90.0,26.0,90.0,36.5,0.314,35.0},{2.0,84.0,50.0,23.0,76.0,30.4,0.968,21.0},{0.0,139.0,62.0,17.0,210.0,22.1,0.207,21.0},{9.0,91.0,68.0,0.0,0.0,24.2,0.2,58.0},{2.0,91.0,62.0,0.0,0.0,27.3,0.525,22.0},{3.0,99.0,54.0,19.0,86.0,25.6,0.154,24.0},{9.0,145.0,88.0,34.0,165.0,30.3,0.771,53.0},{2.0,68.0,70.0,32.0,66.0,25.0,0.187,25.0},{6.0,114.0,0.0,0.0,0.0,0.0,0.189,26.0},{3.0,87.0,60.0,18.0,0.0,21.8,0.444,21.0},{0.0,117.0,66.0,31.0,188.0,30.8,0.493,22.0},{0.0,111.0,65.0,0.0,0.0,24.6,0.66,31.0},{0.0,107.0,76.0,0.0,0.0,45.3,0.686,24.0},{6.0,91.0,0.0,0.0,0.0,29.8,0.501,31.0},{4.0,132.0,0.0,0.0,0.0,32.9,0.302,23.0},{3.0,128.0,72.0,25.0,190.0,32.4,0.549,27.0},{8.0,186.0,90.0,35.0,225.0,34.5,0.423,37.0},{4.0,189.0,110.0,31.0,0.0,28.5,0.68,37.0},{6.0,114.0,88.0,0.0,0.0,27.8,0.247,66.0},{7.0,124.0,70.0,33.0,215.0,25.5,0.161,37.0},{6.0,125.0,76.0,0.0,0.0,33.8,0.121,54.0},{0.0,198.0,66.0,32.0,274.0,41.3,0.502,28.0},{1.0,87.0,68.0,34.0,77.0,37.6,0.401,24.0},{6.0,99.0,60.0,19.0,54.0,26.9,0.497,32.0},{2.0,95.0,54.0,14.0,88.0,26.1,0.748,22.0},{6.0,92.0,62.0,32.0,126.0,32.0,0.085,46.0},{4.0,154.0,72.0,29.0,126.0,31.3,0.338,37.0},{2.0,130.0,96.0,0.0,0.0,22.6,0.268,21.0},{2.0,98.0,60.0,17.0,120.0,34.7,0.198,22.0},{1.0,119.0,44.0,47.0,63.0,35.5,0.28,25.0},{2.0,118.0,80.0,0.0,0.0,42.9,0.693,21.0},{0.0,151.0,90.0,46.0,0.0,42.1,0.371,21.0},{12.0,121.0,78.0,17.0,0.0,26.5,0.259,62.0},{8.0,100.0,76.0,0.0,0.0,38.7,0.19,42.0},{8.0,124.0,76.0,24.0,600.0,28.7,0.687,52.0},{8.0,143.0,66.0,0.0,0.0,34.9,0.129,41.0},{6.0,103.0,66.0,0.0,0.0,24.3,0.249,29.0},{3.0,176.0,86.0,27.0,156.0,33.3,1.154,52.0},{3.0,132.0,80.0,0.0,0.0,34.4,0.402,44.0},{6.0,123.0,72.0,45.0,230.0,33.6,0.733,34.0},{0.0,188.0,82.0,14.0,185.0,32.0,0.682,22.0},{1.0,89.0,24.0,19.0,25.0,27.8,0.559,21.0},{1.0,108.0,88.0,19.0,0.0,27.1,0.4,24.0},{4.0,183.0,0.0,0.0,0.0,28.4,0.212,36.0},{1.0,92.0,62.0,25.0,41.0,19.5,0.482,25.0},{0.0,152.0,82.0,39.0,272.0,41.5,0.27,27.0},{11.0,138.0,74.0,26.0,144.0,36.1,0.557,50.0},{2.0,68.0,62.0,13.0,15.0,20.1,0.257,23.0},{0.0,119.0,0.0,0.0,0.0,32.4,0.141,24.0},{0.0,125.0,68.0,0.0,0.0,24.7,0.206,21.0},{5.0,128.0,80.0,0.0,0.0,34.6,0.144,45.0},{4.0,94.0,65.0,22.0,0.0,24.7,0.148,21.0},{7.0,114.0,64.0,0.0,0.0,27.4,0.732,34.0},{0.0,102.0,78.0,40.0,90.0,34.5,0.238,24.0},{13.0,104.0,72.0,0.0,0.0,31.2,0.465,38.0},{0.0,102.0,86.0,17.0,105.0,29.3,0.695,27.0},{6.0,147.0,80.0,0.0,0.0,29.5,0.178,50.0},{1.0,167.0,74.0,17.0,144.0,23.4,0.447,33.0},{0.0,179.0,50.0,36.0,159.0,37.8,0.455,22.0},{1.0,117.0,60.0,23.0,106.0,33.8,0.466,27.0},{5.0,123.0,74.0,40.0,77.0,34.1,0.269,28.0},{2.0,120.0,54.0,0.0,0.0,26.8,0.455,27.0},{1.0,199.0,76.0,43.0,0.0,42.9,1.394,22.0},{8.0,167.0,106.0,46.0,231.0,37.6,0.165,43.0},{10.0,111.0,70.0,27.0,0.0,27.5,0.141,40.0},{8.0,91.0,82.0,0.0,0.0,35.6,0.587,68.0},{0.0,93.0,60.0,0.0,0.0,35.3,0.263,25.0},{0.0,162.0,76.0,36.0,0.0,49.6,0.364,26.0},{5.0,136.0,82.0,0.0,0.0,0.0,0.64,69.0},{1.0,107.0,50.0,19.0,0.0,28.3,0.181,29.0},{2.0,121.0,70.0,32.0,95.0,39.1,0.886,23.0},{7.0,142.0,90.0,24.0,480.0,30.4,0.128,43.0},{4.0,127.0,88.0,11.0,155.0,34.5,0.598,28.0},{4.0,118.0,70.0,0.0,0.0,44.5,0.904,26.0},{6.0,125.0,78.0,31.0,0.0,27.6,0.565,49.0},{4.0,110.0,76.0,20.0,100.0,28.4,0.118,27.0},{2.0,127.0,46.0,21.0,335.0,34.4,0.176,22.0},{3.0,102.0,74.0,0.0,0.0,29.5,0.121,32.0},{7.0,187.0,50.0,33.0,392.0,33.9,0.826,34.0},{3.0,173.0,78.0,39.0,185.0,33.8,0.97,31.0},{1.0,108.0,60.0,46.0,178.0,35.5,0.415,24.0},{4.0,83.0,86.0,19.0,0.0,29.3,0.317,34.0},{4.0,112.0,78.0,40.0,0.0,39.4,0.236,38.0},{2.0,174.0,88.0,37.0,120.0,44.5,0.646,24.0},{0.0,126.0,86.0,27.0,120.0,27.4,0.515,21.0},{2.0,99.0,60.0,17.0,160.0,36.6,0.453,21.0},{3.0,102.0,44.0,20.0,94.0,30.8,0.4,26.0},{12.0,100.0,84.0,33.0,105.0,30.0,0.488,46.0},{3.0,187.0,70.0,22.0,200.0,36.4,0.408,36.0},{6.0,162.0,62.0,0.0,0.0,24.3,0.178,50.0},{3.0,108.0,62.0,24.0,0.0,26.0,0.223,25.0},{8.0,154.0,78.0,32.0,0.0,32.4,0.443,45.0},{0.0,123.0,72.0,0.0,0.0,36.3,0.258,52.0},{6.0,190.0,92.0,0.0,0.0,35.5,0.278,66.0},{9.0,170.0,74.0,31.0,0.0,44.0,0.403,43.0}};
	unsigned y_test[] =
	{0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1};	
	
	double x_row[NB_FEATURES];
	double means[NB_CLASSES*NB_FEATURES], variances[NB_CLASSES*NB_FEATURES];
	double priors[NB_CLASSES];
	double probabilities[NB_CLASSES*NB_FEATURES];
	unsigned temp_predictions[TEST_SIZE] = {0};
	
	unsigned predictions[TEST_SIZE];	

	summarizeByClass(x_train,y_train,means,variances, priors);
	
	printf("priors\n");
	print_tab_d(priors, NB_CLASSES);
	printf("Means (main)\n");
	print_tab_d(means,NB_CLASSES*NB_FEATURES);
	printf("Variances (main) \n");	
	print_tab_d(variances,NB_CLASSES*NB_FEATURES);

	printf("Running DUT\n");
	getPredictions(means, variances, priors, x_test, predictions);
	print_tab(predictions,TEST_SIZE);
	correct = 0;
	for (i = 0; i < TEST_SIZE; i++)
		if (predictions[i] == y_test[i])
			correct += 1;
	printf("Accuracy is %lf%%\n", (correct/(float)TEST_SIZE)*100);
	
  	if ((fp = fopen("E:\\projets\\mobilite\\C\\NB\\golden_output", "r")) == NULL){
    	printf("fopen source-file");
    	return 1;
  	}

 	count = 0;
 	count_errors = 0;
 	for (i = 0; i < TEST_SIZE; i++){
 		fscanf(fp, "%d", &golden_label);
 		if ( (golden_label != predictions[i]))
    		count_errors+=1,printf("HW label %u does not match SW computed label %d\n", predictions[i], golden_label);
 	}

  	fclose(fp);

  	if (count_errors == 0)		
		printf("Test Passed Successfully\n");
	else
		printf("Test Failed\n");

	return count_errors;
}

