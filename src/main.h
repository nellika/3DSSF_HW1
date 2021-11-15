#pragma once

void StereoEstimation_Naive(const int& window_size, const int& dmin, int height,
                            int width, const cv::Mat& image1,
                            const cv::Mat& image2, cv::Mat& naive_disparities,
                            const double& scale, const int& parallel);

void StereoEstimation_Naive_Normalize(const int& window_size, const int& dmin,
                                      int height, int width,
                                      const cv::Mat& image1,
                                      const cv::Mat& image2,
                                      cv::Mat& naive_disparities,
                                      const double& scale);

void StereoEstimation_Naive_Boxfilter(const int& window_size, const int& dmin,
                                      int height, int width,
                                      const cv::Mat& image1,
                                      const cv::Mat& image2,
                                      cv::Mat& naive_disparities,
                                      const double& scale);

void StereoEstimation_Dynamic(const int& window_size, const int& dmin,
                              int height, int width, const cv::Mat& image1,
                              const cv::Mat& image2, cv::Mat& dynamic_disparities,
                              const double& scale, const int& lambda);

void Disparity2PointCloud(const std::string& output_file, int height, int width,
                          cv::Mat& disparities, const int& window_size,
                          const int& dmin, const double& baseline,
                          const double& focal_length);
