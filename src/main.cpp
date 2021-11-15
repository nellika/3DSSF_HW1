#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "main.h"

double calc_mse(cv::Mat img1, cv::Mat img2) {
  const auto width = img1.cols;
  const auto height = img1.rows;

  double sum = 0.0;
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      double diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
      sum += diff * diff;
    }
  }

  double mse = sum / (width * height);
  return mse;
}

double get_rmse(cv::Mat img1, cv::Mat img2) {
  double mse = calc_mse(img1, img2);
  return sqrt(mse);
}

double get_psnr(cv::Mat img1, cv::Mat img2) {
  int max_px = 255;
  double mse = calc_mse(img1, img2);

  return (10 * log10((max_px * max_px) / mse));
}

int main(int argc, char** argv) {
  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 1247;
  const double baseline = 213;

  // stereo estimation parameters
  const int dmin = 67;
  int window_size = 3;
  const double weight = 500;
  const double scale = 1;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " IMAGE1 IMAGE2 OUTPUT_FILE WINDOW_SIZE TUNING[0/1]"
              << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  std::string disp_name = "../data/" + std::string(argv[1]) + "_disp1.png";
  cv::Mat disp1 = cv::imread(disp_name, cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[3];
  window_size = atoi(argv[4]);
  const int tuning = atoi(argv[5]);

  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "occlusion weights = " << weight << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "scaling of disparity images to show = " << scale << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  std::string approach;
  int parallel = 1;
  int norm = 0;
  int boxfilter = 0;

  if (tuning) {
    std::cout << "Naive or Dynamic? [naive/dynamic] ";
    std::cin >> approach;
    if (approach == "naive") {
      std::cout << "With normalization? [0/1] ";
      std::cin >> norm;

      if (!norm) {
        std::cout << "With boxfilter? [0/1] ";
        std::cin >> boxfilter;
      }

      if (!norm && !boxfilter) {
        std::cout << "Parallel? [0/1] ";
        std::cin >> parallel;
      }

      std::cout << "Method: " << approach << " with norm = " << norm
                << ", boxfilter = " << boxfilter << std::endl;
    } else if (approach != "dynamic") {
      std::cout << "Invalid approach, continue with naive..." << std::endl;
      approach = "naive";
    }

  } else {
    std::cout << "Default setup: naive, parallel" << std::endl;
    approach = "naive";
  }

  ////////////////////
  // Reconstruction //
  ////////////////////

  // Naive disparity image
  // cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width -
  // window_size, CV_8UC1);
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat dynamic_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  if (approach == "naive") {
    if (norm) {
      StereoEstimation_Naive_Normalize(window_size, dmin, height, width, image1,
                                     image2, naive_disparities, scale);
    } else if (boxfilter) {
      StereoEstimation_Naive_Boxfilter(window_size, dmin, height, width, image1,
                                     image2, naive_disparities, scale);
    } else {
      StereoEstimation_Naive(window_size, dmin, height, width, image1, image2,
                            naive_disparities, scale, parallel);
    }
    std::stringstream out1;
    out1 << "out/" << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);
  }

  if (approach == "dynamic") {
    int lambda = 18;
    StereoEstimation_Dynamic(window_size, dmin, height, width, image1, image2,
                             dynamic_disparities, scale, lambda);

    std::stringstream out_dynamic;
    out_dynamic << "out/" << output_file << "_dynamic.png";
    cv::imwrite(out_dynamic.str(), dynamic_disparities);
  }

  ////////////
  // Output //
  ////////////

  // reconstruction
  Disparity2PointCloud(output_file, height, width, disp1, window_size, dmin,
                       baseline, focal_length);

  // save / display images

  // cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  // cv::imshow("Naive", naive_disparities);

  // cv::namedWindow("Dynamic", cv::WINDOW_AUTOSIZE);
  // cv::imshow("Dynamic", dynamic_disparities);

  cv::waitKey(0);

  return 0;
}

void StereoEstimation_Naive(const int& window_size, const int& dmin, int height,
                            int width, const cv::Mat& image1,
                            const cv::Mat& image2, cv::Mat& naive_disparities,
                            const double& scale, const int& parallel) {
  int half_window_size = window_size / 2;

  int progress = 0;  // shared memory

  auto t_begin = std::chrono::high_resolution_clock::now();
// OpenMP
#pragma omp parallel for if (parallel)
  for (int i = half_window_size; i < height - half_window_size; ++i) {
#pragma omp critical
    {
      std::cout << "Calculating disparities for the naive approach... "
                << std::ceil(((progress - half_window_size + 1) /
                              static_cast<double>(height - window_size + 1)) *
                             100)
                << "%\r" << std::flush;
      ++progress;
    }

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      for (int d = -j + half_window_size; d < width - j - half_window_size;
           ++d) {
        int ssd = 0;

        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            int i_left = image1.at<uchar>(i + u, j + v);
            int i_right = image2.at<uchar>(i + u, j + v + d);

            int spectral_distance = i_left - i_right;
            ssd += spectral_distance * spectral_distance;
          }
        }

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) =
          std::abs(disparity) * scale;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      t_end - t_begin);

  std::cout << "Calculating disparities for the naive approach... Done.\r"
            << std::flush;
  std::cout << std::endl;
  std::cout << "Computation took " << duration.count() << " seconds."
            << std::endl;
}

void StereoEstimation_Naive_Normalize(const int& window_size, const int& dmin,
                                      int height, int width,
                                      const cv::Mat& image1,
                                      const cv::Mat& image2,
                                      cv::Mat& naive_disparities,
                                      const double& scale) {
  int half_window_size = window_size / 2;
  int progress = 0;  // shared memory

  auto t_begin = std::chrono::high_resolution_clock::now();
// OpenMP
#pragma omp parallel for
  for (int i = half_window_size; i < height - half_window_size; ++i) {
#pragma omp critical
    {
      std::cout << "Calculating disparities with normalization approach... "
                << std::ceil(((progress - half_window_size + 1) /
                              static_cast<double>(height - window_size + 1)) *
                             100)
                << "%\r" << std::flush;
      ++progress;
    }

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      int mean_left = 0;
      for (int u = -half_window_size; u <= half_window_size; ++u) {
        for (int v = -half_window_size; v <= half_window_size; ++v) {
          int i_left = image1.at<uchar>(i + u, j + v);

          mean_left += i_left;
        }
      }

      mean_left /= window_size * window_size;

      for (int d = -j + half_window_size; d < width - j - half_window_size;
           ++d) {
        int ssd = 0;

        int mean_right = 0;
        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            int i_right = image2.at<uchar>(i + u, j + v + d);

            mean_right += i_right;
          }
        }

        mean_right /= window_size * window_size;

        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            int i_left = image1.at<uchar>(i + u, j + v) - mean_left;
            int i_right = image2.at<uchar>(i + u, j + v + d) - mean_right;

            int spectral_distance = i_left - i_right;
            ssd += spectral_distance * spectral_distance;
          }
        }

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) =
          std::abs(disparity) * scale;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      t_end - t_begin);

  std::cout << "Calculating disparities with normalization... Done.\r"
            << std::flush;
  std::cout << std::endl;
  std::cout << "Computation took " << duration.count() << " seconds."
            << std::endl;
}

void StereoEstimation_Naive_Boxfilter(const int& window_size, const int& dmin,
                                      int height, int width,
                                      const cv::Mat& image1,
                                      const cv::Mat& image2,
                                      cv::Mat& naive_disparities,
                                      const double& scale) {
  int half_window_size = window_size / 2;
  int progress = 0;  // shared memory

  auto t_begin = std::chrono::high_resolution_clock::now();

  cv::Mat mean_image1 = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat mean_image2 = cv::Mat::zeros(height, width, CV_8UC1);
  cv::boxFilter(image1, mean_image1, -1, cv::Size(window_size, window_size));
  cv::boxFilter(image2, mean_image2, -1, cv::Size(window_size, window_size));

// OpenMP
#pragma omp parallel for
  for (int i = half_window_size; i < height - half_window_size; ++i) {
#pragma omp critical
    {
      std::cout << "Calculating disparities with boxfilter..."
                << std::ceil(((progress - half_window_size + 1) /
                              static_cast<double>(height - window_size + 1)) *
                             100)
                << "%\r" << std::flush;
      ++progress;
    }

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      for (int d = -j + half_window_size; d < width - j - half_window_size;
           ++d) {
        int ssd = 0;

        for (int u = -half_window_size; u <= half_window_size; ++u) {
          for (int v = -half_window_size; v <= half_window_size; ++v) {
            int i_left = image1.at<uchar>(i + u, j + v) -
                         mean_image1.at<uchar>(i + u, j + v);
            int i_right = image2.at<uchar>(i + u, j + v + d) -
                          mean_image2.at<uchar>(i + u, j + v + d);

            int spectral_distance = i_left - i_right;
            ssd += spectral_distance * spectral_distance;
          }
        }

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) =
          std::abs(disparity) * scale;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      t_end - t_begin);

  std::cout << "Calculating disparities with boxfilter... Done.\r"
            << std::flush;
  std::cout << std::endl;
  std::cout << "Computation took " << duration.count() << " seconds."
            << std::endl;
}

void StereoEstimation_Dynamic(const int& window_size, const int& dmin,
                              int height, int width, const cv::Mat& image1,
                              const cv::Mat& image2,
                              cv::Mat& dynamic_disparities, const double& scale,
                              const int& lambda) {
  auto t_begin = std::chrono::high_resolution_clock::now();

  cv::Mat C = cv::Mat::zeros(width, width, CV_32SC1);  // costs
  cv::Mat M = cv::Mat::zeros(width, width, CV_32SC1);  // precceding nodes

  for (int y = 0; y < height; y++) {
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
      C.at<int>(i, 0) = i * lambda;
      C.at<int>(0, i) = i * lambda;
    }
#pragma omp parallel for
    for (int i = 1; i < width; i++) {
      for (int j = 1; j < width; j++) {
        int px_1 = image1.at<uchar>(y, i);
        int px_2 = image2.at<uchar>(y, j);
        int px_diff = px_1 - px_2;
        int min_match = C.at<int>(i - 1, j - 1) + (px_diff * px_diff);
        int min_left = C.at<int>(i - 1, j) + lambda;
        int min_right = C.at<int>(i, j - 1) + lambda;
        int min = std::min({min_match, min_left, min_right});
        C.at<int>(i, j) = min;
        if (min == min_match) {
          M.at<int>(i, j) = 0;
        } else if (min == min_left) {
          M.at<int>(i, j) = 1;
        } else if (min == min_right) {
          M.at<int>(i, j) = 2;
        }
      }
    }

    // sink to source
    int x = width - 1;
    int z = width - 1;
    while ((x > 0) && (z > 0)) {
      switch (M.at<int>(x, z)) {
        case 0:
          dynamic_disparities.at<uchar>(y, x) = unsigned(std::abs(x - z));
          x--;
          z--;
          break;
        case 1:
          x--;
          break;
        default:
          z--;
          break;
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      t_end - t_begin);
  std::cout << "Calculating disparities for the dynamic programming "
               "approach... Done.\r"
            << std::flush;
  std::cout << std::endl;
  std::cout << "Computation took " << duration.count() << " seconds."
            << std::endl;
}

void Disparity2PointCloud(const std::string& output_file, int height, int width,
                          cv::Mat& disparities, const int& window_size,
                          const int& dmin, const double& baseline,
                          const double& focal_length) {
  std::stringstream out3d;
  out3d << "out/" << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... "
              << std::ceil(
                     ((i) / static_cast<double>(height - window_size + 1)) *
                     100)
              << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      int disparity = disparities.at<uchar>(i, j);
      if (disparity == 0) continue;

      double Z = baseline * focal_length / (disparity + 200);
      double X = (i - width / 2) * Z / focal_length;
      double Y = (j - height / 2) * Z / focal_length;
      outfile << X << " " << Y << " " << Z << std::endl;
    }
  }

  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r"
            << std::flush;
  std::cout << std::endl;
}

