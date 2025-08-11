/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <cmath>

#include <tf/tf.h>
#include "color.h"

using namespace std;
using namespace cv;
using namespace pcl;

#define TARGET_NUM_CIRCLES 4
#define DEBUG 1
#define GEOMETRY_TOLERANCE 0.06

// namespace CommonLiDAR 
// {
//   struct EIGEN_ALIGN16 Point 
//   {
//     PCL_ADD_POINT4D;     // quad-word XYZ
//     float intensity;     ///< laser intensity reading
//     std::uint16_t ring;  ///< laser ring number
//     float range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
//   };
  
//   void addRange(pcl::PointCloud<CommonLiDAR::Point> &pc) 
//   {
//     for (pcl::PointCloud<Point>::iterator pt = pc.points.begin();
//          pt < pc.points.end(); pt++) {
//       pt->range = sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);
//     }
//   }
  
//   vector<vector<Point *>> getRings(pcl::PointCloud<CommonLiDAR::Point> &pc,
//                                    int rings_count) 
//   {
//     vector<vector<Point *>> rings(rings_count);
//     for (pcl::PointCloud<Point>::iterator pt = pc.points.begin();
//          pt < pc.points.end(); pt++) {
//       rings[pt->ring].push_back(&(*pt));
//     }
//     return rings;
//   }
// }  // namespace Ouster
  
// POINT_CLOUD_REGISTER_POINT_STRUCT(CommonLiDAR::Point,
//                                   (float, x, x)(float, y, y)(float, z, z)(
//                                       float, intensity,
//                                       intensity)(std::uint16_t, ring,
//                                                   ring)(float, range, range));

// 参数结构体
struct Params {
  double x_min, x_max, y_min, y_max, z_min, z_max;
  double fx, fy, cx, cy, k1, k2, p1, p2;
  double marker_size, delta_width_qr_center, delta_height_qr_center;
  double delta_width_circles, delta_height_circles, circle_radius;
  int min_detected_markers;
  string image_path;
  string bag_path;
  string lidar_topic;
  string output_path;
  string midresult_path;
};

// 读取参数
Params loadParameters(ros::NodeHandle &nh) {
  Params params;
  nh.param("fx", params.fx, 1215.31801774424);
  nh.param("fy", params.fy, 1214.72961288138);
  nh.param("cx", params.cx, 1047.86571859677);
  nh.param("cy", params.cy, 745.068353101898);
  nh.param("k1", params.k1, -0.33574781188503);
  nh.param("k2", params.k2, 0.10996870793601);
  nh.param("p1", params.p1, 0.000157303079833973);
  nh.param("p2", params.p2, 0.000544930726278493);
  nh.param("marker_size", params.marker_size, 0.2);
  nh.param("delta_width_qr_center", params.delta_width_qr_center, 0.55);
  nh.param("delta_height_qr_center", params.delta_height_qr_center, 0.35);
  nh.param("delta_width_circles", params.delta_width_circles, 0.5);
  nh.param("delta_height_circles", params.delta_height_circles, 0.4);
  nh.param("min_detected_markers", params.min_detected_markers, 3);
  nh.param("circle_radius", params.circle_radius, 0.12);
  nh.param("image_path", params.image_path, string("/home/chunran/calib_ws/src/fast_calib/data/image.png"));
  nh.param("bag_path", params.bag_path, string("/home/chunran/calib_ws/src/fast_calib/data/input.bag"));
  nh.param("lidar_topic", params.lidar_topic, string("/livox/lidar"));
  nh.param("output_path", params.output_path, string("/home/chunran/calib_ws/src/fast_calib/output"));
  nh.param("midresult_path", params.midresult_path, string("/home/chunran/calib_ws/src/fast_calib/midresult"));
  nh.param("x_min", params.x_min, 1.5);
  nh.param("x_max", params.x_max, 3.0);
  nh.param("y_min", params.y_min, -1.5);
  nh.param("y_max", params.y_max, 2.0);
  nh.param("z_min", params.z_min, -0.5);
  nh.param("z_max", params.z_max, 2.0);
  return params;
}

double computeRMSE(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1, 
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2) 
{
    if (cloud1->size() != cloud2->size()) 
    {
      std::cerr << BOLDRED << "[computeRMSE] Point cloud sizes do not match, cannot compute RMSE." << RESET << std::endl;
      return -1.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < cloud1->size(); ++i) 
    {
      double dx = cloud1->points[i].x - cloud2->points[i].x;
      double dy = cloud1->points[i].y - cloud2->points[i].y;
      double dz = cloud1->points[i].z - cloud2->points[i].z;

      sum += dx * dx + dy * dy + dz * dz;
    }

    double mse = sum / cloud1->size();
    return std::sqrt(mse);
}

// 将 LiDAR 点云转换到 QR 码坐标系
void alignPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
  pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud, const Eigen::Matrix4f &transformation) 
{
  output_cloud->clear();
  for (const auto &pt : input_cloud->points) 
  {
    Eigen::Vector4f pt_homogeneous(pt.x, pt.y, pt.z, 1.0);
    Eigen::Vector4f transformed_pt = transformation * pt_homogeneous;
    output_cloud->push_back(pcl::PointXYZ(transformed_pt(0), transformed_pt(1), transformed_pt(2)));
  }
}

void projectPointCloudToImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  const Eigen::Matrix4f& transformation,
  const cv::Mat& cameraMatrix,
  const cv::Mat& distCoeffs,
  const cv::Mat& image,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud) 
{
  colored_cloud->clear();
  colored_cloud->reserve(cloud->size());

  // Undistort the entire image (preprocess outside if possible)
  cv::Mat undistortedImage;
  cv::undistort(image, undistortedImage, cameraMatrix, distCoeffs);

  // Precompute rotation and translation vectors (zero for this case)
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat zeroDistCoeffs = cv::Mat::zeros(5, 1, CV_32F);

  // Preallocate memory for projection
  std::vector<cv::Point3f> objectPoints(1);
  std::vector<cv::Point2f> imagePoints(1);

  for (const auto& point : *cloud) 
  {
    // Transform the point
    Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
    Eigen::Vector4f transformed_point = transformation * homogeneous_point;

    // Skip points behind the camera
    if (transformed_point(2) < 0) continue;

    // Project the point to the image plane
    objectPoints[0] = cv::Point3f(transformed_point(0), transformed_point(1), transformed_point(2));
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, zeroDistCoeffs, imagePoints);

    int u = static_cast<int>(imagePoints[0].x);
    int v = static_cast<int>(imagePoints[0].y);

    // Check if the point is within the image bounds
    if (u >= 0 && u < undistortedImage.cols && v >= 0 && v < undistortedImage.rows) 
    {
      // Get the color from the undistorted image
      cv::Vec3b color = undistortedImage.at<cv::Vec3b>(v, u);

      // Create a colored point and add it to the cloud
      pcl::PointXYZRGB colored_point;
      colored_point.x = transformed_point(0);
      colored_point.y = transformed_point(1);
      colored_point.z = transformed_point(2);
      colored_point.r = color[2];
      colored_point.g = color[1];
      colored_point.b = color[0];
      colored_cloud->push_back(colored_point);
    }
  }
}

void savemidresult(const pcl::PointCloud<pcl::PointXYZ>::Ptr& lidar_centers,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& qr_centers,
                      const Params& params)
{
    if (lidar_centers->size() != 4 || qr_centers->size() != 4) {
      std::cerr << "[savemidresult] The number of points in lidar_centers or qr_centers is not 4, skip saving." << std::endl;
      return;
    }
    
    std::string midresultDir = params.midresult_path;
    if (midresultDir.back() != '/') midresultDir += '/';
    std::ofstream midFile(midresultDir + "circle_center_record.txt", std::ios::app);

    if (!midFile.is_open()) {
        std::cerr << "[savemidresult] Cannot open file: " << midresultDir + "circle_center_record.txt" << std::endl;
        return;
    }

    // 获取当前系统时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    midFile << "time: " << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << std::endl;

    midFile << "lidar_centers:";
    for (const auto& pt : lidar_centers->points) {
        midFile << " {" << pt.x << "," << pt.y << "," << pt.z << "}";
    }
    midFile << std::endl;
    midFile << "qr_centers:";
    for (const auto& pt : qr_centers->points) {
        midFile << " {" << pt.x << "," << pt.y << "," << pt.z << "}";
    }
    midFile << std::endl;
    midFile.close();
    std::cout << BOLDYELLOW << "[Result] Saved the four pairs of center coordinates to " << BOLDWHITE << midresultDir << "circle_center_record.txt" << RESET << std::endl;
}

void saveCalibrationResults(const Params& params, const Eigen::Matrix4f& transformation, 
     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colored_cloud, const cv::Mat& img_input)
{
  if(colored_cloud->empty()) 
  {
    std::cerr << BOLDRED << "[saveCalibrationResults] Colored point cloud is empty!" << RESET << std::endl;
    return;
  }
  std::string outputDir = params.output_path;
  if (outputDir.back() != '/') outputDir += '/';

  std::ofstream outFile(outputDir + "calib_result.txt");
  if (outFile.is_open()) 
  {
    outFile << "# FAST-LIVO2 calibration format\n";
    outFile << "cam_model: Pinhole\n";
    outFile << "cam_width: " << img_input.cols << "\n";
    outFile << "cam_height: " << img_input.rows << "\n";
    outFile << "scale: 1.0\n";
    outFile << "cam_fx: " << params.fx << "\n";
    outFile << "cam_fy: " << params.fy << "\n";
    outFile << "cam_cx: " << params.cx << "\n";
    outFile << "cam_cy: " << params.cy << "\n";
    outFile << "cam_d0: " << params.k1 << "\n";
    outFile << "cam_d1: " << params.k2 << "\n";
    outFile << "cam_d2: " << params.p1 << "\n";
    outFile << "cam_d3: " << params.p2 << "\n";

    outFile << "\nRcl: [" << std::fixed << std::setprecision(6);
    outFile << std::setw(10) << transformation(0, 0) << ", " << std::setw(10) << transformation(0, 1) << ", " << std::setw(10) << transformation(0, 2) << ",\n";
    outFile << "      " << std::setw(10) << transformation(1, 0) << ", " << std::setw(10) << transformation(1, 1) << ", " << std::setw(10) << transformation(1, 2) << ",\n";
    outFile << "      " << std::setw(10) << transformation(2, 0) << ", " << std::setw(10) << transformation(2, 1) << ", " << std::setw(10) << transformation(2, 2) << "]\n";

    outFile << "Pcl: [";
    outFile << std::setw(10) << transformation(0, 3) << ", " << std::setw(10) << transformation(1, 3) << ", " << std::setw(10) << transformation(2, 3) << "]\n";

    outFile.close();
    std::cout << BOLDYELLOW << "[Result] Calibration results saved to " << BOLDWHITE << outputDir << "calib_result.txt" << RESET << std::endl;
  } 
  else
  {
    std::cerr << BOLDRED << "[Error] Failed to open calib_result.txt for writing!" << RESET << std::endl;
  }
  
  if (pcl::io::savePCDFileASCII(outputDir + "colored_cloud.pcd", *colored_cloud) == 0) 
  {
    std::cout << BOLDYELLOW << "[Result] Saved colored point cloud to: " << BOLDWHITE << outputDir << "colored_cloud.pcd" << RESET << std::endl;
  } 
  else 
  {
    std::cerr << BOLDRED << "[Error] Failed to save colored point cloud to " << outputDir << "colored_cloud.pcd" << "!" << RESET << std::endl;
  }
 
  imwrite(outputDir + "qr_detect.png", img_input);
}

void sortPatternCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, pcl::PointCloud<pcl::PointXYZ>::Ptr v, const std::string& axis_mode = "camera") 
{
  // 0 -- 1
  // |    |
  // 3 -- 2
  if(pc->size() != 4) 
  {
    std::cerr << BOLDRED << "[sortPatternCenters] Number of " << axis_mode << " center points to be sorted is not 4." << RESET << std::endl;
    return;
  }
  if (v->empty()) {
    v->clear();
    v->reserve(4);
  }

  // Check axis mode
  bool is_lidar_mode = (axis_mode == "lidar");

  if (is_lidar_mode)
  {
    for (auto& point : pc->points) 
    {
      float x_new = -point.y;   // LiDAR Y → 相机 -X
      float y_new = -point.z;   // LiDAR Z → 相机 -Y
      float z_new = point.x;    // LiDAR X → 相机  Z

      point.x = x_new;
      point.y = y_new;
      point.z = z_new;
    }
  }

  // Transform points to polar coordinates
  pcl::PointCloud<pcl::PointXYZ>::Ptr spherical_centers(
  new pcl::PointCloud<pcl::PointXYZ>());
  int top_pt = 0;
  int index = 0;  // Auxiliar index to be used inside loop
  for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = pc->points.begin();
  pt < pc->points.end(); pt++, index++) 
  {
    pcl::PointXYZ spherical_center;
    spherical_center.x = atan2(pt->y, pt->x);  // Horizontal
    spherical_center.y =
    atan2(sqrt(pt->x * pt->x + pt->y * pt->y), pt->z);  // Vertical
    spherical_center.z =
    sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);  // Range
    spherical_centers->push_back(spherical_center);

    if (spherical_center.y < spherical_centers->points[top_pt].y) 
    {
      top_pt = index;
    }
  }

  // Compute distances from top-most center to rest of points
  vector<double> distances;
  for (int i = 0; i < 4; i++) {
    pcl::PointXYZ pt = pc->points[i];
    pcl::PointXYZ upper_pt = pc->points[top_pt];
    distances.push_back(sqrt(pow(pt.x - upper_pt.x, 2) +
        pow(pt.y - upper_pt.y, 2) +
        pow(pt.z - upper_pt.z, 2)));
  }

  // Get indices of closest and furthest points
  int min_dist = (top_pt + 1) % 4, max_dist = top_pt;
  for (int i = 0; i < 4; i++) {
    if (i == top_pt) continue;
    if (distances[i] > distances[max_dist]) {
      max_dist = i;
    }
    if (distances[i] < distances[min_dist]) {
      min_dist = i;
    }
  }

  // Second highest point shoud be the one whose distance is the median value
  int top_pt2 = 6 - (top_pt + max_dist + min_dist);  // 0 + 1 + 2 + 3 = 6

  // Order upper row centers
  int lefttop_pt = top_pt;
  int righttop_pt = top_pt2;

  if (spherical_centers->points[top_pt].x <
    spherical_centers->points[top_pt2].x) {
    int aux = lefttop_pt;
    lefttop_pt = righttop_pt;
    righttop_pt = aux;
  }

  // Swap indices if target is located in the pi,-pi discontinuity
  double angle_diff = spherical_centers->points[lefttop_pt].x -
  spherical_centers->points[righttop_pt].x;
  if (angle_diff > M_PI - spherical_centers->points[lefttop_pt].x) {
    int aux = lefttop_pt;
    lefttop_pt = righttop_pt;
    righttop_pt = aux;
  }

  // Define bottom row centers using lefttop == top_pt as hypothesis
  int leftbottom_pt = min_dist;
  int rightbottom_pt = max_dist;

  // If lefttop != top_pt, swap indices
  if (righttop_pt == top_pt) {
    leftbottom_pt = max_dist;
    rightbottom_pt = min_dist;
  }

  // Fill vector with sorted centers
  v->push_back(pc->points[lefttop_pt]);
  v->push_back(pc->points[righttop_pt]);
  v->push_back(pc->points[rightbottom_pt]);
  v->push_back(pc->points[leftbottom_pt]);

  if (is_lidar_mode) 
  {
    for (auto& point : v->points)
    {
      float x_new = point.z;  
      float y_new = -point.x; 
      float z_new = -point.y;  

      point.x = x_new;
      point.y = y_new;
      point.z = z_new;
    }
  }
}

class Square 
{
  private:
   pcl::PointXYZ _center;
   std::vector<pcl::PointXYZ> _candidates;
   float _target_width, _target_height, _target_diagonal;
 
  public:
   Square(std::vector<pcl::PointXYZ> candidates, float width, float height) {
     _candidates = candidates;
     _target_width = width;
     _target_height = height;
     _target_diagonal = sqrt(pow(width, 2) + pow(height, 2));
 
     // Compute candidates centroid
     for (int i = 0; i < candidates.size(); ++i) {
       _center.x += candidates[i].x;
       _center.y += candidates[i].y;
       _center.z += candidates[i].z;
     }
 
     _center.x /= candidates.size();
     _center.y /= candidates.size();
     _center.z /= candidates.size();
   }
 
   float distance(pcl::PointXYZ pt1, pcl::PointXYZ pt2) {
     return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2) +
                 pow(pt1.z - pt2.z, 2));
   }
 
   float perimeter() {  // TODO: It is assumed that _candidates are ordered, it
                        // shouldn't
     float perimeter = 0;
     for (int i = 0; i < 4; ++i) {
       perimeter += distance(_candidates[i], _candidates[(i + 1) % 4]);
     }
     return perimeter;
   }
 
   pcl::PointXYZ at(int i) {
     assert(0 <= i && i < 4);
     return _candidates[i];
   }
 
   bool is_valid() 
   {
     pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(
         new pcl::PointCloud<pcl::PointXYZ>());
     // Check if candidates are at 5% of target's diagonal/2 to their centroid
     for (int i = 0; i < _candidates.size(); ++i) {
       candidates_cloud->push_back(_candidates[i]);
       float d = distance(_center, _candidates[i]);
       if (fabs(d - _target_diagonal / 2.) / (_target_diagonal / 2.) >
           GEOMETRY_TOLERANCE) {
         return false;
       }
     }
     // Check perimeter?
     pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_centers(new pcl::PointCloud<pcl::PointXYZ>());
     sortPatternCenters(candidates_cloud, sorted_centers, "camera");
     float perimeter = 0;
     for (int i = 0; i < sorted_centers->size(); ++i) {
       float current_distance = distance(
           sorted_centers->points[i], sorted_centers->points[(i + 1) % sorted_centers->size()]);
       if (i % 2) {
         if (fabs(current_distance - _target_height) / _target_height >
             GEOMETRY_TOLERANCE) {
           return false;
         }
       } else {
         if (fabs(current_distance - _target_width) / _target_width >
             GEOMETRY_TOLERANCE) {
           return false;
         }
       }
       perimeter += current_distance;
     }
     float ideal_perimeter = (2 * _target_width + 2 * _target_height);
     if (fabs((perimeter - ideal_perimeter) / ideal_perimeter >
              GEOMETRY_TOLERANCE)) {
       return false;
     }
 
     // Check width + height?
     return true;
   }
};

#endif