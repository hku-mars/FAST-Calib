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
  double fx, fy, cx, cy, k1, k2, p1, p2, k3, k4; // 添加 k3 和 k4
  double marker_size, delta_width_qr_center, delta_height_qr_center;
  double delta_width_circles, delta_height_circles, circle_radius;
  int min_detected_markers;
  string image_path;
  string bag_path;
  string lidar_topic;
  string output_path;
  string cam_model; // 新增相机模型
};

// 读取参数
Params loadParameters(ros::NodeHandle &nh) {
  Params params;
  nh.param<string>("cam_model", params.cam_model, "pinhole"); // 读取相机模型，默认为 pinhole
  nh.param("fx", params.fx, 1215.31801774424);
  nh.param("fy", params.fy, 1214.72961288138);
  nh.param("cx", params.cx, 1047.86571859677);
  nh.param("cy", params.cy, 745.068353101898);
  nh.param("k1", params.k1, -0.33574781188503);
  nh.param("k2", params.k2, 0.10996870793601);
  nh.param("p1", params.p1, 0.000157303079833973);
  nh.param("p2", params.p2, 0.000544930726278493);
  nh.param("k3", params.k3, 0.0); // 读取 k3
  nh.param("k4", params.k4, 0.0); // 读取 k4
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

void saveTargetHoleCenters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& lidar_centers,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& qr_centers,
                      const Params& params)
{
    if (lidar_centers->size() != 4 || qr_centers->size() != 4) {
      std::cerr << "[saveTargetHoleCenters] The number of points in lidar_centers or qr_centers is not 4, skip saving." << std::endl;
      return;
    }
    
    std::string saveDir = params.output_path;
    if (saveDir.back() != '/') saveDir += '/';
    std::ofstream saveFile(saveDir + "circle_center_record.txt", std::ios::app);

    if (!saveFile.is_open()) {
        std::cerr << "[saveTargetHoleCenters] Cannot open file: " << saveDir + "circle_center_record.txt" << std::endl;
        return;
    }

    // 获取当前系统时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    saveFile << "time: " << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << std::endl;

    saveFile << "lidar_centers:";
    for (const auto& pt : lidar_centers->points) {
        saveFile << " {" << pt.x << "," << pt.y << "," << pt.z << "}";
    }
    saveFile << std::endl;
    saveFile << "qr_centers:";
    for (const auto& pt : qr_centers->points) {
        saveFile << " {" << pt.x << "," << pt.y << "," << pt.z << "}";
    }
    saveFile << std::endl;
    saveFile.close();
    std::cout << BOLDGREEN << "[Record] Saved four pairs of circular hole centers to " << BOLDWHITE << saveDir << "circle_center_record.txt" << RESET << std::endl;
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

  std::ofstream outFile(outputDir + "single_calib_result.txt");
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
    std::cout << BOLDYELLOW << "[Result] Single-scene calibration results saved to " << BOLDWHITE << outputDir << "single_calib_result.txt" << RESET << std::endl;
  } 
  else
  {
    std::cerr << BOLDRED << "[Error] Failed to open single_calib_result.txt for writing!" << RESET << std::endl;
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

void sortPatternCenters(pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr v,
                        const std::string& axis_mode = "camera") 
{
  if (pc->size() != 4) {
    std::cerr << BOLDRED << "[sortPatternCenters] Number of " << axis_mode << " center points to be sorted is not 4." << RESET << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr work_pc(new pcl::PointCloud<pcl::PointXYZ>());

  // Coordinate transformation (LiDAR -> Camera)
  if (axis_mode == "lidar") {
    for (const auto& p : *pc) {
      pcl::PointXYZ pt;
      pt.x = -p.y;   // LiDAR Y -> Cam -X
      pt.y = -p.z;   // LiDAR Z -> Cam -Y
      pt.z = p.x;    // LiDAR X -> Cam Z
      work_pc->push_back(pt);
    }
  } else {
    *work_pc = *pc;
  }

  // --- Sorting based on the local coordinate system of the pattern ---
  // 1. Calculate the centroid of the points
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*work_pc, centroid);
  pcl::PointXYZ ref_origin(centroid[0], centroid[1], centroid[2]);

  // 2. Project points to the XY plane relative to the centroid and calculate angles
  std::vector<std::pair<float, int>> proj_points;
  for (size_t i = 0; i < work_pc->size(); ++i) {
    const auto& p = work_pc->points[i];
    Eigen::Vector3f rel_vec(p.x - ref_origin.x, p.y - ref_origin.y, p.z - ref_origin.z);
    proj_points.emplace_back(atan2(rel_vec.y(), rel_vec.x()), i);
  }

  // 3. Sort points based on the calculated angle
  std::sort(proj_points.begin(), proj_points.end());

  // 4. Output the sorted points into the result vector 'v'
  v->resize(4);
  for (int i = 0; i < 4; ++i) {
    (*v)[i] = work_pc->points[proj_points[i].second];
  }

  // 5. Verify the order (ensure it's counter-clockwise) and fix if necessary
  const auto& p0 = v->points[0];
  const auto& p1 = v->points[1];
  const auto& p2 = v->points[2];
  Eigen::Vector3f v01(p1.x - p0.x, p1.y - p0.y, 0);
  Eigen::Vector3f v12(p2.x - p1.x, p2.y - p1.y, 0);
  if (v01.cross(v12).z() > 0) {
    std::swap((*v)[1], (*v)[3]);
  }

  // 6. If the original input was in the lidar frame, transform the sorted points back
  if (axis_mode == "lidar") {
    for (auto& point : v->points) {
      float x_new = point.z;    // Cam Z -> LiDAR X
      float y_new = -point.x;   // Cam -X -> LiDAR Y
      float z_new = -point.y;   // Cam -Y -> LiDAR Z
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
      _center.x = _center.y = _center.z = 0;
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
 
    pcl::PointXYZ at(int i) {
      assert(0 <= i && i < 4);
      return _candidates[i];
    }
 
    // ==================================================================================================
    // The original is_valid() was too rigid. This version is more robust by checking for two possible
    // orderings of the side lengths (width-height vs. height-width) after angular sorting.
    // ==================================================================================================
    bool is_valid() 
    {
      if (_candidates.size() != 4) return false;

      pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      for(const auto& p : _candidates) candidates_cloud->push_back(p);

      // Check if candidates are at a reasonable distance from their centroid
      for (int i = 0; i < _candidates.size(); ++i) {
        float d = distance(_center, _candidates[i]);
        // Check if distance from center to corner is close to half the diagonal length
        if (fabs(d - _target_diagonal / 2.) / (_target_diagonal / 2.) > GEOMETRY_TOLERANCE * 2.0) { // Loosened tolerance slightly
          return false;
        }
      }
      
      // Sort the corners counter-clockwise
      pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_centers(new pcl::PointCloud<pcl::PointXYZ>());
      sortPatternCenters(candidates_cloud, sorted_centers, "camera");
      
      // Get the four side lengths from the sorted points
      float s01 = distance(sorted_centers->points[0], sorted_centers->points[1]);
      float s12 = distance(sorted_centers->points[1], sorted_centers->points[2]);
      float s23 = distance(sorted_centers->points[2], sorted_centers->points[3]);
      float s30 = distance(sorted_centers->points[3], sorted_centers->points[0]);

      // Check for pattern 1: width, height, width, height
      bool pattern1_ok = 
        (fabs(s01 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s12 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s23 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s30 - _target_height) / _target_height < GEOMETRY_TOLERANCE);

      // Check for pattern 2: height, width, height, width
      bool pattern2_ok = 
        (fabs(s01 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s12 - _target_width) / _target_width < GEOMETRY_TOLERANCE) &&
        (fabs(s23 - _target_height) / _target_height < GEOMETRY_TOLERANCE) &&
        (fabs(s30 - _target_width) / _target_width < GEOMETRY_TOLERANCE);

      if (!pattern1_ok && !pattern2_ok) {
        return false;
      }
      
      // Final check on perimeter
      float perimeter = s01 + s12 + s23 + s30;
      float ideal_perimeter = 2 * (_target_width + _target_height);
      if (fabs(perimeter - ideal_perimeter) / ideal_perimeter > GEOMETRY_TOLERANCE) {
        return false;
      }
 
      return true;
    }
};

#endif
