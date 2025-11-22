/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef QR_DETECT_HPP
#define QR_DETECT_HPP
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <opencv2/aruco.hpp>
#include "common_lib.h"
#include <opencv2/ccalib/omnidir.hpp> // 添加鱼眼相机模型的头文件

class QRDetect 
{
  private:
    double marker_size_, delta_width_qr_center_, delta_height_qr_center_;
    double delta_width_circles_, delta_height_circles_;
    int min_detected_markers_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    string cam_model_; // 添加成员变量来存储相机模型
    Params params_; // 用于存储lidar安装朝向配置参数

  
  public:
    ros::Publisher qr_pub_;
    cv::Mat imageCopy_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;

    QRDetect(ros::NodeHandle &nh, Params& params) 
    {
      params_ = params; // 保存参数副本
      marker_size_ = params.marker_size;
      delta_width_qr_center_ = params.delta_width_qr_center;
      delta_height_qr_center_ = params.delta_height_qr_center;
      delta_width_circles_ = params.delta_width_circles;
      delta_height_circles_ = params.delta_height_circles;
      min_detected_markers_ = params.min_detected_markers;
      cam_model_ = params.cam_model; // 存储相机模型
      
      // Initialize camera matrix
      cameraMatrix_ = (cv::Mat_<float>(3, 3) << params.fx, 0, params.cx,
                                                0, params.fy, params.cy,
                                                0,         0,        1);
                                                
      // Initialize distortion coefficients
      //distCoeffs_ = (cv::Mat_<float>(1, 5) << params.k1, params.k2, params.p1, params.p2, 0);
      // 根据相机模型初始化畸变系数
      if (cam_model_ == "fisheye") {
        distCoeffs_ = (cv::Mat_<float>(1, 4) << params.k1, params.k2, params.k3, params.k4);
      } else { // 默认为 pinhole
        distCoeffs_ = (cv::Mat_<float>(1, 5) << params.k1, params.k2, params.p1, params.p2, 0);
      }

      // Initialize QR dictionary
      dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

      qr_pub_ = nh.advertise<sensor_msgs::PointCloud2>("qr_cloud", 1);
    }

    Point2f projectPointDist(cv::Point3f pt_cv, const Mat intrinsics, const Mat distCoeffs) 
    {
      // Project a 3D point taking into account distortion
      vector<Point3f> input{pt_cv};
      vector<Point2f> projectedPoints;
      projectedPoints.resize(1);  // TODO: Do it batched? (cv::circle is not batched anyway)
      projectPoints(input, Mat::zeros(3, 1, CV_64FC1), Mat::zeros(3, 1, CV_64FC1),
      intrinsics, distCoeffs, projectedPoints);
      return projectedPoints[0];
    }

    void comb(int N, int K, std::vector<std::vector<int>> &groups) {
      int upper_factorial = 1;
      int lower_factorial = 1;

      for (int i = 0; i < K; i++) {
        upper_factorial *= (N - i);
        lower_factorial *= (K - i);
      }
      int n_permutations = upper_factorial / lower_factorial;

      if (DEBUG)
        cout << N << " centers found. Iterating over " << n_permutations
            << " possible sets of candidates" << endl;

      std::string bitmask(K, 1);  // K leading 1's
      bitmask.resize(N, 0);       // N-K trailing 0's

      // print integers and permute bitmask
      do {
        std::vector<int> group;
        for (int i = 0; i < N; ++i)  // [0..N-1] integers
        {
          if (bitmask[i]) {
            group.push_back(i);
          }
        }
        groups.push_back(group);
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

      assert(groups.size() == n_permutations);
    }

    void detect_qr(cv::Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud) 
    {      
      image.copyTo(imageCopy_);

      // Create vector of markers corners. 4 markers * 4 corners
      // Markers order:
      // 0-------1
      // |       |
      // |   C   |
      // |       |
      // 3-------2

      // WARNING: IDs are in different order:
      // Marker 0 -> aRuCo ID: 1
      // Marker 1 -> aRuCo ID: 2
      // Marker 2 -> aRuCo ID: 4
      // Marker 3 -> aRuCo ID: 3

      std::vector<std::vector<cv::Point3f>> boardCorners;
      std::vector<cv::Point3f> boardCircleCenters;
      float width = delta_width_qr_center_;
      float height = delta_height_qr_center_;
      float circle_width = delta_width_circles_ / 2.;
      float circle_height = delta_height_circles_ / 2.;
      boardCorners.resize(4);
      for (int i = 0; i < 4; ++i) {
        int x_qr_center =
            (i % 3) == 0 ? -1 : 1;  // x distances are substracted for QRs on the
                                    // left, added otherwise
        int y_qr_center =
            (i < 2) ? 1 : -1;  // y distances are added for QRs above target's
                              // center, substracted otherwise
        float x_center = x_qr_center * width;
        float y_center = y_qr_center * height;

        cv::Point3f circleCenter3d(x_qr_center * circle_width,
                                  y_qr_center * circle_height, 0);
        boardCircleCenters.push_back(circleCenter3d);
        for (int j = 0; j < 4; ++j) {
          int x_qr = (j % 3) == 0 ? -1 : 1;  // x distances are added for QRs 0 and
                                            // 3, substracted otherwise
          int y_qr = (j < 2) ? 1 : -1;  // y distances are added for QRs 0 and 1,
                                        // substracted otherwise
          cv::Point3f pt3d(x_center + x_qr * marker_size_ / 2.,
                          y_center + y_qr * marker_size_ / 2., 0);
          boardCorners[i].push_back(pt3d);
        }
      }

      std::vector<int> boardIds{1, 2, 4, 3};  // IDs order as explained above
      cv::Ptr<cv::aruco::Board> board =
          cv::aruco::Board::create(boardCorners, dictionary_, boardIds);

      cv::Ptr<cv::aruco::DetectorParameters> parameters =
          cv::aruco::DetectorParameters::create();
      // set tp use corner refinement for accuracy, values obtained
      // for pixel coordinates are more accurate than the neaterst pixel

    #if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
      parameters->doCornerRefinement = true;
    #else
      parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    #endif

      // Detect markers
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners, undistorted_corners; // undistorted_corners 用于鱼眼模型
      cv::aruco::detectMarkers(image, dictionary_, corners, ids, parameters);

      // 如果是鱼眼模型，先对角点进行去畸变
      if (cam_model_ == "fisheye" && !corners.empty()) {
          undistorted_corners.resize(corners.size());
          for(size_t i = 0; i < corners.size(); i++){
              cv::fisheye::undistortPoints(corners[i], undistorted_corners[i], cameraMatrix_, distCoeffs_, cv::noArray(), cameraMatrix_);
          }
      } else {
          undistorted_corners = corners; // Pinhole模型直接使用检测到的角点
      }

      // Draw detections if at least one marker detected
      if (ids.size() > 0) cv::aruco::drawDetectedMarkers(imageCopy_, corners, ids);

      cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);  // Vectors to store initial guess
      
      // cout << "min_detected_markers_: " << min_detected_markers_ << std::endl;

      // ids.size(): 4
      // Compute initial guess as average of individual markers poses
      if (ids.size() >= min_detected_markers_ && ids.size() <= TARGET_NUM_CIRCLES) 
      {
        // Estimate 3D position of the markers
        vector<Vec3d> rvecs, tvecs;
        Vec3f rvec_sin, rvec_cos;
        // 根据模型选择不同的畸变参数进行位姿估计
        cv::Mat dist_coeffs_for_pose = (cam_model_ == "fisheye") ? cv::Mat() : distCoeffs_;
        // 注意：对于鱼眼模型，我们使用了去畸变后的点，所以畸变参数传空Mat
        cv::aruco::estimatePoseSingleMarkers(undistorted_corners, marker_size_, cameraMatrix_,
                                                 dist_coeffs_for_pose, rvecs, tvecs);

        // Draw markers' axis and centers in color image (Debug purposes)
        for (int i = 0; i < ids.size(); i++) {
          double x = tvecs[i][0];
          double y = tvecs[i][1];
          double z = tvecs[i][2];

          cv::aruco::drawAxis(imageCopy_, cameraMatrix_, distCoeffs_, rvecs[i],
                              tvecs[i], 0.1);

          // Accumulate pose for initial guess
          tvec[0] += tvecs[i][0];
          tvec[1] += tvecs[i][1];
          tvec[2] += tvecs[i][2];
          rvec_sin[0] += sin(rvecs[i][0]);
          rvec_sin[1] += sin(rvecs[i][1]);
          rvec_sin[2] += sin(rvecs[i][2]);
          rvec_cos[0] += cos(rvecs[i][0]);
          rvec_cos[1] += cos(rvecs[i][1]);
          rvec_cos[2] += cos(rvecs[i][2]);
        }

        // Compute average pose. Rotation computed as atan2(sin/cos)
        tvec = tvec / int(ids.size());
        rvec_sin = rvec_sin / int(ids.size());  // Average sin
        rvec_cos = rvec_cos / int(ids.size());  // Average cos
        rvec[0] = atan2(rvec_sin[0], rvec_cos[0]);
        rvec[1] = atan2(rvec_sin[1], rvec_cos[1]);
        rvec[2] = atan2(rvec_sin[2], rvec_cos[2]);

        // cout << "single: " <<  tvec[0] << ", "<< tvec[1] << ", " << tvec[2] << std::endl;

        // pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Estimate 3D position of the board using detected markers
    #if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
        int valid = cv::aruco::estimatePoseBoard(undistorted_corners, ids, board, cameraMatrix_,
                                                     dist_coeffs_for_pose, rvec, tvec);
    #else
        int valid = cv::aruco::estimatePoseBoard(undistorted_corners, ids, board, cameraMatrix_,
                                                     dist_coeffs_for_pose, rvec, tvec, true);
    #endif


        // cout << "board: " <<  tvec[0] << ", "<< tvec[1] << ", " << tvec[2] << std::endl;

        cv::aruco::drawAxis(imageCopy_, cameraMatrix_, distCoeffs_, rvec, tvec, 0.2);

        // Build transformation matrix to calibration target axis
        cv::Mat R(3, 3, cv::DataType<float>::type);
        cv::Rodrigues(rvec, R);

        cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
        t.at<float>(0) = tvec[0];
        t.at<float>(1) = tvec[1];
        t.at<float>(2) = tvec[2];

        cv::Mat board_transform = cv::Mat::eye(3, 4, CV_32F);
        R.copyTo(board_transform.rowRange(0, 3).colRange(0, 3));
        t.copyTo(board_transform.rowRange(0, 3).col(3));

        // Compute coordintates of circle centers
        for (int i = 0; i < boardCircleCenters.size(); ++i) {
          cv::Mat mat = cv::Mat::zeros(4, 1, CV_32F);
          mat.at<float>(0, 0) = boardCircleCenters[i].x;
          mat.at<float>(1, 0) = boardCircleCenters[i].y;
          mat.at<float>(2, 0) = boardCircleCenters[i].z;
          mat.at<float>(3, 0) = 1.0;

          // Transform center to target coords
          cv::Mat mat_qr = board_transform * mat;
          cv::Point3f center3d;
          center3d.x = mat_qr.at<float>(0, 0);
          center3d.y = mat_qr.at<float>(1, 0);
          center3d.z = mat_qr.at<float>(2, 0);

          // Draw center (DEBUG)
          cv::Point2f uv;
          uv = projectPointDist(center3d, cameraMatrix_, distCoeffs_);
          circle(imageCopy_, uv, 5, Scalar(0, 255, 0), -1);

          // Add center to list
          pcl::PointXYZ qr_center;
          qr_center.x = center3d.x;
          qr_center.y = center3d.y;
          qr_center.z = center3d.z;
          candidates_cloud->push_back(qr_center);
        }

        /**
          NOTE: This is included here in the same way as the rest of the modalities
        to avoid obvious misdetections, which sometimes happened in our experiments.
        In this modality, it should be impossible to have more than a set of
        candidates, but we keep the ability of handling different combinations for
        eventual future extensions.

          Geometric consistency check
          At this point, circles' center candidates have been computed
        (found_centers). Now we need to select the set of 4 candidates that best fit
        the calibration target geometry. To that end, the following steps are
        followed: 1) Create a cloud with 4 points representing the exact geometry of
        the calibration target 2) For each possible set of 4 points: compute
        similarity score 3) Rotate back the candidates with the highest score to
        their original position in the cloud, and add them to cumulative cloud
        **/
        std::vector<std::vector<int>> groups;
        comb(candidates_cloud->size(), TARGET_NUM_CIRCLES, groups);
        double groups_scores[groups.size()];  // -1: invalid; 0-1 normalized score
        // groups.size() 1

        for (int i = 0; i < groups.size(); ++i) 
        {
          std::vector<pcl::PointXYZ> candidates;
          // Build candidates set
          for (int j = 0; j < groups[i].size(); ++j) {
            pcl::PointXYZ center;
            center.x = candidates_cloud->at(groups[i][j]).x;
            center.y = candidates_cloud->at(groups[i][j]).y;
            center.z = candidates_cloud->at(groups[i][j]).z;
            candidates.push_back(center);
          }

          // Compute candidates score
          Square square_candidate(candidates, delta_width_circles_,
                                  delta_height_circles_);
          groups_scores[i] = square_candidate.is_valid(params_)
                                ? 1.0
                                : -1;  // -1 when it's not valid, 1 otherwise
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (int i = 0; i < groups.size(); ++i) 
        {
          if (best_candidate_score == 1 && groups_scores[i] == 1) {
            // Exit 4: Several candidates fit target's geometry
            ROS_ERROR(
                "[Mono] More than one set of candidates fit target's geometry. "
                "Please, make sure your parameters are well set. Exiting callback");
            return;
          }
          if (groups_scores[i] > best_candidate_score) {
            best_candidate_score = groups_scores[i];
            best_candidate_idx = i;
          }
        }

        if (best_candidate_idx == -1) 
        {
          // Exit: No candidates fit target's geometry
          ROS_WARN(
              "[Mono] Unable to find a candidate set that matches target's "
              "geometry");
          return;
        }

        for (int j = 0; j < groups[best_candidate_idx].size(); ++j) 
        {
          centers_cloud->push_back(candidates_cloud->at(groups[best_candidate_idx][j]));
        }

        if (DEBUG) 
        {  // Draw centers
          for (int i = 0; i < centers_cloud->size(); i++) {
            cv::Point3f pt_circle1(centers_cloud->at(i).x, centers_cloud->at(i).y,centers_cloud->at(i).z);
            cv::Point2f uv_circle1;
            uv_circle1 = projectPointDist(pt_circle1, cameraMatrix_, distCoeffs_);
            circle(imageCopy_, uv_circle1, 2, Scalar(255, 0, 255), -1);
          }
        }

        // Publish pointcloud messages
      } 
      else 
      {
        // Markers found != TARGET_NUM_CIRCLES
        ROS_WARN("%lu marker(s) found, %d expected. Skipping frame...", ids.size(),
                TARGET_NUM_CIRCLES);
      }
    }
};
typedef std::shared_ptr<QRDetect> QRDetectPtr;

#endif
