/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#define PCL_NO_PRECOMPILE

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;

    // 存储中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr filtered_cloud_;
    pcl::PointCloud<Common::Point>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    LidarDetect(ros::NodeHandle &nh, Params &params)
        : filtered_cloud_(new pcl::PointCloud<Common::Point>),
          plane_cloud_(new pcl::PointCloud<Common::Point>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        delta_width_circles_ = params.delta_width_circles;
        delta_height_circles_ = params.delta_height_circles;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    void detect_mech_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);

        ROS_INFO("Depth filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 拟合平面，提取法向量
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
    
        // 3. 根据每条ring相邻点距离提取边缘点
        edge_cloud_->reserve(filtered_cloud_->size());

        // 先按 ring 分组
        std::unordered_map<unsigned int, std::vector<int>> ring2indices;
        ring2indices.reserve(64);

        for (int i = 0; i < static_cast<int>(filtered_cloud_->size()); ++i)
        {
            const auto &pt = filtered_cloud_->points[i];
            ring2indices[pt.ring].push_back(i);
        }

        const auto &c = plane_coefficients->values; // [a, b, c, d]
        Eigen::Vector3d n(c[0], c[1], c[2]);
        double norm_n = n.norm();
        Eigen::Vector3d normal = n / norm_n;

        // 在每条 ring 内，用相邻点距离检测跳变点作为边缘点
        const double neighbor_gap_threshold = 0.10;  // 邻近点距离阈值
        const int    min_points_per_ring    = 10;    // 太短的 ring 不处理

        for (auto &kv : ring2indices)
        {
            auto &idx_vec = kv.second;
            if (static_cast<int>(idx_vec.size()) < min_points_per_ring) continue;

            for (size_t k = 1; k + 1 < idx_vec.size(); ++k)
            {
                const auto &p_prev = filtered_cloud_->points[idx_vec[k - 1]];
                const auto &p_cur  = filtered_cloud_->points[idx_vec[k]];
                const auto &p_next = filtered_cloud_->points[idx_vec[k + 1]];

                // 只保留落在拟合平面附近的点
                double dist_plane = std::fabs(c[0]*p_cur.x + c[1]*p_cur.y + c[2]*p_cur.z + c[3]) / norm_n;
                if (dist_plane >= 0.03) continue;

                // cur 与 prev 的距离
                double dx1 = static_cast<double>(p_cur.x) - static_cast<double>(p_prev.x);
                double dy1 = static_cast<double>(p_cur.y) - static_cast<double>(p_prev.y);
                double dz1 = static_cast<double>(p_cur.z) - static_cast<double>(p_prev.z);
                double dist_prev = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

                // cur 与 next 的距离
                double dx2 = static_cast<double>(p_cur.x) - static_cast<double>(p_next.x);
                double dy2 = static_cast<double>(p_cur.y) - static_cast<double>(p_next.y);
                double dz2 = static_cast<double>(p_cur.z) - static_cast<double>(p_next.z);
                double dist_next = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

                // 只要和前一个或后一个之间有一个距离超过阈值，就认为当前点是边缘点
                if (dist_prev > neighbor_gap_threshold || dist_next > neighbor_gap_threshold)
                {
                    edge_cloud_->push_back(pcl::PointXYZ(p_cur.x, p_cur.y, p_cur.z));
                }
            }
        }

        ROS_INFO("Extracted %zu edge points (mechanical LiDAR by neighbor distance).", edge_cloud_->size());

        // 4. 将边缘点对齐到 Z=0 平面
        aligned_cloud_->reserve(edge_cloud_->size());
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R_align = rotation.toRotationMatrix();

        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *edge_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 5. 在对齐后的点云中检测圆形，提取圆心
        
        // 拷贝一份工作点云，后面要不停“删掉已拟合的圆”
        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));

        ROS_INFO("[LiDAR] Start circle detection, initial cloud size = %zu", xy_cloud->points.size());

        // 在对齐后的平面上，用 RANSAC 反复检测 2D 圆
        pcl::SACSegmentation<pcl::PointXYZ> circle_segmentation;
        circle_segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
        circle_segmentation.setMethodType(pcl::SAC_RANSAC);
        circle_segmentation.setDistanceThreshold(0.02);
        circle_segmentation.setOptimizeCoefficients(true);
        circle_segmentation.setMaxIterations(1000);
        circle_segmentation.setRadiusLimits(circle_radius_ - 0.03, circle_radius_ + 0.03);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ExtractIndices<pcl::PointXYZ> extract2;

        // 不停在剩余点云中找圆
        while (xy_cloud->points.size() > 3)
        {
            ROS_INFO("[LiDAR] RANSAC on cloud of size %lu", xy_cloud->points.size());

            circle_segmentation.setInputCloud(xy_cloud);
            circle_segmentation.segment(*inliers, *coefficients);

            // 没有 inliers，说明没有可用的圆，结束
            if (inliers->indices.empty())
            {
                ROS_INFO("[LiDAR] No more circles can be found, stop.");
                break;
            }

            // 内点太少就认为是噪声，直接结束
            if ((int)inliers->indices.size() < 5)
            {
                ROS_INFO("[LiDAR] Found circle but inliers too few (%zu < 3), stop.",
                            inliers->indices.size());
                break;
            }

            // ROS_INFO("[LiDAR] Circle found: inliers = %zu, coeffs size = %zu",
            //         inliers->indices.size(), coefficients->values.size());
            // 对 Circle2D 而言，coeffs 通常是 [xc, yc, r]
            // if (coefficients->values.size() >= 3)
            // {
            //     ROS_INFO("[LiDAR]   center = (%.4f, %.4f), r = %.4f",
            //             coefficients->values[0],
            //             coefficients->values[1],
            //             coefficients->values[2]);
            // }
            
            // 记录当前这个圆
            pcl::PointXYZ center_point;
            center_point.x = coefficients->values[0];
            center_point.y = coefficients->values[1];
            center_point.z = 0;
            center_z0_cloud_->push_back(center_point);

            // 把当前圆的 inliers 从点云中移除，继续在剩余点中找
            extract2.setInputCloud(xy_cloud);
            extract2.setIndices(inliers);
            extract2.setNegative(true);  // 保留非 inliers（即剩余点）
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract2.filter(*remaining);
            xy_cloud.swap(remaining);

            // 清空 inliers，避免下一轮残留
            inliers->indices.clear();
        }

        // 6. Geometric consistency check
        std::vector<std::vector<int>> groups;
        comb(center_z0_cloud_->size(), TARGET_NUM_CIRCLES, groups);
        double groups_scores[groups.size()];  // -1: invalid; 0-1 normalized score
        for (int i = 0; i < groups.size(); ++i) 
        {
            std::vector<pcl::PointXYZ> candidates;
            // Build candidates set
            for (int j = 0; j < groups[i].size(); ++j) 
            {
                pcl::PointXYZ center;
                center.x = center_z0_cloud_->at(groups[i][j]).x;
                center.y = center_z0_cloud_->at(groups[i][j]).y;
                center.z = center_z0_cloud_->at(groups[i][j]).z;
                candidates.push_back(center);
            }

            // Compute candidates score
            Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
            groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1;  // -1 when it's not valid, 1 otherwise
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (int i = 0; i < groups.size(); ++i) 
        {
            if (best_candidate_score == 1 && groups_scores[i] == 1) 
            {
                // Exit 4: Several candidates fit target's geometry
                ROS_ERROR(
                    "[LiDAR] More than one set of candidates fit target's geometry. "
                    "Please, make sure your parameters are well set. Exiting callback");
                return;
            }
            if (groups_scores[i] > best_candidate_score) 
            {
                best_candidate_score = groups_scores[i];
                best_candidate_idx = i;
            }
        }
        if (best_candidate_idx == -1) 
        {
            // Exit 5: No candidates fit target's geometry
            ROS_WARN(
                "[LiDAR] Unable to find a candidate set that matches target's "
                "geometry");
            return;
        }
        
        // 7. 将选中的圆心逆变换回原始坐标系
        Eigen::Matrix3d R_inv = R_align.inverse();
        for (int j = 0; j < groups[best_candidate_idx].size(); ++j) 
        {
            pcl::PointXYZ center;
            center.x = center_z0_cloud_->at(groups[best_candidate_idx][j]).x;
            center.y = center_z0_cloud_->at(groups[best_candidate_idx][j]).y;
            center.z = center_z0_cloud_->at(groups[best_candidate_idx][j]).z;

            // 将圆心坐标逆变换回原始坐标系
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }

    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());
        
        pcl::VoxelGrid<Common::Point> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        ROS_INFO("Extracted %zu edge points.", edge_cloud_->size());

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.05); // 设置聚类距离阈值
        ec.setMinClusterSize(50);     // 最小点数
        ec.setMaxClusterSize(1000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        ROS_INFO("Number of edge clusters: %zu", cluster_indices.size());
    
        // 6. 对每个聚类进行圆拟合
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();
    
        // 对每个聚类进行圆拟合
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            // 圆拟合
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01); // 设置距离阈值
            seg.setMaxIterations(1000);     // 设置最大迭代次数
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);
    
            if (inliers->indices.size() > 0) 
            {
                // 计算拟合误差
                double error = 0.0;
                for (const auto& idx : inliers->indices) 
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
                    error += abs(distance);
                }
                error /= inliers->indices.size();
    
                // 如果拟合误差较小，则认为是一个圆洞
                if (error < 0.025) 
                {
                    // 将恢复后的圆心坐标添加到点云中
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    center_z0_cloud_->push_back(center_point);

                    // 将圆心坐标逆变换回原始坐标系
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;

                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    center_cloud->points.push_back(center_point_origin);
                }
            }
        }
    }
    // 获取中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
