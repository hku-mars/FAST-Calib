/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include "CustomMsg.h"
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <fstream>
#include "common_lib.h"

using namespace std;

enum class LiDARType : int {
    Unknown = 0,
    Solid   = 1,   // 固态（如 Livox）
    Mech    = 2    // 机械式多线
};

class DataPreprocess
{
public:
    // 改成带线号的点云
    pcl::PointCloud<Common::Point>::Ptr cloud_input_;
    cv::Mat img_input_;
    LiDARType lidar_type_{LiDARType::Unknown};
    LiDARType lidarType() const { return lidar_type_; }

    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<Common::Point>)
    {
        string bag_path   = params.bag_path;
        string image_path = params.image_path;
        string lidar_topic = params.lidar_topic;

        // 读图像
        img_input_ = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (img_input_.empty())
        {
            std::string msg = "Loading the image " + image_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }

        // 先检查包是否存在
        std::fstream file_;
        file_.open(bag_path, ios::in);
        if (!file_)
        {
            std::string msg = "Loading the rosbag " + bag_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }
        ROS_INFO("Loading the rosbag %s", bag_path.c_str());

        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException &e) {
            ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
            return;
        }

        std::vector<string> lidar_topic_vec = {lidar_topic};
        rosbag::View view(bag, rosbag::TopicQuery(lidar_topic_vec));

        // 累计读取
        for (const rosbag::MessageInstance &m : view)
        {
            // 1) Livox 自定义消息（含 line 字段）
            if (auto livox_custom_msg = m.instantiate<livox_ros_driver::CustomMsg>())
            {
                lidar_type_ = LiDARType::Solid;
                cloud_input_->reserve(livox_custom_msg->point_num);
                for (uint32_t i = 0; i < livox_custom_msg->point_num; ++i)
                {
                    Common::Point p;
                    p.x = livox_custom_msg->points[i].x;
                    p.y = livox_custom_msg->points[i].y;
                    p.z = livox_custom_msg->points[i].z;
                    // Livox 的 CustomPoint 有 line 字段（uint8 / uint16 视版本而定）
                    p.ring = static_cast<std::uint16_t>(livox_custom_msg->points[i].line);
                    cloud_input_->push_back(p);
                }
                continue;
            }

            // 2) 机械雷达 / 通用 PointCloud2
            if (auto pcl_msg = m.instantiate<sensor_msgs::PointCloud2>())
            {
                // 优先判断是否有 ring 字段
                bool has_ring = false;
                for (const auto &f : pcl_msg->fields)
                {
                    if (f.name == "ring") { has_ring = true; break; }
                }

                // 使用 iterator 安全读取
                sensor_msgs::PointCloud2ConstIterator<float> it_x(*pcl_msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> it_y(*pcl_msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> it_z(*pcl_msg, "z");

                // ring 可能不存在：不存在时用 0xFFFF 表示未知
                std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<std::uint16_t>> it_ring_ptr;
                if (has_ring)
                {
                    it_ring_ptr.reset(new sensor_msgs::PointCloud2ConstIterator<std::uint16_t>(*pcl_msg, "ring"));
                    lidar_type_ = LiDARType::Mech;
                }
                else
                {
                    lidar_type_ = LiDARType::Solid;
                }

                const size_t n = static_cast<size_t>(pcl_msg->width) * pcl_msg->height;
                cloud_input_->reserve(n);

                // cout << "Loading PointCloud2 with " << n << " points. Has ring: " << has_ring << endl;

                for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z)
                {
                    Common::Point p;
                    p.x = *it_x;
                    p.y = *it_y;
                    p.z = *it_z;

                    if (has_ring)
                    {
                        // 解引用 ring 迭代器并前进
                        p.ring = **it_ring_ptr;
                        ++(*it_ring_ptr);
                        // if (i % 32 == 0) cout << "ring: " << p.ring << endl;
                        // if (i % 32 == 1) cout << "ring: " << p.ring << endl;
                    }
                    else
                    {
                        p.ring = 0xFFFF; // 未知线号
                    }

                    cloud_input_->push_back(p);
                }
                continue;
            }

            // 其他类型忽略
        }

        ROS_INFO("Loaded %zu points from the rosbag.", cloud_input_->size());
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif // DATA_PREPROCESS_HPP