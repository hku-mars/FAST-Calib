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

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

class DataPreprocess
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_;

    cv::Mat img_input_;

    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        string bag_path = params.bag_path;
        string image_path = params.image_path;
        string lidar_topic = params.lidar_topic;
        string camera_topic = params.camera_topic;

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

        img_input_ = cv::imread(params.image_path, cv::IMREAD_UNCHANGED);
        if (img_input_.empty()) 
        {

            std::string msg = "Loading the image " + image_path + " failed";
            ROS_INFO(msg.c_str());
            ROS_INFO("Attempting to extract image from rosbag...");

            bool image_found = false;

            if (camera_topic.empty()) {
                ROS_ERROR("Camera topic is empty. Please provide a valid camera topic to extract image from rosbag.");
                return;
            }

            std::vector<string> camera_topic_vec = {camera_topic};
            rosbag::View camera_view(bag, rosbag::TopicQuery(camera_topic_vec));

            for (const rosbag::MessageInstance &m : camera_view) {
                if (m.getDataType() == "sensor_msgs/Image") {
                    auto img_msg = m.instantiate<sensor_msgs::Image>();
                    try {
                        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                        img_input_ = cv_ptr->image;
                        image_found = true;
                        ROS_INFO("Successfully extracted image from specified camera topic");
                        break;
                    } catch (cv_bridge::Exception &e) {
                        ROS_ERROR("cv_bridge exception: %s", e.what());
                    }
                } else if (m.getDataType() == "sensor_msgs/CompressedImage") {
                    auto img_msg = m.instantiate<sensor_msgs::CompressedImage>();
                    try {
                        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                        img_input_ = cv_ptr->image;
                        image_found = true;
                        ROS_INFO("Successfully extracted compressed image from specified camera topic");
                        break;
                    } catch (cv_bridge::Exception &e) {
                        ROS_ERROR("cv_bridge exception: %s", e.what());
                    }
                }
            }
            if (!image_found) {
                ROS_ERROR("No valid image messages found in the specified camera topic.");
                return;
            }        
        }
        

        std::vector<string> lidar_topic_vec = {lidar_topic};
        rosbag::View view(bag, rosbag::TopicQuery(lidar_topic_vec));

        for (const rosbag::MessageInstance &m : view) 
        {
            // Determine if the message is a Livox custom message
            
            auto livox_custom_msg = m.instantiate<livox_ros_driver::CustomMsg>();
            if (livox_custom_msg) 
            {
                // Handle Livox custom message
                cloud_input_->reserve(livox_custom_msg->point_num);
                for (uint i = 0; i < livox_custom_msg->point_num; ++i) 
                {
                    pcl::PointXYZ p;
                    p.x = livox_custom_msg->points[i].x;
                    p.y = livox_custom_msg->points[i].y;
                    p.z = livox_custom_msg->points[i].z;
                    cloud_input_->points.push_back(p);
                }
            }
            else 
            {
                // Handle PCL format (Livox and Mechanical LiDAR)
                auto pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
                pcl::PointCloud<pcl::PointXYZ> temp_cloud;
                pcl::fromROSMsg(*pcl_msg, temp_cloud);
                *cloud_input_ += temp_cloud;
            } 
        }
        ROS_INFO("Loaded %ld points from the rosbag.", cloud_input_->size()); 
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif