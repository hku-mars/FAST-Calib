
#include <ros/ros.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#include "multi_process.hpp"
#include "common_lib.h"
#include "data_preprocess.hpp"

using Eigen::Vector3d;

static bool parseCentersLine(const std::string& line, std::vector<Vector3d>& out_pts)
{
    // 支持形如：lidar_centers: {x,y,z} {x,y,z} {x,y,z} {x,y,z}
    // 或 qr_centers: {x,y,z} {x,y,z} ...
    std::regex brace_re("\\{([^\\}]*)\\}");
    auto begin = std::sregex_iterator(line.begin(), line.end(), brace_re);
    auto end   = std::sregex_iterator();

    out_pts.clear();
    for (auto it = begin; it != end; ++it) {
        std::string xyz = (*it)[1]; // "x,y,z"
        // 去空格
        xyz.erase(remove_if(xyz.begin(), xyz.end(), ::isspace), xyz.end());
        // 用逗号分割
        std::vector<double> vals;
        std::stringstream ss(xyz);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
        try {
            vals.push_back(std::stod(tok));
        } catch (...) { return false; }
        }
        if (vals.size() != 3) return false;
        out_pts.emplace_back(vals[0], vals[1], vals[2]);
    }
    return !out_pts.empty();
}

struct Block {
  std::string time_line;
  std::vector<Vector3d> lidar_pts; // 4
  std::vector<Vector3d> qr_pts;    // 4
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "multi_fast_calib");
    ros::NodeHandle nh;
    Params params = loadParameters(nh);

    DataPreprocessPtr dataPreprocessPtr;
    dataPreprocessPtr.reset(new DataPreprocess(params));
    cv::Mat img_input = dataPreprocessPtr->img_input_;

    if (params.midresult_path.back() != '/') params.midresult_path += '/';
    std::string midtxt_path = params.midresult_path + "circle_center_record.txt";

    if (params.output_path.back() != '/') params.output_path += '/';
    std::string multi_output_path = params.output_path + "multi_calib_result.txt";

    // 读取全部行
    std::ifstream fin(midtxt_path);
    if (!fin.is_open())
    {
        ROS_ERROR("Failed to open txt file: %s", midtxt_path.c_str());
        return 1;
    }
    std::vector<std::string> lines;
    for (std::string line; std::getline(fin, line); )
    {
        if (!line.empty()) lines.push_back(line);
    }
    fin.close();
    if (lines.size() < 9) {
        ROS_ERROR("File has fewer than 9 lines, cannot get 3 blocks.");
        return 1;
    }

    // 解析所有 block（按三行一组：time + lidar_centers + qr_centers）
    std::vector<Block> blocks;
    for (size_t i = 0; i + 2 < lines.size(); ++i) 
    {
        if (lines[i].rfind("time:", 0) == 0 &&
            lines[i+1].find("lidar_centers:") != std::string::npos &&
            lines[i+2].find("qr_centers:")    != std::string::npos) 
        {
            Block b;
            b.time_line = lines[i];

            if (!parseCentersLine(lines[i+1], b.lidar_pts)) continue;
            if (!parseCentersLine(lines[i+2], b.qr_pts))    continue;
            // 要求每组正好4个
            if (b.lidar_pts.size() == 4 && b.qr_pts.size() == 4) 
            {
                blocks.push_back(std::move(b));
                i += 2; // 跳过这个block
            }
        }
    }
    if (blocks.size() < 3) 
    {
        ROS_ERROR("Parsed blocks < 3 (got %zu).", blocks.size());
        return 1;
    }

    // 取最后3个 block
    std::vector<Vector3d> L, C;
    for (size_t k = blocks.size() - 3; k < blocks.size(); ++k) 
    {
        const auto& b = blocks[k];
        // 依次拼入，保持顺序一致
        for (int i = 0; i < 4; ++i) 
        {
            L.push_back(b.lidar_pts[i]);
            C.push_back(b.qr_pts[i]);
        }
    }
    if (L.size() != 12 || C.size() != 12) {
        ROS_ERROR("Merged pairs not equal to 12 (L=%zu, C=%zu).", L.size(), C.size());
        return 1;
    }


    std::cout << "Lidar centers (L):" << std::endl;
    for (size_t i = 0; i < L.size(); ++i) {
        std::cout << "L[" << i << "]: (" << L[i](0) << ", " << L[i](1) << ", " << L[i](2) << ")" << std::endl;
    }
    std::cout << "QR centers (C):" << std::endl;
    for (size_t i = 0; i < C.size(); ++i) {
        std::cout << "C[" << i << "]: (" << C[i](0) << ", " << C[i](1) << ", " << C[i](2) << ")" << std::endl;
    }


    // 一次性求解
    auto res = SolveRigidTransformWeighted(L, C, nullptr);
    if (!res.ok) {
        ROS_ERROR("SolveRigidTransformWeighted failed.");
        return 1;
    }

    // 打印 / 保存
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = res.R;
    T.block<3,1>(0,3) = res.t;

    std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
      << res.rms << " m" << RESET << std::endl;

    std::cout << BOLDYELLOW << "[Result] multi_scene_calib extrinsic parameters T_cam_lidar: " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << T << RESET << std::endl;

    std::ofstream fout(multi_output_path);
    if (fout.is_open()) 
    {
        fout << "# FAST-LIVO2 calibration format\n";
        fout << "cam_model: Pinhole\n";
        fout << "cam_width: " << img_input.cols << "\n";
        fout << "cam_height: " << img_input.rows << "\n";
        fout << "scale: 1.0\n";
        fout << "cam_fx: " << params.fx << "\n";
        fout << "cam_fy: " << params.fy << "\n";
        fout << "cam_cx: " << params.cx << "\n";
        fout << "cam_cy: " << params.cy << "\n";
        fout << "cam_d0: " << params.k1 << "\n";
        fout << "cam_d1: " << params.k2 << "\n";
        fout << "cam_d2: " << params.p1 << "\n";
        fout << "cam_d3: " << params.p2 << "\n";

        fout << std::fixed << std::setprecision(6);
        fout << "Rcl: [ "
            << std::setw(9) << res.R(0,0) << ", " << std::setw(9) << res.R(0,1) << ", " << std::setw(9) << res.R(0,2) << ",\n"
            << "      " << std::setw(9) << res.R(1,0) << ", " << std::setw(9) << res.R(1,1) << ", " << std::setw(9) << res.R(1,2) << ",\n"
            << "      " << std::setw(9) << res.R(2,0) << ", " << std::setw(9) << res.R(2,1) << ", " << std::setw(9) << res.R(2,2) << "]\n";
        fout << "Pcl: [ "
            << std::setw(9) << res.t(0) << ", " << std::setw(9) << res.t(1) << ", " << std::setw(9) << res.t(2) << "]\n";
        fout.close();
        std::cout << BOLDYELLOW << "[Result] Saved multi_scene calibration results to " << BOLDWHITE << multi_output_path << RESET << std::endl;
    } else {
        ROS_WARN("Failed to write out file: %s", multi_output_path.c_str());
    }

    return 0;
}
