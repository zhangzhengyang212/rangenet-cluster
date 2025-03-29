#include "../include/bounding_box.hpp"  // 使用相对路径包含
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>
#include <Eigen/Dense>

// 确保与头文件中的类型定义一致
using PointType = pcl::PointXYZI;

// 构造函数
BoundingBoxGenerator::BoundingBoxGenerator(
    double cluster_tolerance, 
    int min_cluster_size, 
    int max_cluster_size) 
    : vehicle_cluster_tolerance_(cluster_tolerance),
      vehicle_min_cluster_size_(min_cluster_size),
      vehicle_max_cluster_size_(max_cluster_size),
      pedestrian_cluster_tolerance_(cluster_tolerance),
      pedestrian_min_cluster_size_(min_cluster_size),
      pedestrian_max_cluster_size_(max_cluster_size),
      current_frame_cloud(new pcl::PointCloud<pcl::PointXYZRGB>) {
        
  ROS_INFO("BoundingBoxGenerator initialized with:");
  ROS_INFO("  - Cluster tolerance: %.2f m", cluster_tolerance);
  ROS_INFO("  - Min cluster size: %d points", min_cluster_size);
  ROS_INFO("  - Max cluster size: %d points", max_cluster_size);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr BoundingBoxGenerator::processBoundingBoxes(
  const pcl::PointCloud<PointType>& input_cloud,
  const int* labels) {

  // 空指针检查
  if (labels == nullptr) {
    ROS_ERROR("Labels pointer is null");
    return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  }

  // 创建结果点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  result_cloud->points.resize(input_cloud.points.size());
  result_cloud->width = input_cloud.width;
  result_cloud->height = input_cloud.height;

  ROS_INFO("Processing %zu points", input_cloud.points.size());

  // 统计标签频率
  std::map<int, int> label_counts;
  for (size_t i = 0; i < input_cloud.points.size(); ++i) {
    label_counts[labels[i]]++;
  }

  // 输出标签统计
  ROS_INFO("Label distribution:");
  for (const auto& pair : label_counts) {
    ROS_INFO("  Label %d: %d points", pair.first, pair.second);
  }

  // 更新车辆和行人标签
  const std::vector<int> vehicle_labels = {1, 2, 3, 4, 5}; // 车辆包括小车、卡车和其他可能的车辆
  const std::vector<int> pedestrian_labels = {6, 7}; // 行人包括普通行人和骑自行车的行人

  // 更新颜色映射
  std::map<int, std::array<uint8_t, 3>> color_map;
  color_map[1] = {255, 0, 0};         // 红色 - 车辆
  color_map[2] = {255, 0, 0};         // 红色 - 
  color_map[3] = {255, 0, 0};         // 红色 - 
  color_map[4] = {255, 0, 0};         // 红色 - 
  color_map[5] = {255, 0, 0};         // 红色 - 卡车
  color_map[6] = {255, 0, 255};       // 紫色 - 行人
  color_map[7] = {180, 0, 255};       // 紫色 - 骑自行车的行人
  /*color_map[9] = {128, 128, 128};     // 灰色 - 地面/道路
  color_map[10] = {245, 180, 230};    // 淡紫色 - 停车位
  color_map[11] = {75, 0, 130};      // 深紫色 - 人行道 
  color_map[12] = {255, 20, 147};     // 深粉色 - 
  color_map[13] = {218, 165, 32};     // 深黄色/金菊黄 - 墙
  color_map[14] = {101, 67, 33};      // 深棕色 - 栅栏 
  color_map[15] = {0, 255, 0};        // 绿色 - 植被
  color_map[16] = {139, 69, 19};      // 棕褐色/马鞍棕色 - 树干
  color_map[17] = {34, 139, 34};      // 森林绿 - 草地/地形
  color_map[18] = {255, 215, 0};      // 金色 - 路灯 */

  // 安全地为每个点设置颜色
  for (size_t i = 0; i < input_cloud.points.size(); ++i) {
    pcl::PointXYZRGB& colored_point = result_cloud->points[i];
    const auto& input_point = input_cloud.points[i];
    
    // 复制坐标
    colored_point.x = input_point.x;
    colored_point.y = input_point.y;
    colored_point.z = input_point.z;
    
    // 获取当前点的标签
    int label = labels[i];
    
    // 使用颜色映射
    if (color_map.find(label) != color_map.end()) {
      const auto& color = color_map[label];
      colored_point.r = color[0];
      colored_point.g = color[1];
      colored_point.b = color[2];
    } else {
      // 默认颜色 - 白色
      colored_point.r = 255;
      colored_point.g = 255;
      colored_point.b = 255;
    }
  }

  // 输出检测结果
  for (const auto& label : vehicle_labels) {
    if (label_counts.find(label) != label_counts.end() && label_counts[label] > 0) {
      ROS_INFO("Detected vehicle type %d: %d points", label, label_counts[label]);
    }
  }

  for (const auto& label : pedestrian_labels) {
    if (label_counts.find(label) != label_counts.end() && label_counts[label] > 0) {
      ROS_INFO("Detected pedestrian type %d: %d points", label, label_counts[label]);
    }
  }

  // 发布边界框
  publishBoundingBoxes(result_cloud, labels, "velodyne");

  ROS_INFO("Processed all points successfully");
  return result_cloud;
}

void BoundingBoxGenerator::publishBoundingBoxes(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const int* labels,
  const std::string& frame_id) {
  
// 创建ROS节点句柄
ros::NodeHandle nh;

// 创建边界框发布者
ros::Publisher vehicle_box_pub = nh.advertise<visualization_msgs::MarkerArray>("vehicle_boxes", 1);
ros::Publisher pedestrian_box_pub = nh.advertise<visualization_msgs::MarkerArray>("pedestrian_boxes", 1);
// 创建点云补全发布者
static ros::Publisher cloud_completion_pub = nh.advertise<sensor_msgs::PointCloud2>("/vehicle_completion/point_cloud", 1);

// 清空当前帧的点云
current_frame_cloud->points.clear();

// 定义车辆和行人标签
const std::vector<int> vehicle_labels = {1, 2, 3, 4, 5};
const std::vector<int> pedestrian_labels = {6, 7};

// 提取车辆和行人点的索引
std::vector<int> vehicle_indices;
std::vector<int> pedestrian_indices;

for (size_t i = 0; i < cloud->points.size(); ++i) {
  int label = labels[i];
  
  // 检查是否是车辆标签
  if (std::find(vehicle_labels.begin(), vehicle_labels.end(), label) != vehicle_labels.end()) {
    vehicle_indices.push_back(i);
  }
  
  // 检查是否是行人标签
  if (std::find(pedestrian_labels.begin(), pedestrian_labels.end(), label) != pedestrian_labels.end()) {
    pedestrian_indices.push_back(i);
  }
}

ROS_INFO("Found %zu vehicle points and %zu pedestrian points", 
         vehicle_indices.size(), pedestrian_indices.size());

// 如果没有检测到物体，清除所有边界框并返回
if (vehicle_indices.empty() && pedestrian_indices.empty()) {
  ROS_WARN("No vehicles or pedestrians detected");
  
  // 发布空边界框消息来清除先前的边界框
  visualization_msgs::MarkerArray empty_vehicle_boxes;
  visualization_msgs::MarkerArray empty_pedestrian_boxes;
  
  // 添加一个删除所有之前边界框的标记
  visualization_msgs::Marker delete_all_marker;
  delete_all_marker.header.frame_id = frame_id;
  delete_all_marker.header.stamp = ros::Time::now();
  delete_all_marker.ns = "vehicle_obb";  // 与车辆OBB的命名空间匹配
  delete_all_marker.action = visualization_msgs::Marker::DELETEALL;
  empty_vehicle_boxes.markers.push_back(delete_all_marker);
  
  delete_all_marker.ns = "vehicle_direction";  // 与车辆方向箭头的命名空间匹配
  empty_vehicle_boxes.markers.push_back(delete_all_marker);
  
  delete_all_marker.ns = "vehicle_lines";  // 与车辆线框的命名空间匹配
  empty_vehicle_boxes.markers.push_back(delete_all_marker);
  
  delete_all_marker.ns = "pedestrian_obb";  // 与行人OBB的命名空间匹配
  empty_pedestrian_boxes.markers.push_back(delete_all_marker);
  
  delete_all_marker.ns = "pedestrian_direction";  // 与行人方向箭头的命名空间匹配
  empty_pedestrian_boxes.markers.push_back(delete_all_marker);
  
  delete_all_marker.ns = "pedestrian_lines";  // 与行人线框的命名空间匹配
  empty_pedestrian_boxes.markers.push_back(delete_all_marker);
  
  vehicle_box_pub.publish(empty_vehicle_boxes);
  pedestrian_box_pub.publish(empty_pedestrian_boxes);
  
  // 同时清除点云补全
  if (cloud_completion_pub.getNumSubscribers() > 0) {
    // 创建一个空的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr empty_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    empty_cloud->points.clear();
    empty_cloud->width = 0;
    empty_cloud->height = 0;
    
    // 转换为ROS消息
    sensor_msgs::PointCloud2 empty_cloud_msg;
    pcl::toROSMsg(*empty_cloud, empty_cloud_msg);
    empty_cloud_msg.header.frame_id = frame_id;
    empty_cloud_msg.header.stamp = ros::Time::now();
    
    // 发布空点云以清除旧的点云
    cloud_completion_pub.publish(empty_cloud_msg);
    ROS_INFO("清除了补全点云显示");
  }
  
  return;
}

// 对车辆和行人点进行聚类
std::vector<pcl::PointIndices> vehicle_clusters;
std::vector<pcl::PointIndices> pedestrian_clusters;

if (!vehicle_indices.empty()) {
  vehicle_clusters = performClustering(cloud, vehicle_indices, vehicle_cluster_tolerance_);
  ROS_INFO("Clustered vehicles: %zu clusters", vehicle_clusters.size());
}

if (!pedestrian_indices.empty()) {
  pedestrian_clusters = performClustering(cloud, pedestrian_indices, pedestrian_cluster_tolerance_);
  ROS_INFO("Clustered pedestrians: %zu clusters", pedestrian_clusters.size());
}

// 生成边界框
visualization_msgs::MarkerArray vehicle_boxes;
visualization_msgs::MarkerArray pedestrian_boxes;

// 先添加一个删除所有先前标记的指令
visualization_msgs::Marker delete_vehicle_markers;
delete_vehicle_markers.header.frame_id = frame_id;
delete_vehicle_markers.header.stamp = ros::Time::now();
delete_vehicle_markers.ns = "vehicle_obb";
delete_vehicle_markers.action = visualization_msgs::Marker::DELETEALL;
vehicle_boxes.markers.push_back(delete_vehicle_markers);

delete_vehicle_markers.ns = "vehicle_direction";
vehicle_boxes.markers.push_back(delete_vehicle_markers);

delete_vehicle_markers.ns = "vehicle_lines";
vehicle_boxes.markers.push_back(delete_vehicle_markers);

visualization_msgs::Marker delete_pedestrian_markers;
delete_pedestrian_markers.header.frame_id = frame_id;
delete_pedestrian_markers.header.stamp = ros::Time::now();
delete_pedestrian_markers.ns = "pedestrian_obb";
delete_pedestrian_markers.action = visualization_msgs::Marker::DELETEALL;
pedestrian_boxes.markers.push_back(delete_pedestrian_markers);

delete_pedestrian_markers.ns = "pedestrian_direction";
pedestrian_boxes.markers.push_back(delete_pedestrian_markers);

delete_pedestrian_markers.ns = "pedestrian_lines";
pedestrian_boxes.markers.push_back(delete_pedestrian_markers);

if (!vehicle_clusters.empty()) {
  // 检测雷达是否在转弯
  bool is_turning = isLidarTurning(cloud, vehicle_indices);
  
  if (is_turning) {
    // 转弯时使用定向边界框(OBB)，并清除点云显示
    ROS_INFO("Lidar is turning, using oriented bounding boxes");
    
    // 清空当前帧点云，避免显示上一帧AABB方法的补全点云
    current_frame_cloud->points.clear();
    current_frame_cloud->width = 0;
    current_frame_cloud->height = 0;
    
    // 发布空点云以清除显示
    sensor_msgs::PointCloud2 empty_cloud_msg;
    pcl::toROSMsg(*current_frame_cloud, empty_cloud_msg);
    empty_cloud_msg.header.frame_id = frame_id;
    empty_cloud_msg.header.stamp = ros::Time::now();
    cloud_completion_pub.publish(empty_cloud_msg);
    
    ROS_INFO("转弯中，清除点云补全显示");
    
    // 生成OBB边界框
    visualization_msgs::MarkerArray obb_markers = generateOrientedBoundingBoxes(cloud, vehicle_clusters, frame_id, "vehicle", 1);
    vehicle_boxes.markers.insert(vehicle_boxes.markers.end(), obb_markers.markers.begin(), obb_markers.markers.end());
  } else {
    // 直线行驶时使用轴对齐边界框(AABB)
    ROS_INFO("Lidar is moving straight, using axis-aligned bounding boxes");
    visualization_msgs::MarkerArray aabb_markers = generateBoundingBoxes(cloud, vehicle_clusters, frame_id, "vehicle", 1);
    vehicle_boxes.markers.insert(vehicle_boxes.markers.end(), aabb_markers.markers.begin(), aabb_markers.markers.end());
  }
  vehicle_box_pub.publish(vehicle_boxes);
}

if (!pedestrian_clusters.empty()) {
  // 行人始终使用轴对齐边界框(AABB)
  visualization_msgs::MarkerArray ped_markers = generateBoundingBoxes(cloud, pedestrian_clusters, frame_id, "pedestrian", 2);
  pedestrian_boxes.markers.insert(pedestrian_boxes.markers.end(), ped_markers.markers.begin(), ped_markers.markers.end());
  pedestrian_box_pub.publish(pedestrian_boxes);
}

// 发布合并的补全点云
if (current_frame_cloud->points.size() > 0) {
  // 设置点云属性
  current_frame_cloud->width = current_frame_cloud->points.size();
  current_frame_cloud->height = 1;
  current_frame_cloud->is_dense = true;
  
  // 转换到ROS消息
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*current_frame_cloud, cloud_msg);
  cloud_msg.header.frame_id = frame_id;
  cloud_msg.header.stamp = ros::Time::now();
  
  // 发布点云消息
  cloud_completion_pub.publish(cloud_msg);
  
  ROS_INFO("发布了合并的补全点云，包含 %zu 个点", current_frame_cloud->points.size());
}
}

std::vector<pcl::PointIndices> BoundingBoxGenerator::performClustering(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const std::vector<int>& object_indices,
  float cluster_tolerance) {
  
// 创建用于欧氏聚类的搜索树
pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
tree->setInputCloud(cloud);

// 创建点索引集合
pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices);
indices_ptr->indices = object_indices;

// 设置欧氏聚类提取器
pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
ec.setClusterTolerance(cluster_tolerance); // 设置聚类容差
ec.setMinClusterSize(200);                  // 最小聚类大小
ec.setMaxClusterSize(25000);               // 最大聚类大小
ec.setSearchMethod(tree);
ec.setInputCloud(cloud);
ec.setIndices(indices_ptr);

// 执行聚类
std::vector<pcl::PointIndices> cluster_indices;
ec.extract(cluster_indices);

return cluster_indices;
}

// 计算定向边界框(OBB)，解决车辆朝向问题
visualization_msgs::MarkerArray BoundingBoxGenerator::generateOrientedBoundingBoxes(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const std::vector<pcl::PointIndices>& clusters,
  const std::string& frame_id,
  const std::string& ns,
  int object_type) {
  
  visualization_msgs::MarkerArray marker_array;

  for (size_t i = 0; i < clusters.size(); ++i) {
    // 提取当前聚类
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& idx : clusters[i].indices) {
      cluster_cloud->points.push_back(cloud->points[idx]);
    }
    cluster_cloud->width = cluster_cloud->points.size();
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = true;
    
    // 跳过过小的聚类
    if (cluster_cloud->points.size() < 10) {
      continue;
    }
    
    // 创建2D点云，仅使用x和y坐标，忽略z坐标
    // 这样更好地提取水平方向的主要信息，尤其对于车辆这类在水平面上延展的物体
    int numPoints = cluster_cloud->points.size();
    Eigen::MatrixXd dataMatrix(2, numPoints);
    
    // 计算质心 - 更精确的方法
    Eigen::Vector4f centroid4f;
    pcl::compute3DCentroid(*cluster_cloud, centroid4f);
    Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);
    
    // 提取所有点的2D坐标
    for (int j = 0; j < numPoints; ++j) {
      dataMatrix(0, j) = cluster_cloud->points[j].x;
      dataMatrix(1, j) = cluster_cloud->points[j].y;
    }
    
    // 计算均值
    Eigen::Vector2d meanVector = dataMatrix.rowwise().mean();
    
    // 中心化数据
    dataMatrix.colwise() -= meanVector;
    
    // 使用SVD (奇异值分解)来找主方向 - 这是PCA的一种实现方式
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(dataMatrix, Eigen::ComputeThinU);
    
    // 获取主方向向量(第一个左奇异向量)
    Eigen::Vector2d mainSingularVector = svd.matrixU().col(0);
    
    // 计算旋转角度
    double rot_angle = std::atan2(mainSingularVector(1), mainSingularVector(0));
    
    // 创建旋转矩阵
    double cosAngle = std::cos(rot_angle);
    double sinAngle = std::sin(rot_angle);
    Eigen::Matrix2d rot_2d;
    rot_2d << cosAngle, -sinAngle,
              sinAngle, cosAngle;
    
    // 将2D点旋转到主方向坐标系
    std::vector<Eigen::Vector2d> rotated_points;
    for (int j = 0; j < numPoints; ++j) {
      Eigen::Vector2d pt(cluster_cloud->points[j].x, cluster_cloud->points[j].y);
      Eigen::Vector2d rotated_pt = rot_2d.transpose() * (pt - Eigen::Vector2d(meanVector));
      rotated_points.push_back(rotated_pt);
    }
    
    // 计算Z轴范围
    double min_z = cluster_cloud->points[0].z;
    double max_z = cluster_cloud->points[0].z;
    for (const auto& pt : cluster_cloud->points) {
      min_z = std::min(min_z, static_cast<double>(pt.z));
      max_z = std::max(max_z, static_cast<double>(pt.z));
    }
    double box_height = max_z - min_z;
    
    // 在旋转后坐标系中计算AABB (axis-aligned bounding box)
    double min_x = rotated_points[0].x();
    double max_x = rotated_points[0].x();
    double min_y = rotated_points[0].y();
    double max_y = rotated_points[0].y();
    
    for (const auto& p : rotated_points) {
      min_x = std::min(min_x, p.x());
      max_x = std::max(max_x, p.x());
      min_y = std::min(min_y, p.y());
      max_y = std::max(max_y, p.y());
    }
    
    // 边界框尺寸计算，加上余量
    const float margin = 0.15f; // 15厘米的边距，增大边距确保包含所有点
    float size_x = max_x - min_x + margin;
    float size_y = max_y - min_y + margin;
    float size_z = box_height + margin;
    
    // 确保尺寸合理
    size_x = std::max(size_x, 0.3f); // 至少30厘米宽
    size_y = std::max(size_y, 0.3f); // 至少30厘米长
    size_z = std::max(size_z, 0.3f); // 至少30厘米高
    
    // 计算边界框中心点在旋转坐标系中的位置
    Eigen::Vector2d box_center_2d_rot(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
    
    // 将中心点转回原始坐标系
    Eigen::Vector2d box_center_2d = rot_2d * box_center_2d_rot + Eigen::Vector2d(meanVector);
    
    // 计算四个角点在旋转坐标系中的位置
    Eigen::Vector2d corner_tl_rot(box_center_2d_rot.x() - 0.5 * size_x, box_center_2d_rot.y() + 0.5 * size_y);
    Eigen::Vector2d corner_tr_rot(box_center_2d_rot.x() + 0.5 * size_x, box_center_2d_rot.y() + 0.5 * size_y);
    Eigen::Vector2d corner_br_rot(box_center_2d_rot.x() + 0.5 * size_x, box_center_2d_rot.y() - 0.5 * size_y);
    Eigen::Vector2d corner_bl_rot(box_center_2d_rot.x() - 0.5 * size_x, box_center_2d_rot.y() - 0.5 * size_y);
    
    // 将角点转回原始坐标系
    Eigen::Vector2d corner_tl = rot_2d * corner_tl_rot + Eigen::Vector2d(meanVector);
    Eigen::Vector2d corner_tr = rot_2d * corner_tr_rot + Eigen::Vector2d(meanVector);
    Eigen::Vector2d corner_br = rot_2d * corner_br_rot + Eigen::Vector2d(meanVector);
    Eigen::Vector2d corner_bl = rot_2d * corner_bl_rot + Eigen::Vector2d(meanVector);
    
    // 对于车辆，检查长宽比，确保长边是车辆的长度方向
    if (object_type == 1) {
      if (size_x < size_y) {
        // 交换尺寸
        std::swap(size_x, size_y);
        
        // 重新计算角点，交换了角色，所以角点也要重新计算
        corner_tl_rot = Eigen::Vector2d(box_center_2d_rot.x() - 0.5 * size_x, box_center_2d_rot.y() + 0.5 * size_y);
        corner_tr_rot = Eigen::Vector2d(box_center_2d_rot.x() + 0.5 * size_x, box_center_2d_rot.y() + 0.5 * size_y);
        corner_br_rot = Eigen::Vector2d(box_center_2d_rot.x() + 0.5 * size_x, box_center_2d_rot.y() - 0.5 * size_y);
        corner_bl_rot = Eigen::Vector2d(box_center_2d_rot.x() - 0.5 * size_x, box_center_2d_rot.y() - 0.5 * size_y);
        
        // 更新角点
        corner_tl = rot_2d * corner_tl_rot + Eigen::Vector2d(meanVector);
        corner_tr = rot_2d * corner_tr_rot + Eigen::Vector2d(meanVector);
        corner_br = rot_2d * corner_br_rot + Eigen::Vector2d(meanVector);
        corner_bl = rot_2d * corner_bl_rot + Eigen::Vector2d(meanVector);
        
        // 使用RANSAC进一步细化主方向
        if (size_x > 1.0f && size_y > 0.5f) { // 只对足够大的车辆应用
          // 计算从边界框中心到车前方向的向量
          Eigen::Vector2d front_dir = (corner_tr + corner_br) / 2.0 - Eigen::Vector2d(box_center_2d);
          front_dir.normalize();
          
          // 获取向前沿中心线上的点
          std::vector<int> front_indices;
          for (int j = 0; j < numPoints; ++j) {
            Eigen::Vector2d pt(cluster_cloud->points[j].x, cluster_cloud->points[j].y);
            Eigen::Vector2d vec = pt - Eigen::Vector2d(box_center_2d);
            
            // 如果点和前向方向的点积为正，说明点在车辆前向
            if (vec.dot(front_dir) > 0) {
              front_indices.push_back(j);
            }
          }
          
          // 如果有足够多的前向点，重新计算前向方向
          if (front_indices.size() > 10) {
            Eigen::MatrixXd frontMatrix(2, front_indices.size());
            for (size_t j = 0; j < front_indices.size(); ++j) {
              frontMatrix(0, j) = cluster_cloud->points[front_indices[j]].x;
              frontMatrix(1, j) = cluster_cloud->points[front_indices[j]].y;
            }
            
            // 计算前向点的均值
            Eigen::Vector2d frontMean = frontMatrix.rowwise().mean();
            
            // 用前向点的中心更新边界框前向方向
            Eigen::Vector2d new_front_dir = frontMean - Eigen::Vector2d(box_center_2d);
            if (new_front_dir.norm() > 0.1) {
              new_front_dir.normalize();
              
              // 使用新前向方向重新计算旋转角度
              double new_rot_angle = std::atan2(new_front_dir.y(), new_front_dir.x());
              
              // 更新旋转矩阵
              cosAngle = std::cos(new_rot_angle);
              sinAngle = std::sin(new_rot_angle);
              rot_2d << cosAngle, -sinAngle,
                        sinAngle, cosAngle;
            }
          }
        }
      }
    }
    
    // 对车辆类型执行点云补全
    // 检查是否是车辆类型（标签1-5都是车辆）
    const std::vector<int> vehicle_types = {1, 2, 3, 4, 5};
    bool is_vehicle = std::find(vehicle_types.begin(), vehicle_types.end(), object_type) != vehicle_types.end();
    if (is_vehicle && cluster_cloud->points.size() >= 30) { // 使用原始阈值30
      // 计算点云中心点
      Eigen::Vector3f centroid(0, 0, 0);
      for (const auto& point : cluster_cloud->points) {
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
      }
      centroid /= static_cast<float>(cluster_cloud->points.size());
      
      // 创建车辆尺寸
      Eigen::Vector3f dimensions(size_x, size_y, size_z);

      // 确定车辆方向 - 对于OBB方法，使用主方向矩阵
      Eigen::Matrix3f rotation_matrix = Eigen::Matrix3f::Identity();
      rotation_matrix.block<2, 2>(0, 0) = rot_2d.cast<float>();
      
      /* // 注释掉OBB方法的点云补全，仅使用AABB方法生成点云补全
      // 执行点云补全
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr completed_cloud = 
          completeVehiclePointCloud(cluster_cloud, centroid, dimensions, rotation_matrix);
      
      // 将补全后的点云添加到合并点云中
      for (const auto& point : *completed_cloud) {
        current_frame_cloud->points.push_back(point);
      }
      
      ROS_INFO("完成了车辆 #%zu 的点云补全", i);
      */
    }
    
    // 创建线框边界框，使用LINE_LIST类型
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = ns + "_obb";
    line_marker.id = i;
    line_marker.type = visualization_msgs::Marker::LINE_LIST;
    line_marker.action = visualization_msgs::Marker::ADD;
    
    // 设置中心点
    line_marker.pose.position.x = 0;
    line_marker.pose.position.y = 0;
    line_marker.pose.position.z = 0;
    line_marker.pose.orientation.w = 1.0; // 不旋转
    
    // 设置线宽
    line_marker.scale.x = 0.03; // 3厘米宽的线
    
    // 设置线颜色 - 使用黄色
    if (object_type == 1) { // 车辆
      line_marker.color.r = 1.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0; // 黄色
      line_marker.color.a = 1.0; // 完全不透明
    } else { // 行人
      line_marker.color.r = 1.0;
      line_marker.color.g = 0.5;
      line_marker.color.b = 0.0; // 橙色
      line_marker.color.a = 1.0; // 完全不透明
    }
    
    line_marker.lifetime = ros::Duration(1.5); // 持续显示1.5秒
    
    // 计算立方体的8个顶点
    // 顶面四个点
    geometry_msgs::Point p[8];
    
    // 底面四个点
    p[0].x = corner_bl.x(); p[0].y = corner_bl.y(); p[0].z = min_z;
    p[1].x = corner_br.x(); p[1].y = corner_br.y(); p[1].z = min_z;
    p[2].x = corner_tr.x(); p[2].y = corner_tr.y(); p[2].z = min_z;
    p[3].x = corner_tl.x(); p[3].y = corner_tl.y(); p[3].z = min_z;
    
    // 顶面四个点
    p[4].x = corner_bl.x(); p[4].y = corner_bl.y(); p[4].z = max_z;
    p[5].x = corner_br.x(); p[5].y = corner_br.y(); p[5].z = max_z;
    p[6].x = corner_tr.x(); p[6].y = corner_tr.y(); p[6].z = max_z;
    p[7].x = corner_tl.x(); p[7].y = corner_tl.y(); p[7].z = max_z;
    
    // 添加12条边
    // 底面
    line_marker.points.push_back(p[0]); line_marker.points.push_back(p[1]);
    line_marker.points.push_back(p[1]); line_marker.points.push_back(p[2]);
    line_marker.points.push_back(p[2]); line_marker.points.push_back(p[3]);
    line_marker.points.push_back(p[3]); line_marker.points.push_back(p[0]);
    
    // 顶面
    line_marker.points.push_back(p[4]); line_marker.points.push_back(p[5]);
    line_marker.points.push_back(p[5]); line_marker.points.push_back(p[6]);
    line_marker.points.push_back(p[6]); line_marker.points.push_back(p[7]);
    line_marker.points.push_back(p[7]); line_marker.points.push_back(p[4]);
    
    // 侧面
    line_marker.points.push_back(p[0]); line_marker.points.push_back(p[4]);
    line_marker.points.push_back(p[1]); line_marker.points.push_back(p[5]);
    line_marker.points.push_back(p[2]); line_marker.points.push_back(p[6]);
    line_marker.points.push_back(p[3]); line_marker.points.push_back(p[7]);
    
    marker_array.markers.push_back(line_marker);
    
    // 添加一个箭头标记表示主方向
    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = frame_id;
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = ns + "_direction";
    arrow_marker.id = i;
    arrow_marker.type = visualization_msgs::Marker::ARROW;
    arrow_marker.action = visualization_msgs::Marker::ADD;
    
    // 设置箭头起点为中心点
    arrow_marker.pose.position.x = box_center_2d.x();
    arrow_marker.pose.position.y = box_center_2d.y();
    arrow_marker.pose.position.z = min_z + box_height / 2.0;
    
    // 从2D旋转矩阵创建四元数
    Eigen::Matrix3f rot_matrix = Eigen::Matrix3f::Identity();
    rot_matrix.block<2, 2>(0, 0) = rot_2d.cast<float>();
    Eigen::Quaternionf quaternion(rot_matrix);
    
    arrow_marker.pose.orientation.x = quaternion.x();
    arrow_marker.pose.orientation.y = quaternion.y();
    arrow_marker.pose.orientation.z = quaternion.z();
    arrow_marker.pose.orientation.w = quaternion.w();
    
    // 设置箭头尺寸
    arrow_marker.scale.x = std::max(size_x, size_y) * 1.2;  // 箭头长度
    arrow_marker.scale.y = 0.1;  // 箭头宽度
    arrow_marker.scale.z = 0.1;  // 箭头高度
    
    // 设置箭头颜色
    arrow_marker.color.r = 1.0;
    arrow_marker.color.g = 0.0;
    arrow_marker.color.b = 0.0;  // 红色
    arrow_marker.color.a = 1.0;
    
    arrow_marker.lifetime = ros::Duration(1.5);
    marker_array.markers.push_back(arrow_marker);
  }

  return marker_array;
}

// 提取特定标签的点云
pcl::PointCloud<PointType>::Ptr BoundingBoxGenerator::extractLabeledPointCloud(
  const pcl::PointCloud<PointType>& cloud,
  const int* labels,
  const std::vector<int>& target_labels) {
  
pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>);

for (size_t i = 0; i < cloud.points.size(); ++i) {
  int label = labels[i];
  if (std::find(target_labels.begin(), target_labels.end(), label) != target_labels.end()) {
    result->points.push_back(cloud.points[i]);
  }
}

result->width = result->points.size();
result->height = 1;
result->is_dense = cloud.is_dense;

return result;
}

// 对点云进行欧式聚类
std::vector<pcl::PointIndices> BoundingBoxGenerator::euclideanClustering(
  const pcl::PointCloud<PointType>::Ptr& cloud,
  double tolerance,
  int min_size,
  int max_size) {
  
// 创建KD树用于搜索
pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
tree->setInputCloud(cloud);

// 欧式聚类
pcl::EuclideanClusterExtraction<PointType> ec;
ec.setClusterTolerance(tolerance);
ec.setMinClusterSize(min_size);
ec.setMaxClusterSize(max_size);
ec.setSearchMethod(tree);
ec.setInputCloud(cloud);

// 执行聚类
std::vector<pcl::PointIndices> cluster_indices;
ec.extract(cluster_indices);

return cluster_indices;
}

// 轴对齐边界框(AABB)方法，用于行人等对象
visualization_msgs::MarkerArray BoundingBoxGenerator::generateBoundingBoxes(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const std::vector<pcl::PointIndices>& clusters,
  const std::string& frame_id,
  const std::string& ns,
  int object_type) {
  
  visualization_msgs::MarkerArray marker_array;

  for (size_t i = 0; i < clusters.size(); ++i) {
    // 提取当前聚类
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& idx : clusters[i].indices) {
      cluster_cloud->points.push_back(cloud->points[idx]);
    }
    cluster_cloud->width = cluster_cloud->points.size();
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = true;
    
    // 跳过过小的聚类
    if (cluster_cloud->points.size() < 10) {
      continue;
    }
    
    // 计算边界框
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);
    
    // 计算边界框的中心点和尺寸
    float center_x = (min_pt.x + max_pt.x) / 2;
    float center_y = (min_pt.y + max_pt.y) / 2;
    float center_z = (min_pt.z + max_pt.z) / 2;
    float size_x = max_pt.x - min_pt.x;
    float size_y = max_pt.y - min_pt.y;
    float size_z = max_pt.z - min_pt.z;

    // 对车辆类型执行点云补全
    // 检查是否是车辆类型（标签1-5都是车辆）
    const std::vector<int> vehicle_types = {1, 2, 3, 4, 5};
    bool is_vehicle = std::find(vehicle_types.begin(), vehicle_types.end(), object_type) != vehicle_types.end();
    if (is_vehicle && cluster_cloud->points.size() >= 100) { // 提高最小点数阈值，从30增加到100
      // 只对一定尺寸以上的车辆执行点云补全
      if (size_x > 1.5f && size_y > 0.8f && size_z > 0.5f) {
        // 创建车辆尺寸
        Eigen::Vector3f dimensions(size_x, size_y, size_z);

        // 计算点云中心点
        Eigen::Vector3f centroid(center_x, center_y, center_z);
        
        // 确定车辆方向 - 对于AABB方法，假设车辆朝向与长边平行
        Eigen::Matrix3f rotation_matrix = Eigen::Matrix3f::Identity();
        
        // 判断哪个边是长边，确定车辆朝向
        if (size_x > size_y) {
          // X方向是车辆长轴，不需要额外旋转
        } else {
          // Y方向是车辆长轴，需要旋转90度
          float angle = M_PI / 2.0; // 90度角
          rotation_matrix(0, 0) = std::cos(angle);
          rotation_matrix(0, 1) = -std::sin(angle);
          rotation_matrix(1, 0) = std::sin(angle);
          rotation_matrix(1, 1) = std::cos(angle);
        }
        
        // 执行点云补全
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr completed_cloud = 
            completeVehiclePointCloud(cluster_cloud, centroid, dimensions, rotation_matrix);
        
        // 将补全后的点云添加到合并点云中
        for (const auto& point : *completed_cloud) {
          current_frame_cloud->points.push_back(point);
        }
        
        ROS_INFO("从AABB方法发布了车辆 #%zu 的补全点云 (尺寸: %.2f x %.2f x %.2f 米)", 
                i, size_x, size_y, size_z);
      } else {
        ROS_INFO("车辆 #%zu 尺寸过小 (%.2f x %.2f x %.2f 米), 跳过点云补全", 
                i, size_x, size_y, size_z);
      }
    }
    
    // 创建线框边界框，使用LINE_LIST类型
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = ns + "_lines";
    line_marker.id = i;
    line_marker.type = visualization_msgs::Marker::LINE_LIST;
    line_marker.action = visualization_msgs::Marker::ADD;
    
    // 设置中心点和方向
    line_marker.pose.position.x = center_x;
    line_marker.pose.position.y = center_y;
    line_marker.pose.position.z = center_z;
    line_marker.pose.orientation.w = 1.0;
    
    // 设置线宽
    line_marker.scale.x = 0.05;  // 线宽为5cm
    
    // 设置线颜色 - 使用不同于点云的颜色
    if (object_type == 1) {  // 车辆
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;  // 绿色
      line_marker.color.a = 1.0;  // 完全不透明
    } else {  // 行人
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 1.0;  // 青色
      line_marker.color.a = 1.0;  // 完全不透明
    }
    
    line_marker.lifetime = ros::Duration(1.5);  // 持续显示1.5秒
    
    // 计算立方体的8个顶点（相对于中心点）
    float half_x = size_x / 2;
    float half_y = size_y / 2;
    float half_z = size_z / 2;
    
    // 定义顶点
    geometry_msgs::Point p[8];
    p[0].x = -half_x; p[0].y = -half_y; p[0].z = -half_z;
    p[1].x =  half_x; p[1].y = -half_y; p[1].z = -half_z;
    p[2].x =  half_x; p[2].y =  half_y; p[2].z = -half_z;
    p[3].x = -half_x; p[3].y =  half_y; p[3].z = -half_z;
    p[4].x = -half_x; p[4].y = -half_y; p[4].z =  half_z;
    p[5].x =  half_x; p[5].y = -half_y; p[5].z =  half_z;
    p[6].x =  half_x; p[6].y =  half_y; p[6].z =  half_z;
    p[7].x = -half_x; p[7].y =  half_y; p[7].z =  half_z;
    
    // 添加12条边
    // 底面
    line_marker.points.push_back(p[0]); line_marker.points.push_back(p[1]);
    line_marker.points.push_back(p[1]); line_marker.points.push_back(p[2]);
    line_marker.points.push_back(p[2]); line_marker.points.push_back(p[3]);
    line_marker.points.push_back(p[3]); line_marker.points.push_back(p[0]);
    
    // 顶面
    line_marker.points.push_back(p[4]); line_marker.points.push_back(p[5]);
    line_marker.points.push_back(p[5]); line_marker.points.push_back(p[6]);
    line_marker.points.push_back(p[6]); line_marker.points.push_back(p[7]);
    line_marker.points.push_back(p[7]); line_marker.points.push_back(p[4]);
    
    // 侧面
    line_marker.points.push_back(p[0]); line_marker.points.push_back(p[4]);
    line_marker.points.push_back(p[1]); line_marker.points.push_back(p[5]);
    line_marker.points.push_back(p[2]); line_marker.points.push_back(p[6]);
    line_marker.points.push_back(p[3]); line_marker.points.push_back(p[7]);
    
    marker_array.markers.push_back(line_marker);
  }

  return marker_array;
}

// 为单个聚类添加边界框
void BoundingBoxGenerator::addBoundingBox(
  pcl::PointCloud<pcl::PointXYZRGB>& output_cloud,
  const pcl::PointCloud<PointType>& cluster_cloud,
  const std::array<float, 3>& color) {
  
// 计算边界框
PointType min_pt, max_pt;
pcl::getMinMax3D(cluster_cloud, min_pt, max_pt);

// 创建边界框的8个顶点
Eigen::Vector3f vertices[8];
vertices[0] = Eigen::Vector3f(min_pt.x, min_pt.y, min_pt.z);
vertices[1] = Eigen::Vector3f(max_pt.x, min_pt.y, min_pt.z);
vertices[2] = Eigen::Vector3f(max_pt.x, max_pt.y, min_pt.z);
vertices[3] = Eigen::Vector3f(min_pt.x, max_pt.y, min_pt.z);
vertices[4] = Eigen::Vector3f(min_pt.x, min_pt.y, max_pt.z);
vertices[5] = Eigen::Vector3f(max_pt.x, min_pt.y, max_pt.z);
vertices[6] = Eigen::Vector3f(max_pt.x, max_pt.y, max_pt.z);
vertices[7] = Eigen::Vector3f(min_pt.x, max_pt.y, max_pt.z);

// 添加边界框的12条边
// 底面
addLine(output_cloud, vertices[0], vertices[1], color);
addLine(output_cloud, vertices[1], vertices[2], color);
addLine(output_cloud, vertices[2], vertices[3], color);
addLine(output_cloud, vertices[3], vertices[0], color);

// 顶面
addLine(output_cloud, vertices[4], vertices[5], color);
addLine(output_cloud, vertices[5], vertices[6], color);
addLine(output_cloud, vertices[6], vertices[7], color);
addLine(output_cloud, vertices[7], vertices[4], color);

// 侧面
addLine(output_cloud, vertices[0], vertices[4], color);
addLine(output_cloud, vertices[1], vertices[5], color);
addLine(output_cloud, vertices[2], vertices[6], color);
addLine(output_cloud, vertices[3], vertices[7], color);
}

// 添加边界框线段
void BoundingBoxGenerator::addLine(
  pcl::PointCloud<pcl::PointXYZRGB>& cloud,
  const Eigen::Vector3f& p1, 
  const Eigen::Vector3f& p2,
  const std::array<float, 3>& color) {
  
// 两点之间的步长
const int steps = 10;
Eigen::Vector3f step = (p2 - p1) / steps;

// 添加线段上的点
for (int i = 0; i <= steps; ++i) {
  Eigen::Vector3f p = p1 + step * i;
  
  pcl::PointXYZRGB point;
  point.x = p.x();
  point.y = p.y();
  point.z = p.z();
  point.r = color[0];
  point.g = color[1];
  point.b = color[2];
  
  cloud.points.push_back(point);
}
}

// 检测雷达是否在转弯
bool BoundingBoxGenerator::isLidarTurning(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                         const std::vector<int>& vehicle_indices) {
  if (vehicle_indices.size() < 100) {
    return false; // 点太少，无法可靠判断
  }
  
  // 提取所有车辆点
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr vehicle_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (const auto& idx : vehicle_indices) {
    vehicle_cloud->points.push_back(cloud->points[idx]);
  }
  vehicle_cloud->width = vehicle_cloud->points.size();
  vehicle_cloud->height = 1;
  vehicle_cloud->is_dense = true;
  
  // 分析点云的分布特性
  // 1. 检查点云在xy平面的分布
  pcl::PointXYZRGB min_pt, max_pt;
  pcl::getMinMax3D(*vehicle_cloud, min_pt, max_pt);
  
  // 计算XY平面上的宽度和长度
  float width_x = max_pt.x - min_pt.x;
  float width_y = max_pt.y - min_pt.y;
  float width_z = max_pt.z - min_pt.z;
  
  // 转弯时，点云在XY平面上的分布可能更加方形(比值接近1)，而非直线时的矩形
  float xy_ratio = std::max(width_x, width_y) / std::min(width_x, width_y);
  const float aspect_threshold = 1.8f; // 降低阈值，使检测更敏感
  bool shape_indicates_turning = (xy_ratio < aspect_threshold);
  
  // 2. 使用PCA分析点云的主方向分布
  pcl::PCA<pcl::PointXYZRGB> pca;
  pca.setInputCloud(vehicle_cloud);
  
  // 获取特征值，表示各方向上的变化程度
  Eigen::Vector3f eigenvalues = pca.getEigenValues();
  
  // 计算特征值比例 - 主方向与次方向的比例
  float eigenvalue_ratio = eigenvalues(0) / eigenvalues(1);
  const float eigenvalue_threshold = 3.0f; // 调整阈值
  bool eigenvalue_indicates_turning = (eigenvalue_ratio < eigenvalue_threshold);
  
  // 3. 分析主方向与坐标轴的关系
  Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
  Eigen::Vector3f main_direction = eigenvectors.col(0);
  
  // 将Z轴分量置为0，只关注XY平面
  main_direction(2) = 0;
  
  // 如果投影后向量太短，使用第二个特征向量
  if (main_direction.norm() < 0.1) {
    main_direction = eigenvectors.col(1);
    main_direction(2) = 0; // 确保水平
  }
  
  // 重新归一化
  if (main_direction.norm() > 0) {
    main_direction.normalize();
  }
  else {
    main_direction = Eigen::Vector3f::UnitX(); // 默认使用X轴
  }
  
  // 计算主方向与X轴和Y轴的夹角
  float angle_with_x = std::abs(std::acos(main_direction.dot(Eigen::Vector3f::UnitX())));
  float angle_with_y = std::abs(std::acos(main_direction.dot(Eigen::Vector3f::UnitY())));
  
  // 转换为度数
  angle_with_x = angle_with_x * 180.0f / M_PI;
  angle_with_y = angle_with_y * 180.0f / M_PI;
  
  // 如果与任一轴的角度接近45度(±20度)，则很可能是在转弯
  bool angle_indicates_turning = (std::abs(angle_with_x - 45.0f) < 20.0f || 
                                std::abs(angle_with_y - 45.0f) < 20.0f);
  
  // 4. 检查点云在雷达坐标系中的分布 - 是否横跨多个象限
  int quadrant_count[4] = {0}; // 四个象限中点的数量
  for (const auto& p : vehicle_cloud->points) {
    if (p.x >= 0 && p.y >= 0) quadrant_count[0]++;
    else if (p.x < 0 && p.y >= 0) quadrant_count[1]++;
    else if (p.x < 0 && p.y < 0) quadrant_count[2]++;
    else quadrant_count[3]++;
  }
  
  // 计算非空象限的数量
  int non_empty_quadrants = 0;
  for (int i = 0; i < 4; i++) {
    if (quadrant_count[i] > vehicle_cloud->points.size() * 0.05) // 如果某象限包含超过5%的点
      non_empty_quadrants++;
  }
  
  // 如果点云分布在多个象限中，可能是转弯
  bool quadrant_indicates_turning = (non_empty_quadrants >= 2);
  
  // 综合所有因素做出判断
  // 转弯的情况通常有以下几种情况的至少两种为真:
  // 1. 点云形状接近正方形
  // 2. PCA特征值比例较小
  // 3. 主方向与坐标轴成45度角
  // 4. 点云分布在多个象限
  int turning_indicators = 0;
  if (shape_indicates_turning) turning_indicators++;
  if (eigenvalue_indicates_turning) turning_indicators++;
  if (angle_indicates_turning) turning_indicators++;
  if (quadrant_indicates_turning) turning_indicators++;
  
  bool is_turning = (turning_indicators >= 2);
  
  // 输出调试信息
  ROS_INFO("Turning detection - Shape ratio: %.2f, Eigenvalue ratio: %.2f, "
           "Angle with X: %.2f, Angle with Y: %.2f, Non-empty quadrants: %d, "
           "Turning indicators: %d, Is turning: %s",
           xy_ratio, eigenvalue_ratio, 
           angle_with_x, angle_with_y, 
           non_empty_quadrants, turning_indicators,
           is_turning ? "true" : "false");
  
  return is_turning;
}

// 车辆点云补全功能实现
pcl::PointCloud<pcl::PointXYZRGB>::Ptr BoundingBoxGenerator::completeVehiclePointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const Eigen::Vector3f& centroid,
    const Eigen::Vector3f& dimensions,
    const Eigen::Matrix3f& rotation_matrix) {
  
  // 创建补全后的点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr completed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  // 如果输入点云太小，不进行补全
  if (input_cloud->points.size() < 50) {
    ROS_WARN("输入点云太小 (%zu 点), 跳过补全", input_cloud->points.size());
    return input_cloud; // 直接返回原始点云
  }
  
  // 首先复制原始点云 - 给原始点云着色
  for (const auto& point : input_cloud->points) {
      pcl::PointXYZRGB colored_point = point;
      colored_point.r = 255;  // 原始点云用红色
      colored_point.g = 0;
      colored_point.b = 0;
      completed_cloud->points.push_back(colored_point);
  }
  
  // 设置补全点的颜色 - 使用原始的蓝色
  uint8_t r = 0;    // 蓝色
  uint8_t g = 150;
  uint8_t b = 255;
  
  // 提取旋转矩阵的列向量，表示车辆的主方向
  Eigen::Vector3f x_axis = rotation_matrix.col(0); // 车辆前后方向
  Eigen::Vector3f y_axis = rotation_matrix.col(1); // 车辆左右方向
  Eigen::Vector3f z_axis = rotation_matrix.col(2); // 车辆上下方向
  
  // 分析点云在车辆左右两侧的分布
  int left_points = 0, right_points = 0;
  for (const auto& point : input_cloud->points) {
      Eigen::Vector3f p(point.x, point.y, point.z);
      float rel_y = (p - centroid).dot(y_axis);
      if (rel_y > 0) right_points++;
      else left_points++;
  }
  
  // 决定对称方向，通常我们选择点数较少的一侧进行补全
  bool mirror_left_to_right = (left_points > right_points);
  bool mirror_right_to_left = (right_points > left_points);
  
  // 如果两侧点数相差不大，则双向对称补全 - 使用原始代码的条件
  if (std::abs(left_points - right_points) < input_cloud->points.size() * 0.2) {
      mirror_left_to_right = true;
      mirror_right_to_left = true;
  }
  
  // 同样分析前后分布
  int front_points = 0, rear_points = 0;
  for (const auto& point : input_cloud->points) {
      Eigen::Vector3f p(point.x, point.y, point.z);
      float rel_x = (p - centroid).dot(x_axis);
      if (rel_x > 0) front_points++;
      else rear_points++;
  }
  
  // 决定是否需要前后对称补全 - 使用原始代码的条件
  bool mirror_front_to_rear = (front_points > rear_points * 2);
  bool mirror_rear_to_front = (rear_points > front_points * 2);
  
  ROS_INFO("点云分布: 左侧=%d, 右侧=%d, 前部=%d, 后部=%d", 
           left_points, right_points, front_points, rear_points);
  
  // 执行左右对称
  int mirrored_points = 0;
  
  // 左侧点映射到右侧
  if (mirror_left_to_right) {
      for (const auto& point : input_cloud->points) {
          // 计算点相对于中心的位置向量
          Eigen::Vector3f point_vec(point.x, point.y, point.z);
          Eigen::Vector3f rel_pos = point_vec - centroid;
          
          // 计算点在车辆坐标系下的坐标
          float px = rel_pos.dot(x_axis); // 前后坐标
          float py = rel_pos.dot(y_axis); // 左右坐标
          float pz = rel_pos.dot(z_axis); // 上下坐标
          
          // 只对左侧点(py < 0)进行镜像
          if (py < 0) {
              // 左右对称变换（y坐标取反）
              float mirror_py = -py;
              
              // 将对称点变换回世界坐标系
              Eigen::Vector3f mirror_point_local = px * x_axis + mirror_py * y_axis + pz * z_axis;
              Eigen::Vector3f mirror_point_global = mirror_point_local + centroid;
              
              // 创建对称点
              pcl::PointXYZRGB new_point;
              new_point.x = mirror_point_global(0);
              new_point.y = mirror_point_global(1);
              new_point.z = mirror_point_global(2);
              new_point.r = r;
              new_point.g = g;
              new_point.b = b;
              
              // 检查是否超出边界框
              bool in_bounds = true;
              
              // 边界框检查 - 使用原始代码的边距
              float margin = 0.1; // 10厘米余量
              if (std::abs(mirror_py) > dimensions(1)/2 + margin) in_bounds = false;
              
              // 如果在边界内，则添加
              if (in_bounds) {
                  completed_cloud->points.push_back(new_point);
                  mirrored_points++;
                  
                  // 额外添加一些近邻点，使点云更密集 - 使用原始代码的参数
                  const float jitter = 0.05f;  // 5厘米的抖动
                  for (int j = 0; j < 3; j++) {  // 为每个点额外添加3个邻居
                      pcl::PointXYZRGB neighbor = new_point;
                      neighbor.x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      completed_cloud->points.push_back(neighbor);
                      mirrored_points++;
                  }
              }
          }
      }
  }
  
  // 右侧点映射到左侧
  if (mirror_right_to_left) {
      for (const auto& point : input_cloud->points) {
          // 计算点相对于中心的位置向量
          Eigen::Vector3f point_vec(point.x, point.y, point.z);
          Eigen::Vector3f rel_pos = point_vec - centroid;
          
          // 计算点在车辆坐标系下的坐标
          float px = rel_pos.dot(x_axis); // 前后坐标
          float py = rel_pos.dot(y_axis); // 左右坐标
          float pz = rel_pos.dot(z_axis); // 上下坐标
          
          // 只对右侧点(py > 0)进行镜像
          if (py > 0) {
              // 左右对称变换（y坐标取反）
              float mirror_py = -py;
              
              // 将对称点变换回世界坐标系
              Eigen::Vector3f mirror_point_local = px * x_axis + mirror_py * y_axis + pz * z_axis;
              Eigen::Vector3f mirror_point_global = mirror_point_local + centroid;
              
              // 创建对称点
              pcl::PointXYZRGB new_point;
              new_point.x = mirror_point_global(0);
              new_point.y = mirror_point_global(1);
              new_point.z = mirror_point_global(2);
              new_point.r = r;
              new_point.g = g;
              new_point.b = b;
              
              // 检查是否超出边界框
              bool in_bounds = true;
              
              // 边界框检查 - 允许一定的余量
              float margin = 0.1; // 10厘米余量
              if (std::abs(mirror_py) > dimensions(1)/2 + margin) in_bounds = false;
              
              // 如果在边界内，则添加
              if (in_bounds) {
                  completed_cloud->points.push_back(new_point);
                  mirrored_points++;
                  
                  // 额外添加一些近邻点，使点云更密集
                  const float jitter = 0.05f;  // 5厘米的抖动
                  for (int j = 0; j < 3; j++) {  // 为每个点额外添加3个邻居
                      pcl::PointXYZRGB neighbor = new_point;
                      neighbor.x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      completed_cloud->points.push_back(neighbor);
                      mirrored_points++;
                  }
              }
          }
      }
  }
  
  // 执行前后对称（通常车辆前后不对称，但在某些情况下有用）
  if (mirror_front_to_rear) {
      for (const auto& point : input_cloud->points) {
          Eigen::Vector3f point_vec(point.x, point.y, point.z);
          Eigen::Vector3f rel_pos = point_vec - centroid;
          
          float px = rel_pos.dot(x_axis);
          float py = rel_pos.dot(y_axis);
          float pz = rel_pos.dot(z_axis);
          
          // 只对前部点(px > 0)进行镜像到后部
          if (px > 0) {
              float mirror_px = -px;
              
              Eigen::Vector3f mirror_point_local = mirror_px * x_axis + py * y_axis + pz * z_axis;
              Eigen::Vector3f mirror_point_global = mirror_point_local + centroid;
              
              pcl::PointXYZRGB new_point;
              new_point.x = mirror_point_global(0);
              new_point.y = mirror_point_global(1);
              new_point.z = mirror_point_global(2);
              new_point.r = r;
              new_point.g = 100; // 使用稍微不同的颜色
              new_point.b = b;
              
              bool in_bounds = true;
              
              float margin = 0.1;
              if (std::abs(mirror_px) > dimensions(0)/2 + margin) in_bounds = false;
              
              if (in_bounds) {
                  completed_cloud->points.push_back(new_point);
                  mirrored_points++;
                  
                  // 额外添加一些近邻点
                  const float jitter = 0.05f;
                  for (int j = 0; j < 3; j++) {
                      pcl::PointXYZRGB neighbor = new_point;
                      neighbor.x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      completed_cloud->points.push_back(neighbor);
                      mirrored_points++;
                  }
              }
          }
      }
  }
  
  if (mirror_rear_to_front) {
      for (const auto& point : input_cloud->points) {
          Eigen::Vector3f point_vec(point.x, point.y, point.z);
          Eigen::Vector3f rel_pos = point_vec - centroid;
          
          float px = rel_pos.dot(x_axis);
          float py = rel_pos.dot(y_axis);
          float pz = rel_pos.dot(z_axis);
          
          // 只对后部点(px < 0)进行镜像到前部
          if (px < 0) {
              float mirror_px = -px;
              
              Eigen::Vector3f mirror_point_local = mirror_px * x_axis + py * y_axis + pz * z_axis;
              Eigen::Vector3f mirror_point_global = mirror_point_local + centroid;
              
              pcl::PointXYZRGB new_point;
              new_point.x = mirror_point_global(0);
              new_point.y = mirror_point_global(1);
              new_point.z = mirror_point_global(2);
              new_point.r = r;
              new_point.g = 100; // 使用稍微不同的颜色
              new_point.b = b;
              
              bool in_bounds = true;
              
              float margin = 0.1;
              if (std::abs(mirror_px) > dimensions(0)/2 + margin) in_bounds = false;
              
              if (in_bounds) {
                  completed_cloud->points.push_back(new_point);
                  mirrored_points++;
                  
                  // 额外添加一些近邻点
                  const float jitter = 0.05f;
                  for (int j = 0; j < 3; j++) {
                      pcl::PointXYZRGB neighbor = new_point;
                      neighbor.x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      neighbor.z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter;
                      completed_cloud->points.push_back(neighbor);
                      mirrored_points++;
                  }
              }
          }
      }
  }
  
  // 更新点云属性
  completed_cloud->width = completed_cloud->points.size();
  completed_cloud->height = 1;
  completed_cloud->is_dense = true;
  
  ROS_INFO("原始点云: %zu 点, 通过对称补全添加: %d 点, 最终点云: %zu 点", 
           input_cloud->points.size(),
           mirrored_points,
           completed_cloud->points.size());
  
  return completed_cloud;
}

// 发布补全的车辆点云
void BoundingBoxGenerator::publishCompletedVehicleCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& completed_cloud,
    const std::string& frame_id) {
    
    // 创建ROS节点句柄
    ros::NodeHandle nh;
    
    // 创建点云发布者 - 使用更具描述性的话题名称
    static ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vehicle_completion/point_cloud", 1);
    
    // 转换到ROS消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*completed_cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = ros::Time::now();
    
    // 发布点云消息
    cloud_pub.publish(cloud_msg);
    
    // 记录日志
    static ros::Time last_log_time = ros::Time::now();
    if ((ros::Time::now() - last_log_time).toSec() > 1.0) {  // 每秒最多记录一次日志，避免日志过多
        ROS_INFO("-------------------------");
        ROS_INFO("发布了补全点云到话题: /vehicle_completion/point_cloud");
        ROS_INFO("原始点云: %zu 点, 补全后点云: %zu 点", 
                completed_cloud->points.size() / 4,  // 估计原始点云大约是总点数的1/4
                completed_cloud->points.size());
        ROS_INFO("请在RViz中添加PointCloud2显示，订阅话题: /vehicle_completion/point_cloud");
        ROS_INFO("-------------------------");
        last_log_time = ros::Time::now();
    }
}