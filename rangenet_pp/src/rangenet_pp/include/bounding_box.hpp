#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

// 确保与现有代码中的PointType兼容
using PointType = pcl::PointXYZI; // 根据您的具体类型调整

class BoundingBoxGenerator {
public:
  BoundingBoxGenerator(double cluster_tolerance = 0.5, 
                int min_cluster_size = 10, 
                int max_cluster_size = 10000);
  
  // 处理语义分割结果，生成带有边界框的点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr processBoundingBoxes(
    const pcl::PointCloud<PointType>& input_cloud,
    const int* labels);
    
  // 发布边界框，可以在 processBoundingBoxes 中调用
  void publishBoundingBoxes(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      const int* labels,
      const std::string& frame_id);

  // 车辆点云补全 - 新增函数
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr completeVehiclePointCloud(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
      const Eigen::Vector3f& centroid,
      const Eigen::Vector3f& dimensions,
      const Eigen::Matrix3f& rotation_matrix);
      
  // 发布补全的车辆点云
  void publishCompletedVehicleCloud(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& completed_cloud,
      const std::string& frame_id);

private:
  // 聚类参数
  double vehicle_cluster_tolerance_;
  int vehicle_min_cluster_size_;
  int vehicle_max_cluster_size_;
  
  double pedestrian_cluster_tolerance_;
  int pedestrian_min_cluster_size_;
  int pedestrian_max_cluster_size_;
  
  // 添加合并点云作为成员变量，以便在不同方法之间共享
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_frame_cloud;
  
  // 移除原有的静态标签定义，这些现在直接在方法中定义
  // const std::vector<int> vehicle_labels_ = {1}; // 车辆 
  // const std::vector<int> pedestrian_labels_ = {7}; // 行人
  
  // 提取特定标签的点云
  pcl::PointCloud<PointType>::Ptr extractLabeledPointCloud(
    const pcl::PointCloud<PointType>& cloud,
    const int* labels,
    const std::vector<int>& target_labels);
    
  // 对点云进行欧式聚类
  std::vector<pcl::PointIndices> euclideanClustering(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    double tolerance,
    int min_size,
    int max_size);
    
  // 为单个聚类添加边界框
  void addBoundingBox(
    pcl::PointCloud<pcl::PointXYZRGB>& output_cloud,
    const pcl::PointCloud<PointType>& cluster_cloud,
    const std::array<float, 3>& color);
    
  // 添加边界框线段
  void addLine(
    pcl::PointCloud<pcl::PointXYZRGB>& cloud,
    const Eigen::Vector3f& p1, 
    const Eigen::Vector3f& p2,
    const std::array<float, 3>& color);
    
  // 聚类分析，将点云中的物体点分组
  std::vector<pcl::PointIndices> performClustering(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      const std::vector<int>& object_indices,
      float cluster_tolerance = 0.5);
      
  // 计算点云聚类的3D边界框
  visualization_msgs::MarkerArray generateBoundingBoxes(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      const std::vector<pcl::PointIndices>& clusters,
      const std::string& frame_id,
      const std::string& ns,
      int object_type);
      
  // 计算定向边界框(OBB)，解决车辆朝向问题
  visualization_msgs::MarkerArray generateOrientedBoundingBoxes(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      const std::vector<pcl::PointIndices>& clusters,
      const std::string& frame_id,
      const std::string& ns,
      int object_type);
      
  // 检测雷达是否在转弯
  bool isLidarTurning(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                     const std::vector<int>& vehicle_indices);
};