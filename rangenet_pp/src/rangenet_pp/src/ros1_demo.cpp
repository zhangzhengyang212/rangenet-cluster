#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "ros/ros.h"
#include <filesystem>
#include <functional>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include <visualization_msgs/MarkerArray.h>
#include "../include/bounding_box.hpp"  // 修改为正确的相对路径

// 确保与bounding_box.hpp中定义一致
using PointType = pcl::PointXYZI;

class ROS_DEMO {
public:
  explicit ROS_DEMO(ros::NodeHandle *pnh);

private:
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
  ros::NodeHandle *pnh_;
  ros::Publisher pub_;
  ros::Publisher vehicle_box_pub_;     // 添加车辆边界框发布者
  ros::Publisher pedestrian_box_pub_;  // 添加行人边界框发布者
  ros::Subscriber sub_;
  std::unique_ptr<rangenet::segmentation::Net> net_;
  std::unique_ptr<BoundingBoxGenerator> bbox_generator_;
};

ROS_DEMO::ROS_DEMO(ros::NodeHandle *pnh) : pnh_(pnh) {

  std::filesystem::path file_path(__FILE__);
  std::string model_dir = std::string(file_path.parent_path().parent_path() / "model/");
  ROS_INFO("model_dir: %s", model_dir.c_str());

  sub_ = pnh_->subscribe<sensor_msgs::PointCloud2>("/points_raw", 10, &ROS_DEMO::pointcloudCallback, this);
  pub_ = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud", 1, true);
  
  // 添加边界框发布者
  vehicle_box_pub_ = pnh_->advertise<visualization_msgs::MarkerArray>("/vehicle_boxes", 1);
  pedestrian_box_pub_ = pnh_->advertise<visualization_msgs::MarkerArray>("/pedestrian_boxes", 1);

  net_ = std::unique_ptr<rangenet::segmentation::Net>(new rangenet::segmentation::NetTensorRT(model_dir, false));
  
  // 初始化边界框生成器
  // 参数分别是: 聚类容忍度(米), 最小聚类点数, 最大聚类点数
  bbox_generator_ = std::make_unique<BoundingBoxGenerator>(0.5, 50, 25000);
};

void ROS_DEMO::pointcloudCallback(
  const sensor_msgs::PointCloud2::ConstPtr &pc_msg) {
  // 添加调试信息
  ROS_INFO("Received pointcloud with %d points", pc_msg->width * pc_msg->height);

  // ROS 消息类型 -> PCL 点云类型
  pcl::PointCloud<PointType>::Ptr pc_ros(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc_msg, *pc_ros);
  
  ROS_INFO("Converted to PCL cloud with %zu points", pc_ros->size());

  // 预测
  auto labels = std::make_unique<int[]>(pc_ros->size());
  ROS_INFO("About to call doInfer");
  net_->doInfer(*pc_ros, labels.get());
  ROS_INFO("doInfer completed");

  // 使用边界框生成器处理点云
  ROS_INFO("About to call processBoundingBoxes");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud = 
    bbox_generator_->processBoundingBoxes(*pc_ros, labels.get());
  ROS_INFO("processBoundingBoxes completed successfully");
  
  // 发布带边界框的点云
  sensor_msgs::PointCloud2 ros_msg;
  pcl::toROSMsg(*result_cloud, ros_msg);
  ros_msg.header = pc_msg->header;
  pub_.publish(ros_msg);
  
  // 提取并发布边界框
  // 这里不需要再次调用publishBoundingBoxes，因为在processBoundingBoxes中已经调用了
  // 但如果您想要更精细地控制发布，可以在这里添加额外的处理逻辑
  
  ROS_INFO("Published processed pointcloud");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ros1_demo");
  ros::NodeHandle pnh("~");
  ROS_DEMO node(&pnh);
  ros::spin();
  return 0;
}