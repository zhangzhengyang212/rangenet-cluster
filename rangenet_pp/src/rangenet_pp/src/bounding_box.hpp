// 轴对齐边界框(AABB)方法，用于行人等对象
visualization_msgs::MarkerArray generateBoundingBoxes(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const std::vector<pcl::PointIndices>& clusters,
  const std::string& frame_id,
  const std::string& ns,
  int object_type,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& merged_completed_cloud,
  bool& has_completed_cloud);

// 计算定向边界框(OBB)，解决车辆朝向问题
visualization_msgs::MarkerArray generateOrientedBoundingBoxes(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
  const std::vector<pcl::PointIndices>& clusters,
  const std::string& frame_id,
  const std::string& ns,
  int object_type,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& merged_completed_cloud,
  bool& has_completed_cloud); 