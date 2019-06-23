/**
 * @file lolLocalization.cpp
 * @author heng zhang (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2019-05-04
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "lego_lol/lolLocalization.h"

namespace localization
{
LolLocalization::LolLocalization(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<ros::NodeHandle> pnh) : nh_(nh), pnh_(pnh)
{

  pnh_->param<float>("surround_search_radius", surround_search_radius_, 50.0);
  pnh_->param<int>("surround_search_num", surround_search_num_, 50);
  pnh_->param<float>("corner_leaf", corner_leaf_, 0.2);
  pnh_->param<float>("surf_leaf", surf_leaf_, 0.4);
  pnh_->param<float>("outlier_leaf", outlier_leaf_, 0.4);
  pnh_->param<float>("surround_keyposes_leaf", surround_keyposes_leaf_, 0.3);
  pnh_->param<std::string>("fn_poses", fn_poses_, std::string("~"));
  pnh_->param<std::string>("fn_corner", fn_corner_, std::string("~"));
  pnh_->param<std::string>("fn_surf", fn_surf_, std::string("~"));
  pnh_->param<std::string>("fn_outlier", fn_outlier_, std::string("~"));
  pnh_->param<float>("target_update_dist", target_update_dist_, 5.);

  tf_b2l_ = Eigen::Matrix4f::Identity();
  float roll, pitch, yaw;
  if (!nh_->getParam("tf_b2l_x", tf_b2l_(0, 3)) || !nh_->getParam("tf_b2l_y", tf_b2l_(1, 3)) || !nh_->getParam("tf_b2l_z", tf_b2l_(2, 3)) || !nh_->getParam("tf_b2l_roll", roll) || !nh_->getParam("tf_b2l_pitch", pitch) || !nh_->getParam("tf_b2l_yaw", yaw))
  {
    ROS_ERROR("transform between /base_link to /laser not set.");
    exit(-1);
  }
  Eigen::AngleAxisf rx(roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf ry(pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rz(yaw, Eigen::Vector3f::UnitZ());
  tf_b2l_.block(0, 0, 3, 3) = (rz * ry * rx).matrix();

  if (!init())
  {
    ROS_ERROR("failed init.");
    exit(-1);
  }

  lol_ = std::shared_ptr<LolLocalization>(this);

  pub_lol_pose_ = nh_->advertise<geometry_msgs::PoseStamped>("/current_pose", 5);

  sub_odom_ = nh_->subscribe<nav_msgs::Odometry>("/odom", 40, boost::bind(&LolLocalization::odomCB, this, _1));
  sub_corner_ = nh_->subscribe<sensor_msgs::PointCloud2>("/corner", 1, boost::bind(&LolLocalization::cornerCB, this, _1));
  sub_surf_ = nh_->subscribe<sensor_msgs::PointCloud2>("/surf", 1, boost::bind(&LolLocalization::surfCB, this, _1));
  sub_outlier_ = nh_->subscribe<sensor_msgs::PointCloud2>("/outlier", 1, boost::bind(&LolLocalization::outlierCB, this, _1));
  sub_initial_pose_ = nh_->subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, boost::bind(&LolLocalization::initialPoseCB, this, _1));

  pub_corner_target_ = nh_->advertise<sensor_msgs::PointCloud2>("/corner_target", 1);
  pub_surf_target_ = nh_->advertise<sensor_msgs::PointCloud2>("/surf_target", 1);
  pub_corner_source_ = nh_->advertise<sensor_msgs::PointCloud2>("/corner_source", 1);
  pub_surf_source_ = nh_->advertise<sensor_msgs::PointCloud2>("/surf_source", 1);
}

bool LolLocalization::init()
{

  keyposes_3d_.reset(new pcl::PointCloud<PointType>());

  laser_corner_.reset(new pcl::PointCloud<PointType>());
  laser_surf_.reset(new pcl::PointCloud<PointType>());
  laser_outlier_.reset(new pcl::PointCloud<PointType>());
  laser_corner_ds_.reset(new pcl::PointCloud<PointType>());
  laser_surf_ds_.reset(new pcl::PointCloud<PointType>());
  laser_outlier_ds_.reset(new pcl::PointCloud<PointType>());
  pc_corner_target_.reset(new pcl::PointCloud<PointType>());
  pc_surf_target_.reset(new pcl::PointCloud<PointType>());
  pc_corner_target_ds_.reset(new pcl::PointCloud<PointType>());
  pc_surf_target_ds_.reset(new pcl::PointCloud<PointType>());
  target_center_.x = target_center_.y = target_center_.z = -100.;
  pc_surround_keyposes_.reset(new pcl::PointCloud<PointType>());

  kdtree_keyposes_3d_.reset(new pcl::KdTreeFLANN<PointType>());
  kdtree_corner_target_.reset(new pcl::KdTreeFLANN<PointType>());
  kdtree_surf_target_.reset(new pcl::KdTreeFLANN<PointType>());

  ds_corner_.setLeafSize(corner_leaf_, corner_leaf_, corner_leaf_);
  ds_surf_.setLeafSize(surf_leaf_, surf_leaf_, surf_leaf_);
  ds_outlier_.setLeafSize(outlier_leaf_, outlier_leaf_, outlier_leaf_);
  ds_surround_keyposes_.setLeafSize(surround_keyposes_leaf_, surround_keyposes_leaf_, surround_keyposes_leaf_);

  cur_laser_pose_ = geometry_msgs::PoseStamped();
  pre_laser_pose_ = geometry_msgs::PoseStamped();

  // 加载地图和位姿 pcd 文件
  pcl::PointCloud<PointType>::Ptr corner_pc(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surf_pc(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr outlier_pc(new pcl::PointCloud<PointType>());
  auto start = std::chrono::system_clock::now();
  if (pcl::io::loadPCDFile(fn_poses_, *keyposes_3d_) == -1 || pcl::io::loadPCDFile(fn_corner_, *corner_pc) == -1 ||
      pcl::io::loadPCDFile(fn_surf_, *surf_pc) == -1 || pcl::io::loadPCDFile(fn_outlier_, *outlier_pc) == -1)
  {
    ROS_ERROR("couldn't load pcd file");
    return false;
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  ROS_INFO("time: %f s ----> keyposes: %d, corner pc: %d, surf pc: %d, outlier pc: %d", elapsed.count(), keyposes_3d_->points.size(), corner_pc->points.size(), surf_pc->points.size(), outlier_pc->points.size());

  kdtree_keyposes_3d_->setInputCloud(keyposes_3d_);

  corner_keyframes_.resize(keyposes_3d_->points.size());
  surf_keyframes_.resize(keyposes_3d_->points.size());
  outlier_keyframes_.resize(keyposes_3d_->points.size());
  // std::fill(corner_keyframes_.begin(), corner_keyframes_.end(), pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>()));
  // std::fill(surf_keyframes_.begin(), surf_keyframes_.end(), pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>()));
  // std::fill(outlier_keyframes_.begin(), outlier_keyframes_.end(), pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>()));
  for (int i = 0; i < keyposes_3d_->points.size(); ++i)
  {
    corner_keyframes_[i] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    surf_keyframes_[i] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    outlier_keyframes_[i] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
  }

  for (int i = 0; i < corner_pc->points.size(); ++i)
  {
    const auto &p = corner_pc->points[i];
    corner_keyframes_[int(p.intensity)]->points.push_back(p);
  }
  for (int i = 0; i < surf_pc->points.size(); ++i)
  {
    const auto &p = surf_pc->points[i];
    surf_keyframes_[int(p.intensity)]->points.push_back(p);
  }
  for (int i = 0; i < outlier_pc->points.size(); ++i)
  {
    const auto &p = outlier_pc->points[i];
    outlier_keyframes_[int(p.intensity)]->points.push_back(p);
  }

  time_laser_corner_ = 0.;
  time_laser_surf_ = 0.;
  time_laser_outlier_ = 0.;
  new_laser_corner_ = false;
  new_laser_surf_ = false;
  new_laser_outlier_ = false;
  new_odom_ = false;

  odom_front_ = 0;
  odom_last_ = 0;

  for (int i = 0; i < 6; ++i)
  {
    tobe_optimized_[i] = 0;
  }
  options_.minimizer_progress_to_stdout = true;
  // options_.linear_solver_type = ceres::DENSE_QR;
  options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options_.max_num_iterations = 10;
  // options_.initial_trust_region_radius = 1e2;
  // options_.max_trust_region_radius = 1e6;
  // options_.min_trust_region_radius = 1e-10;
  options_.callbacks.push_back(new IterCB(lol_));
  options_.update_state_every_iteration = true;

  ROS_INFO("init ok.");

  return true;
}

void LolLocalization::odomCB(const nav_msgs::OdometryConstPtr &msg)
{
  msg_odoms_[odom_last_] = *msg;
  odom_last_ = (odom_last_ + 1) % 50;
  if (odom_last_ == odom_front_)
  {
    odom_front_ = (odom_front_ + 1) % 50;
  }
  new_odom_ = true;
}

void LolLocalization::cornerCB(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  laser_corner_->clear();
  pcl::fromROSMsg(*msg, *laser_corner_);
  time_laser_corner_ = msg->header.stamp.toSec();
  new_laser_corner_ = true;
}
void LolLocalization::surfCB(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  laser_surf_->clear();
  pcl::fromROSMsg(*msg, *laser_surf_);
  time_laser_surf_ = msg->header.stamp.toSec();
  new_laser_surf_ = true;
}
void LolLocalization::outlierCB(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  laser_outlier_->clear();
  pcl::fromROSMsg(*msg, *laser_outlier_);
  time_laser_outlier_ = msg->header.stamp.toSec();
  new_laser_outlier_ = true;
}

void LolLocalization::initialPoseCB(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
{
  PointType p;
  p.x = msg->pose.pose.position.x;
  p.y = msg->pose.pose.position.y;
  p.z = msg->pose.pose.position.z;
  pre_laser_pose_.pose = cur_laser_pose_.pose = msg->pose.pose;
  pre_laser_pose_.header = cur_laser_pose_.header = msg->header;
  extractSurroundKeyFrames(p);
}

void LolLocalization::odomThread()
{
  ros::Duration duration(0.05);

  while (odom_last_ == 0)
  {
    duration.sleep();
    ros::spinOnce();
  }

  while (ros::ok())
  {

    // 用不用加锁？
    nav_msgs::Odometry cur_laser_odom;                                             // 与 cur_laser_pose_ 时间最相近的 odom
    nav_msgs::Odometry recent_laser_odom = msg_odoms_[(odom_last_ + 50 - 1) % 50]; // 目前最近的 odom
    int ptr = odom_front_;
    while (ptr != odom_last_)
    {
      if (msg_odoms_[ptr].header.stamp.toSec() >= cur_laser_pose_.header.stamp.toSec())
      {
        break;
      }
      ptr = (ptr + 1) % 50;
    }

    if (ptr == odom_last_ || odom_front_)
    {
      cur_laser_odom = msg_odoms_[(odom_last_ + 50 - 1) % 50];
    }
    else
    {
      cur_laser_odom = msg_odoms_[ptr];

      int ptr_back = (ptr + 50 - 1) % 50;
      float ratio_back = (cur_laser_odom.header.stamp.toSec() - msg_odoms_[ptr_back].header.stamp.toSec()) / (msg_odoms_[ptr].header.stamp.toSec() - msg_odoms_[ptr_back].header.stamp.toSec());
      Eigen::Quaternionf q_b(msg_odoms_[ptr_back].pose.pose.orientation.w, msg_odoms_[ptr_back].pose.pose.orientation.x, msg_odoms_[ptr_back].pose.pose.orientation.y, msg_odoms_[ptr_back].pose.pose.orientation.z);
      Eigen::Quaternionf q_f(msg_odoms_[ptr].pose.pose.orientation.w, msg_odoms_[ptr].pose.pose.orientation.x, msg_odoms_[ptr].pose.pose.orientation.y, msg_odoms_[ptr].pose.pose.orientation.z);
      auto euler_b = q_b.toRotationMatrix().eulerAngles(2, 1, 0);
      auto euler_f = q_f.toRotationMatrix().eulerAngles(2, 1, 0);
      Eigen::Vector3f euler = (1 - ratio_back) * euler_b + ratio_back * euler_f;
      Eigen::Quaternionf q_cur = Eigen::AngleAxisf(euler(0), Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(euler(1), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(euler(2), Eigen::Vector3f::UnitX());

      cur_laser_odom.pose.pose.position.x = (1 - ratio_back) * msg_odoms_[ptr_back].pose.pose.position.x + ratio_back * msg_odoms_[ptr].pose.pose.position.x;
      cur_laser_odom.pose.pose.position.y = (1 - ratio_back) * msg_odoms_[ptr_back].pose.pose.position.y + ratio_back * msg_odoms_[ptr].pose.pose.position.y;
      cur_laser_odom.pose.pose.position.z = (1 - ratio_back) * msg_odoms_[ptr_back].pose.pose.position.z + ratio_back * msg_odoms_[ptr].pose.pose.position.z;
      cur_laser_odom.pose.pose.orientation.w = q_cur.w();
      cur_laser_odom.pose.pose.orientation.x = q_cur.x();
      cur_laser_odom.pose.pose.orientation.y = q_cur.y();
      cur_laser_odom.pose.pose.orientation.z = q_cur.z();
      cur_laser_odom.header.stamp.fromSec((1 - ratio_back) * msg_odoms_[ptr_back].header.stamp.toSec() + ratio_back * msg_odoms_[ptr].header.stamp.toSec());
    }

    if ((recent_laser_odom.header.stamp - cur_laser_odom.header.stamp).toSec() < 0.01)
    {
      // 没有新的里程计数据，用 cur_laser_pose 和 pre_laser_pose 预测目前 laser 的位姿情况
      // 还是不预测了
      ROS_WARN("no new odom msg current.");
      ROS_INFO("recent :%f, cur %f", recent_laser_odom.header.stamp.toSec(), cur_laser_odom.header.stamp.toSec());
      pre_laser_pose_.header.stamp = cur_laser_pose_.header.stamp;
      cur_laser_pose_.header.stamp = cur_laser_odom.header.stamp;
    }
    else
    {
      // 使用里程计数据预测目前 laser 的位姿情况，T_{o-cur_laser_pose} * T_{o-cur_laser_odom}^{-1} * T_{o-recent_laser_odom}
      // 做一个插值
      Eigen::Matrix4f T_olp = Eigen::Matrix4f::Identity();
      Eigen::Quaternionf q_olc(cur_laser_pose_.pose.orientation.w, cur_laser_pose_.pose.orientation.x, cur_laser_pose_.pose.orientation.y, cur_laser_pose_.pose.orientation.z);
      Eigen::Quaternionf q_ooc(cur_laser_odom.pose.pose.orientation.w, cur_laser_odom.pose.pose.orientation.x, cur_laser_odom.pose.pose.orientation.y, cur_laser_odom.pose.pose.orientation.z);
      Eigen::Quaternionf q_oor(recent_laser_odom.pose.pose.orientation.w, recent_laser_odom.pose.pose.orientation.x, recent_laser_odom.pose.pose.orientation.y, recent_laser_odom.pose.pose.orientation.z);
      Eigen::Quaternionf q_olp = q_olc * q_ooc.inverse() * q_oor;
      Eigen::Vector3f t_olc(cur_laser_pose_.pose.position.x, cur_laser_pose_.pose.position.y, cur_laser_pose_.pose.position.z);
      Eigen::Vector3f t_ooc(cur_laser_odom.pose.pose.position.x, cur_laser_odom.pose.pose.position.y, cur_laser_odom.pose.pose.position.z);
      Eigen::Vector3f t_oor(recent_laser_odom.pose.pose.position.x, recent_laser_odom.pose.pose.position.y, recent_laser_odom.pose.pose.position.z);
      Eigen::Vector3f t_olp = t_olc - q_olc * q_ooc.inverse() * (t_ooc - t_oor);
      T_olp.block<3, 3>(0, 0) = q_olp.matrix();
      T_olp.block<3, 1>(0, 3) = t_olp;

      pre_laser_pose_ = cur_laser_pose_;
      cur_laser_pose_.pose.position.x = t_olp(0);
      cur_laser_pose_.pose.position.y = t_olc(1);
      cur_laser_pose_.pose.position.z = t_olc(2);
      cur_laser_pose_.pose.orientation.w = q_olp.w();
      cur_laser_pose_.pose.orientation.x = q_olp.x();
      cur_laser_pose_.pose.orientation.y = q_olp.y();
      cur_laser_pose_.pose.orientation.z = q_olp.z();
      cur_laser_pose_.header.stamp = recent_laser_odom.header.stamp;
    }

    cur_laser_pose_.header.frame_id = "map";

    tf::StampedTransform tf_m2l;
    tf_m2l.setRotation(tf::Quaternion(cur_laser_pose_.pose.orientation.x, cur_laser_pose_.pose.orientation.y, cur_laser_pose_.pose.orientation.z, cur_laser_pose_.pose.orientation.w));
    tf_m2l.setOrigin(tf::Vector3(cur_laser_pose_.pose.position.x, cur_laser_pose_.pose.position.y, cur_laser_pose_.pose.position.z));
    tf_m2l.stamp_ = cur_laser_pose_.header.stamp;
    tf_m2l.frame_id_ = "map";
    tf_m2l.child_frame_id_ = "/laser";
    tf_broadcaster_.sendTransform(tf_m2l);

    pub_lol_pose_.publish(cur_laser_pose_);

    duration.sleep();
    ros::spinOnce();
  }
}

void LolLocalization::extractSurroundKeyFrames(const PointType &p)
{
  ROS_INFO("extract surround keyframes");
  // pc_surround_keyposes_->clear();
  pcl::PointCloud<PointType>::Ptr tmp_keyposes(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr tmp_keyposes_ds(new pcl::PointCloud<PointType>());
  kdtree_keyposes_3d_->radiusSearch(p, double(surround_search_radius_), point_search_idx_, point_search_dist_, 0);
  for (int i = 0; i < point_search_idx_.size(); ++i)
  {
    tmp_keyposes->points.push_back(keyposes_3d_->points[point_search_idx_[i]]);
  }
  ds_surround_keyposes_.setInputCloud(tmp_keyposes);
  ds_surround_keyposes_.filter(*tmp_keyposes_ds);

  bool existing_flag = false;
  // 1. 剔除多余位姿和点云
  for (int i = 0; i < surround_keyposes_id_.size(); ++i)
  {
    existing_flag = false;
    for (int j = 0; j < tmp_keyposes_ds->points.size(); ++j)
    {
      if (int(tmp_keyposes_ds->points[j].intensity) == surround_keyposes_id_[i])
      {
        existing_flag = true;
        break;
      }
    }

    if (!existing_flag)
    {
      surround_keyposes_id_.erase(surround_keyposes_id_.begin() + i);
      surround_corner_keyframes_.erase(surround_corner_keyframes_.begin() + i);
      surround_surf_keyframes_.erase(surround_surf_keyframes_.begin() + i);
      surround_outlier_keyframes_.erase(surround_outlier_keyframes_.begin() + i);
      --i;
    }
  }

  // 2. 添加缺少的位姿和点云
  for (int i = 0; i < tmp_keyposes_ds->points.size(); ++i)
  {
    existing_flag = false;
    int pose_id = int(tmp_keyposes_ds->points[i].intensity);
    for (int j = 0; j < surround_keyposes_id_.size(); ++j)
    {
      if (pose_id == surround_keyposes_id_[j])
      {
        existing_flag = true;
        break;
      }
    }

    if (!existing_flag)
    {
      surround_keyposes_id_.push_back(pose_id);
      surround_corner_keyframes_.push_back(corner_keyframes_[pose_id]);
      surround_surf_keyframes_.push_back(surf_keyframes_[pose_id]);
      surround_outlier_keyframes_.push_back(outlier_keyframes_[pose_id]);
    }
  }

  pc_corner_target_->clear();
  pc_surf_target_->clear();
  pc_corner_target_ds_->clear();
  pc_surf_target_ds_->clear();

  for (int i = 0; i < surround_keyposes_id_.size(); ++i)
  {
    *pc_corner_target_ += *(surround_corner_keyframes_[i]);
    *pc_surf_target_ += *(surround_surf_keyframes_[i]);
    *pc_surf_target_ += *(surround_outlier_keyframes_[i]);
  }

  ds_corner_.setInputCloud(pc_corner_target_);
  ds_corner_.filter(*pc_corner_target_ds_);
  ds_surf_.setInputCloud(pc_surf_target_);
  ds_surf_.filter(*pc_surf_target_ds_);

  sensor_msgs::PointCloud2 msg_corner_target, msg_surf_target;
  pcl::toROSMsg(*pc_corner_target_ds_, msg_corner_target);
  pcl::toROSMsg(*pc_surf_target_ds_, msg_surf_target);
  msg_corner_target.header.stamp = ros::Time::now();
  msg_corner_target.header.frame_id = "map";
  msg_surf_target.header = msg_corner_target.header;
  pub_corner_target_.publish(msg_corner_target);
  pub_surf_target_.publish(msg_surf_target);

  ROS_INFO("pc_corner_target_ds_: %d, pc_surf_target_ds_: %d", pc_corner_target_ds_->points.size(), pc_surf_target_ds_->points.size());
}

void LolLocalization::downsampleCurrentScan()
{
  ROS_INFO("before downsample: corner size %d, surf size %d", laser_corner_->points.size(), laser_surf_->points.size() + laser_outlier_->points.size());

  laser_corner_ds_->clear();
  laser_surf_ds_->clear();
  laser_outlier_ds_->clear();
  ds_corner_.setInputCloud(laser_corner_);
  ds_corner_.filter(*laser_corner_ds_);
  ds_surf_.setInputCloud(laser_surf_);
  ds_surf_.filter(*laser_surf_ds_);
  ds_outlier_.setInputCloud(laser_outlier_);
  ds_outlier_.filter(*laser_outlier_ds_);

  *laser_surf_ds_ += *laser_outlier_ds_;

  sensor_msgs::PointCloud2 msg_corner_source, msg_surf_source;
  pcl::toROSMsg(*laser_corner_ds_, msg_corner_source);
  pcl::toROSMsg(*laser_surf_ds_, msg_surf_source);
  msg_corner_source.header.stamp = ros::Time::now();
  msg_corner_source.header.frame_id = "/laser";
  msg_surf_source.header = msg_corner_source.header;

  pub_corner_source_.publish(msg_corner_source);
  pub_surf_source_.publish(msg_surf_source);

  ROS_INFO("after downsample: corner size %d, surf size %d", laser_corner_ds_->points.size(), laser_surf_ds_->points.size());
}

void LolLocalization::scanToMapRegistration()
{
  ROS_INFO("scanToMapRegistration");
  if (laser_corner_ds_->points.size() < 10 || laser_surf_ds_->points.size() < 100)
  {
    ROS_WARN("few feature points to registration");
    return;
  }
  // TODO: 使用最新的位姿数据（odomThread 处获得）、特征点数据（downsample 之后的）与局部 target map 做配准

  kdtree_corner_target_->setInputCloud(pc_corner_target_ds_);
  kdtree_surf_target_->setInputCloud(pc_surf_target_ds_);

  // 更新优化初值
  const Eigen::Matrix3f R_m2l = Eigen::Quaternionf(cur_laser_pose_.pose.orientation.w, cur_laser_pose_.pose.orientation.x, cur_laser_pose_.pose.orientation.y, cur_laser_pose_.pose.orientation.z).toRotationMatrix();
  const Eigen::Vector3f T_m2l(cur_laser_pose_.pose.position.x, cur_laser_pose_.pose.position.y, cur_laser_pose_.pose.position.z);
  Eigen::Vector3f ypr = R_m2l.eulerAngles(2, 1, 0);
  tobe_optimized_[0] = cur_laser_pose_.pose.position.x;
  tobe_optimized_[1] = cur_laser_pose_.pose.position.y;
  tobe_optimized_[2] = cur_laser_pose_.pose.position.z;
  tobe_optimized_[3] = ypr(0);
  tobe_optimized_[4] = ypr(1);
  tobe_optimized_[5] = ypr(2);
  R_tobe_ = R_m2l;
  T_tobe_ = T_m2l;

  ROS_INFO("init optimize");

  // part1: corner constraint
  for (int i = 0; i < laser_corner_ds_->points.size(); ++i)
  {
    // TODO: 调参
    problem_.AddResidualBlock(new CornerCostFunction(lol_, Eigen::Vector3f(laser_corner_ds_->points[i].x, laser_corner_ds_->points[i].y, laser_corner_ds_->points[i].z)),
                              new ceres::HuberLoss(1), tobe_optimized_);
  }

  // part2: surf constraint
  for (int i = 0; i < laser_surf_ds_->points.size(); ++i)
  {
    problem_.AddResidualBlock(new SurfCostFunction(lol_, Eigen::Vector3f(laser_surf_ds_->points[i].x, laser_surf_ds_->points[i].y, laser_surf_ds_->points[i].z)),
                              new ceres::HuberLoss(1), tobe_optimized_);
  }

  ROS_INFO("start optimize");
  auto start = std::chrono::system_clock::now();

  ceres::Solve(options_, &problem_, &summary_);

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  ROS_INFO("time elapesd: %f\n%s", elapsed.count(), summary_.BriefReport().c_str());

  // TODO: 最好能仅对有效的 ResidualBlock 优化，也就是 Evaluate 返回为 true 的
}

void LolLocalization::transformUpdate()
{
  ROS_INFO("transformUpdate");
  pre_laser_pose_ = cur_laser_pose_;
  Eigen::Quaternionf q_cur = Eigen::AngleAxisf(tobe_optimized_[5], Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(tobe_optimized_[4], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(tobe_optimized_[3], Eigen::Vector3f::UnitX());
  cur_laser_pose_.header.stamp = ros::Time::now();
  cur_laser_pose_.header.frame_id = "map";
  cur_laser_pose_.pose.position.x = tobe_optimized_[0];
  cur_laser_pose_.pose.position.y = tobe_optimized_[1];
  cur_laser_pose_.pose.position.z = tobe_optimized_[2];
  cur_laser_pose_.pose.orientation.w = q_cur.w();
  cur_laser_pose_.pose.orientation.x = q_cur.x();
  cur_laser_pose_.pose.orientation.y = q_cur.y();
  cur_laser_pose_.pose.orientation.z = q_cur.z();

  pub_lol_pose_.publish(cur_laser_pose_);
}

void LolLocalization::run()
{
  ros::Duration duration(0.5);
  PointType cur_pose;

  while (ros::ok())
  {

    duration.sleep();
    ros::spinOnce();

    if (!new_laser_corner_ || !new_laser_surf_ || !new_laser_outlier_ || !new_odom_)
    {
      continue;
    }

    cur_pose.x = cur_laser_pose_.pose.position.x;
    cur_pose.y = cur_laser_pose_.pose.position.y;
    cur_pose.z = cur_laser_pose_.pose.position.z;

    // 目前就只看水平的偏移
    if (hypot(cur_pose.x - target_center_.x, cur_pose.y - target_center_.y) > target_update_dist_)
    {
      extractSurroundKeyFrames(cur_pose);
    }

    downsampleCurrentScan();

    scanToMapRegistration();

    transformUpdate();

    new_laser_corner_ = new_laser_surf_ = new_laser_outlier_ = new_odom_ = false;
  }
}

CallbackReturnType LolLocalization::IterCB::operator()(const IterationSummary &summary)
{

  ROS_INFO("iteration cb, effect_residuals %d", lol_->effect_residuals_);
  lol_->effect_residuals_ = 0;
  lol_->R_tobe_ = Eigen::AngleAxisf(lol_->tobe_optimized_[5], Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(lol_->tobe_optimized_[4], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(lol_->tobe_optimized_[3], Eigen::Vector3f::UnitX());
  lol_->T_tobe_ = Eigen::Vector3f(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
  static Eigen::Vector3f pre_r, pre_t;
  if (summary.iteration == 0)
  {
    pre_r = Eigen::Vector3f(lol_->tobe_optimized_[3], lol_->tobe_optimized_[4], lol_->tobe_optimized_[5]);
    pre_t = Eigen::Vector3f(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
  }
  else
  {
    float delta_r = std::sqrt(std::pow(pcl::rad2deg(lol_->tobe_optimized_[3] - pre_r(0)), 2) +
                              std::pow(lol_->tobe_optimized_[4] - pre_r(1), 2) +
                              std::pow(lol_->tobe_optimized_[5] - pre_r(2), 2));
    float delta_t = std::sqrt(std::pow(pcl::rad2deg(lol_->tobe_optimized_[0] - pre_t(0)), 2) +
                              std::pow(lol_->tobe_optimized_[1] - pre_t(1), 2) +
                              std::pow(lol_->tobe_optimized_[2] - pre_t(2), 2));
    // 变化较小，提前终止迭代
    if (delta_r < 0.05 && delta_t < 0.05)
    {
      return SOLVER_TERMINATE_SUCCESSFULLY;
    }
    pre_r = Eigen::Vector3f(lol_->tobe_optimized_[3], lol_->tobe_optimized_[4], lol_->tobe_optimized_[5]);
    pre_t = Eigen::Vector3f(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
  }

  return SOLVER_CONTINUE;
}

// TODO: 写个单元测试看看对不对，还有 gradient check
// 能不能写个把 sin, cos 存在表里，到时候直接查表提高计算速度
bool LolLocalization::CornerCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  // 将点投影到 map 中，近邻查找地图中最近的 corner，根据该 corner 所在直线约束该点
  // 通过 PCA 地图中附近的 corner 点的主成分得到直线的方向，进而计算得到 p_j, p_l
  const Eigen::Vector3f p_i = lol_->R_tobe_ * p_o_ + lol_->T_tobe_;
  PointType p;
  p.x = p_i.x();
  p.y = p_i.y();
  p.z = p_i.z();
  lol_->kdtree_corner_target_->nearestKSearch(p, 5, lol_->point_search_idx_, lol_->point_search_dist_);

  // TODO: 测试阈值
  if (lol_->point_search_dist_[4] < 1.)
  {
    float cx = 0, cy = 0, cz = 0;
    for (int j = 0; j < 5; ++j)
    {
      cx += lol_->pc_corner_target_ds_->points[lol_->point_search_idx_[j]].x;
      cy += lol_->pc_corner_target_ds_->points[lol_->point_search_idx_[j]].y;
      cz += lol_->pc_corner_target_ds_->points[lol_->point_search_idx_[j]].z;
    }
    cx /= 5;
    cy /= 5;
    cz /= 5;

    float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    float ax, ay, az;
    for (int j = 0; j < 5; ++j)
    {
      ax = lol_->pc_corner_target_ds_->points[j].x - cx;
      ay = lol_->pc_corner_target_ds_->points[j].y - cy;
      az = lol_->pc_corner_target_ds_->points[j].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }
    a11 /= 5;
    a12 /= 5;
    a13 /= 5;
    a22 /= 5;
    a23 /= 5;
    a33 /= 5;
    Eigen::Matrix3f A;
    A << a11, a12, a13, a12, a22, a23, a13, a23, a33;
    Eigen::EigenSolver<Eigen::Matrix3f> es(A);
    Eigen::Matrix3f D = es.pseudoEigenvalueMatrix();
    Eigen::Matrix3f V = es.pseudoEigenvalueMatrix();
    int max_r, max_c;
    int min_r, min_c;
    int mid_r, mid_c;
    D.maxCoeff(&max_r, &max_c);
    D.minCoeff(&min_r, &min_c);
    mid_r = 3 - (max_r + min_r);
    mid_c = mid_r;

    // 得保证最大特征值远大于其余特征值，才能保证求解的是直线上的 corner
    // TODO: 调参
    if (D(max_r, max_c) > 3 * D(mid_r, mid_c))
    {
      Eigen::Vector3f p_j(cx + 0.1 * V(0, max_r), cy + 0.1 * V(1, max_r), cz + 0.1 * V(2, max_r));
      Eigen::Vector3f p_l(cx - 0.1 * V(0, max_r), cy - 0.1 * V(1, max_r), cz - 0.1 * V(2, max_r));

      double k = std::sqrt(std::pow(p_j.x() - p_l.x(), 2) + std::pow(p_j.y() - p_l.y(), 2) + std::pow(p_j.z() - p_l.z(), 2));
      double a = (p_i.y() - p_j.y()) * (p_i.z() - p_l.z()) - (p_i.z() - p_j.z()) * (p_i.y() - p_l.y());
      double b = (p_i.z() - p_j.z()) * (p_i.x() - p_l.x()) - (p_i.x() - p_j.x()) * (p_i.z() - p_l.z());
      double c = (p_i.x() - p_j.x()) * (p_i.y() - p_i.y()) - (p_i.y() - p_j.y()) * (p_i.x() - p_l.x());
      double m = std::sqrt(a * a + b * b + c * c);

      ROS_INFO("k: %f, m: %f", k, m);

      residuals[0] = m / k;

      double dm_dx = (b * (p_j.z() - p_l.z()) + c * (p_j.y() - p_l.y())) / m;
      double dm_dy = (a * (p_j.z() - p_l.z()) - c * (p_j.x() - p_l.x())) / m;
      double dm_dz = (-a * (p_j.y() - p_l.y()) - b * (p_j.x() - p_l.x())) / m;

      double sr = std::sin(parameters[0][3]);
      double cr = std::cos(parameters[0][3]);
      double sp = std::sin(parameters[0][4]);
      double cp = std::cos(parameters[0][4]);
      double sy = std::sin(parameters[0][5]);
      double cy = std::cos(parameters[0][5]);

      double dx_dr = (cy * sp * cr + sr * sy) * p_o_.y() + (sy * cr - cy * sr * sp) * p_o_.z();
      double dy_dr = (-cy * sr + sy * sp * cr) * p_o_.y() + (-sr * sy * sp - cy * cr) * p_o_.z();
      double dz_dr = cp * cr * p_o_.y() - cp * sr * p_o_.z();

      double dx_dp = -cy * sp * p_o_.x() + cy * cp * sr * p_o_.y() + cy * cr * cp * p_o_.z();
      double dy_dp = -sp * sy * p_o_.x() + sy * cp * sr * p_o_.y() + cr * sr * cp * p_o_.z();
      double dz_dp = -cp * p_o_.x() - sp * sr * p_o_.y() - sp * cr * p_o_.z();

      double dx_dy = -sy * cp * p_o_.x() - (sy * sp * sr + cr * cy) * p_o_.y() + (cy * sr - sy * cr * sp) * p_o_.z();
      double dy_dy = cp * cy * p_o_.x() + (-sy * cr + cy * sp * sr) * p_o_.y() + (cy * cr * sp + sy * sr) * p_o_.z();
      double dz_dy = 0.;

      if (jacobians && jacobians[0])
      {
        jacobians[0][0] = dm_dx / k;
        jacobians[0][1] = dm_dy / k;
        jacobians[0][2] = dm_dz / k;
        jacobians[0][3] = (dm_dx * dx_dr + dm_dy * dy_dr + dm_dz * dz_dr) / k;
        jacobians[0][4] = (dm_dx * dx_dp + dm_dy * dy_dp + dm_dz * dz_dp) / k;
        jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;
      }

      ++lol_->effect_residuals_;

      ROS_INFO("test corner");

      return true;
    }
  }

  residuals[0] = 0;
  if (!jacobians && !jacobians[0])
  {
    jacobians[0][0] = 0;
    jacobians[0][1] = 0;
    jacobians[0][2] = 0;
    jacobians[0][3] = 0;
    jacobians[0][4] = 0;
    jacobians[0][5] = 0;
  }

  return true;
}

bool LolLocalization::SurfCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  const Eigen::Vector3f p_i = lol_->R_tobe_ * p_o_ + lol_->T_tobe_;
  PointType p;
  p.x = p_i.x();
  p.y = p_i.y();
  p.z = p_i.z();
  lol_->kdtree_surf_target_->nearestKSearch(p, 5, lol_->point_search_idx_, lol_->point_search_dist_);

  if (lol_->point_search_dist_[4] < 1.0)
  {
    Eigen::Matrix<float, 5, 3> A;
    Eigen::Matrix<float, 5, 1> B;
    B << -1, -1, -1, -1, -1;
    for (int j = 0; j < 5; ++j)
    {
      A(j, 0) = lol_->pc_surf_target_ds_->points[j].x;
      A(j, 1) = lol_->pc_surf_target_ds_->points[j].y;
      A(j, 2) = lol_->pc_surf_target_ds_->points[j].z;
    }
    Eigen::Vector3f X = A.colPivHouseholderQr().solve(B);
    float pa = X(0);
    float pb = X(1);
    float pc = X(2);
    float pd = 1;
    float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
    bool plane_valid = true;
    for (int j = 0; j < 5; ++j)
    {
      if (std::fabs(pa * A(j, 0) + pb * A(j, 1) + pc * A(j, 2) + pd) / ps > 0.2)
      {
        plane_valid = false;
        break;
      }
    }

    if (plane_valid)
    {
      double m = pa * p_i.x() + pb * p_i.y() + pc * p_i.z() + pd;
      double k = ps;

      residuals[0] = m / k;

      double dm_dx, dm_dy, dm_dz;
      if (m < 0.)
      {
        dm_dx = -pa;
        dm_dy = -pb;
        dm_dz = -pc;
      }
      else
      {
        dm_dx = pa;
        dm_dy = pb;
        dm_dz = pc;
      }

      double sr = std::sin(parameters[0][3]);
      double cr = std::cos(parameters[0][3]);
      double sp = std::sin(parameters[0][4]);
      double cp = std::cos(parameters[0][4]);
      double sy = std::sin(parameters[0][5]);
      double cy = std::cos(parameters[0][5]);

      double dx_dr = (cy * sp * cr + sr * sy) * p_o_.y() + (sy * cr - cy * sr * sp) * p_o_.z();
      double dy_dr = (-cy * sr + sy * sp * cr) * p_o_.y() + (-sr * sy * sp - cy * cr) * p_o_.z();
      double dz_dr = cp * cr * p_o_.y() - cp * sr * p_o_.z();

      double dx_dp = -cy * sp * p_o_.x() + cy * cp * sr * p_o_.y() + cy * cr * cp * p_o_.z();
      double dy_dp = -sp * sy * p_o_.x() + sy * cp * sr * p_o_.y() + cr * sr * cp * p_o_.z();
      double dz_dp = -cp * p_o_.x() - sp * sr * p_o_.y() - sp * cr * p_o_.z();

      double dx_dy = -sy * cp * p_o_.x() - (sy * sp * sr + cr * cy) * p_o_.y() + (cy * sr - sy * cr * sp) * p_o_.z();
      double dy_dy = cp * cy * p_o_.x() + (-sy * cr + cy * sp * sr) * p_o_.y() + (cy * cr * sp + sy * sr) * p_o_.z();
      double dz_dy = 0.;

      if (jacobians && jacobians[0])
      {
        jacobians[0][0] = dm_dx / k;
        jacobians[0][1] = dm_dy / k;
        jacobians[0][2] = dm_dz / k;
        jacobians[0][3] = (dm_dx * dx_dr + dm_dy * dy_dr + dm_dz * dz_dr) / k;
        jacobians[0][4] = (dm_dx * dx_dp + dm_dy * dy_dp + dm_dz * dz_dp) / k;
        jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;
      }

      ++lol_->effect_residuals_;

      return true;
    }
  }
  residuals[0] = 0;

  if (!jacobians && !jacobians[0])
  {
    jacobians[0][0] = 0;
    jacobians[0][1] = 0;
    jacobians[0][2] = 0;
    jacobians[0][3] = 0;
    jacobians[0][4] = 0;
    jacobians[0][5] = 0;
  }

  return true;
}

} // namespace localization
