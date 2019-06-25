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

  tf_b2l_ = Eigen::Matrix4d::Identity();
  float roll, pitch, yaw;
  if (!nh_->getParam("tf_b2l_x", tf_b2l_(0, 3)) || !nh_->getParam("tf_b2l_y", tf_b2l_(1, 3)) || !nh_->getParam("tf_b2l_z", tf_b2l_(2, 3)) || !nh_->getParam("tf_b2l_roll", roll) || !nh_->getParam("tf_b2l_pitch", pitch) || !nh_->getParam("tf_b2l_yaw", yaw))
  {
    ROS_ERROR("transform between /base_link to /laser not set.");
    exit(-1);
  }
  Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
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
  
  TicToc t_data;
  if (pcl::io::loadPCDFile(fn_poses_, *keyposes_3d_) == -1 || pcl::io::loadPCDFile(fn_corner_, *corner_pc) == -1 ||
      pcl::io::loadPCDFile(fn_surf_, *surf_pc) == -1 || pcl::io::loadPCDFile(fn_outlier_, *outlier_pc) == -1)
  {
    ROS_ERROR("couldn't load pcd file");
    return false;
  }

  ROS_INFO("time: %f s ----> keyposes: %d, corner pc: %d, surf pc: %d, outlier pc: %d", t_data.toc(), keyposes_3d_->points.size(), corner_pc->points.size(), surf_pc->points.size(), outlier_pc->points.size());

  kdtree_keyposes_3d_->setInputCloud(keyposes_3d_);

  corner_keyframes_.resize(keyposes_3d_->points.size());
  surf_keyframes_.resize(keyposes_3d_->points.size());
  outlier_keyframes_.resize(keyposes_3d_->points.size());
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
  // options_.minimizer_progress_to_stdout = true;
  options_.linear_solver_type = ceres::DENSE_QR;
  options_.max_num_iterations = 20;
  // options_.initial_trust_region_radius = 1e2;
  // options_.max_trust_region_radius = 1e6;
  // options_.min_trust_region_radius = 1e-10;
  options_.gradient_check_relative_precision = 1e-4;
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
      Eigen::Quaterniond q_b(msg_odoms_[ptr_back].pose.pose.orientation.w, msg_odoms_[ptr_back].pose.pose.orientation.x, msg_odoms_[ptr_back].pose.pose.orientation.y, msg_odoms_[ptr_back].pose.pose.orientation.z);
      Eigen::Quaterniond q_f(msg_odoms_[ptr].pose.pose.orientation.w, msg_odoms_[ptr].pose.pose.orientation.x, msg_odoms_[ptr].pose.pose.orientation.y, msg_odoms_[ptr].pose.pose.orientation.z);
      auto euler_b = q_b.toRotationMatrix().eulerAngles(2, 1, 0);
      auto euler_f = q_f.toRotationMatrix().eulerAngles(2, 1, 0);
      Eigen::Vector3d euler = (1 - ratio_back) * euler_b + ratio_back * euler_f;
      Eigen::Quaterniond q_cur = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitX());

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
      Eigen::Matrix4d T_olp = Eigen::Matrix4d::Identity();
      Eigen::Quaterniond q_olc(cur_laser_pose_.pose.orientation.w, cur_laser_pose_.pose.orientation.x, cur_laser_pose_.pose.orientation.y, cur_laser_pose_.pose.orientation.z);
      Eigen::Quaterniond q_ooc(cur_laser_odom.pose.pose.orientation.w, cur_laser_odom.pose.pose.orientation.x, cur_laser_odom.pose.pose.orientation.y, cur_laser_odom.pose.pose.orientation.z);
      Eigen::Quaterniond q_oor(recent_laser_odom.pose.pose.orientation.w, recent_laser_odom.pose.pose.orientation.x, recent_laser_odom.pose.pose.orientation.y, recent_laser_odom.pose.pose.orientation.z);
      Eigen::Quaterniond q_olp = q_olc * q_ooc.inverse() * q_oor;
      Eigen::Vector3d t_olc(cur_laser_pose_.pose.position.x, cur_laser_pose_.pose.position.y, cur_laser_pose_.pose.position.z);
      Eigen::Vector3d t_ooc(cur_laser_odom.pose.pose.position.x, cur_laser_odom.pose.pose.position.y, cur_laser_odom.pose.pose.position.z);
      Eigen::Vector3d t_oor(recent_laser_odom.pose.pose.position.x, recent_laser_odom.pose.pose.position.y, recent_laser_odom.pose.pose.position.z);
      Eigen::Vector3d t_olp = t_olc - q_olc * q_ooc.inverse() * (t_ooc - t_oor);
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
  const Eigen::Matrix3d R_m2l = Eigen::Quaterniond(cur_laser_pose_.pose.orientation.w, cur_laser_pose_.pose.orientation.x, cur_laser_pose_.pose.orientation.y, cur_laser_pose_.pose.orientation.z).toRotationMatrix();
  const Eigen::Vector3d T_m2l(cur_laser_pose_.pose.position.x, cur_laser_pose_.pose.position.y, cur_laser_pose_.pose.position.z);
  Eigen::Vector3d ypr = R_m2l.eulerAngles(2, 1, 0);
  tobe_optimized_[0] = cur_laser_pose_.pose.position.x;
  tobe_optimized_[1] = cur_laser_pose_.pose.position.y;
  tobe_optimized_[2] = cur_laser_pose_.pose.position.z;
  tobe_optimized_[3] = ypr(0);
  tobe_optimized_[4] = ypr(1);
  tobe_optimized_[5] = ypr(2);
  R_tobe_ = R_m2l;
  T_tobe_ = T_m2l;

  ROS_INFO("init optimize");

  for (int iter_cnt = 0; iter_cnt < 2; ++iter_cnt)
  {
    TicToc t_data;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    // problem.AddParameterBlock(parameters, 4, q_parameterization);
    problem.AddParameterBlock(tobe_optimized_, 6);

    int corner_correspondace = 0, surf_correnspondance = 0;
    int test_p = 0;

    // part1: corner constraint
    for (int i = 0; i < laser_corner_ds_->points.size(); ++i)
    {
      PointType point_sel;
      pointAssociateToMap(laser_corner_ds_->points[i], point_sel);
      kdtree_corner_target_->nearestKSearch(point_sel, 5, point_search_idx_, point_search_dist_);
      if (point_search_dist_[4] < 1.0)
      {
        std::vector<Eigen::Vector3d> nearCorners;
        Eigen::Vector3d center(0., 0., 0.);
        for (int j = 0; j < 5; j++)
        {
          Eigen::Vector3d tmp(pc_corner_target_ds_->points[point_search_idx_[j]].x,
                              pc_corner_target_ds_->points[point_search_idx_[j]].y,
                              pc_corner_target_ds_->points[point_search_idx_[j]].z);
          center = center + tmp;
          nearCorners.push_back(tmp);
        }
        center = center / 5.0;

        Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
        for (int j = 0; j < 5; j++)
        {
          Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
          covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

        // if is indeed line feature
        // note Eigen library sort eigenvalues in increasing order
        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
        Eigen::Vector3d cp(laser_corner_ds_->points[i].x, laser_corner_ds_->points[i].y, laser_corner_ds_->points[i].z);
        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
        {
          Eigen::Vector3d point_on_line = center;
          Eigen::Vector3d lpj = 0.1 * unit_direction + point_on_line;
          Eigen::Vector3d lpl = -0.1 * unit_direction + point_on_line;
          problem.AddResidualBlock(new LidarEdgeCostFunction(cp, lpj, lpl),
                                   loss_function, tobe_optimized_);
          ++corner_correspondace;
        }
        else
        {
          ++test_p;
        }
      }
    }

    // part2: surf constraint
    for (int i = 0; i < laser_surf_ds_->points.size(); ++i)
    {
      PointT point_sel;
      pointAssociateToMap(laser_surf_ds_->points[i], point_sel);
      kdtree_surf_target_->nearestKSearch(point_sel, 5, point_search_idx_, point_search_dist_);
      Eigen::Matrix<double, 5, 3> matA0;
      Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
      if (point_search_dist_[4] < 1.0)
      {
        for (int j = 0; j < 5; j++)
        {
          matA0(j, 0) = pc_surf_target_ds_->points[point_search_idx_[j]].x;
          matA0(j, 1) = pc_surf_target_ds_->points[point_search_idx_[j]].y;
          matA0(j, 2) = pc_surf_target_ds_->points[point_search_idx_[j]].z;
        }
        // find the norm of plane
        Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
        double negative_OA_dot_norm = 1 / norm.norm();
        norm.normalize();

        // Here n(pa, pb, pc) is unit norm of plane
        bool planeValid = true;
        for (int j = 0; j < 5; j++)
        {
          // if OX * n > 0.2, then plane is not fit well
          if (fabs(norm(0) * pc_surf_target_ds_->points[point_search_idx_[j]].x +
                   norm(1) * pc_surf_target_ds_->points[point_search_idx_[j]].y +
                   norm(2) * pc_surf_target_ds_->points[point_search_idx_[j]].z + negative_OA_dot_norm) > 0.2)
          {
            planeValid = false;
            ROS_WARN_ONCE("plane is not fit well");
            break;
          }
        }
        if (planeValid)
        {
          Eigen::Vector3d cp(laser_surf_ds_->points[i].x, laser_surf_ds_->points[i].y, laser_surf_ds_->points[i].z);
          problem.AddResidualBlock(new LidarPlaneCostFunction(cp, norm, negative_OA_dot_norm),
                                   loss_function, tobe_optimized_);
          // TODO: 先解决 corner 数量过少的问题，少了十倍
          ++surf_correnspondance;
        }
      }
    }

    ROS_INFO("start optimize");

    ceres::Solve(options_, &problem_, &summary_);

    ROS_INFO("time elapesd: %.3fms\n%s", t_data.toc(), summary_.BriefReport().c_str());
  }
}

void LolLocalization::transformUpdate()
{
  ROS_INFO("transformUpdate");
  pre_laser_pose_ = cur_laser_pose_;
  Eigen::Quaterniond q_cur = Eigen::AngleAxisd(tobe_optimized_[5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(tobe_optimized_[4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(tobe_optimized_[3], Eigen::Vector3d::UnitX());
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
  lol_->R_tobe_ = Eigen::AngleAxisd(lol_->tobe_optimized_[5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(lol_->tobe_optimized_[4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(lol_->tobe_optimized_[3], Eigen::Vector3d::UnitX());
  lol_->T_tobe_ = Eigen::Vector3d(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
  static Eigen::Vector3d pre_r, pre_t;
  if (summary.iteration == 0)
  {
    pre_r = Eigen::Vector3d(lol_->tobe_optimized_[3], lol_->tobe_optimized_[4], lol_->tobe_optimized_[5]);
    pre_t = Eigen::Vector3d(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
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
    pre_r = Eigen::Vector3d(lol_->tobe_optimized_[3], lol_->tobe_optimized_[4], lol_->tobe_optimized_[5]);
    pre_t = Eigen::Vector3d(lol_->tobe_optimized_[0], lol_->tobe_optimized_[1], lol_->tobe_optimized_[2]);
  }

  return SOLVER_CONTINUE;
}

} // namespace localization
