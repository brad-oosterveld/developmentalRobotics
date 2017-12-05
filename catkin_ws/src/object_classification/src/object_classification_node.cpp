#include <ros/ros.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/historgram_visualizer.h>

#include <iostream>

std::string dir = "/home/tmfrasca/tufts/2017/developmental_robotics/project/rgbd-dataset/";
void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
  vfh.setSearchMethod(tree);
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>());
  vfh.compute(*vfhs);
}

void viewCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::visualization::CloudViewer viewer("pc viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
  }
}

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (dir + "apple/apple_1/apple_1_1_1.pcd", *cloud) == -1) {
    PCL_ERROR("could not read file\n");
    return(-1);
  }
  std::cout << "loaded "
            << cloud->width * cloud->height
            << " data points"<< std::endl;


  viewCloud(cloud);
  extractFeatures(cloud);


  return(0);

}

