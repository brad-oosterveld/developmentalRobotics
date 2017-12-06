#include <iostream>
#include <ros/ros.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>

std::string wd = "/home/tmfrasca/tufts/2017/developmental_robotics/project/developmentalRobotics/";
std::string imgDir = wd + "image_samples/";
std::string featureDir = wd + "feature_extractions/";

void estimateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals) {

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (0.03);
  ne.compute(*normals);
}

void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs) {
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  estimateNormals(cloud, normals);
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
  vfh.setSearchMethod(tree);
  vfh.compute(*vfhs);
}

void viewCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::visualization::CloudViewer viewer("pc viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
  }
}

void viewHistogram(pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs) {
  pcl::visualization::PCLHistogramVisualizer *vis = new pcl::visualization::PCLHistogramVisualizer();
  vis->addFeatureHistogram(*vfhs,500);
  vis->spinOnce(100);
}

void saveHistogram(pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, std::string filename) {
  pcl::io::savePCDFileASCII(filename, *vfhs);

}

int main(int argc, char** argv) { pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  std::ifstream imgLabelFile;
  imgLabelFile.open(wd+"img_labels.txt", ios::in);
  std::vector<std::string> objFilenames;
  std::string fileData;
  
  std::cout<<imgLabelFile.is_open()<<std::endl;
  while (getline(imgLabelFile, fileData)) {
    std::string objFilename = fileData.substr(0, fileData.find_first_of(","));
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (imgDir + objFilename, *cloud) == -1) {
      PCL_ERROR("could not read file\n");
      return(-1);
    }
      //std::cout<<"opening and feature extracting " + objFilename<<std::endl; 
      //viewCloud(cloud);
      pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>());
      extractFeatures(cloud, vfhs);
      //viewHistogram(vfhs);
      //std::cout<<"writing to " + featureDir + objFilename<<std::endl; 
      saveHistogram(vfhs, featureDir + objFilename);

  }
  imgLabelFile.close();
  
  return(0);

}

