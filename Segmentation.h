#pragma once

#include <iostream>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace ark {
	/**
	* Utility class that allows segmentation (clustering) of an unordered point cloud.
	*/
	class Segmentation
	{
	private:
		std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> mvecpCluster;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plane;

		/**
		* Removes the plane from the objects from a point cloud.
		* @param [in] cloud
		* @param [out] cloud_plane the cloud representing plane.
		* @param [out] cloud_objs the cloud representing objects (the rest of points).
		*/
		void removePlane(
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane,
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_objs
		);

	public:
		/**
		* Reads a .PCD file.
		* @param [in] filepath local filepath and filename of .PCD file.
		* @param [out] cloud the cloud that was stored in the .PCD file
		*/
		void readPcd(std::string filepath, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

		/**
		* Computes the clusters of an input point cloud.
		* @param [in] cloud
		* @return vector of clusters, where each is a point cloud pointer
		*/
		std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> computeClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

		/**
		* Performs segmentation on a point cloud of a scene.
		* Call `getCluster` to retrieve the clusters.
		* @param [in] cloud scene point cloud
		*/
		void segment(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

		/**
		* Retrieves the clusters without the plane. To be called after `segment`.
		* @return vector of clusters, where each is a PointCloud
		*/
		std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> getCluster()
		{
			return mvecpCluster;
		};

		/**
		* Retrieves the plane cluster. To be called after `segment`.
		* @return plane cluster (PointCloud)
		*/
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPlane()
		{
			return plane;
		};
	};
}