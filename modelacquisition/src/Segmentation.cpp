#include "Segmentation.h"
#define SHOULD_DOWN_SAMPLE 1 // 1 true, 0 false

namespace ark {
	void Segmentation::readPcd(std::string filepath, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {
		if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(filepath, *cloud) == -1) {
			PCL_ERROR("File loading error");
		}
	}

	void Segmentation::removePlane(
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane,
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_objs
	) {
		// RANSAC plane fitting.
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.07);
		seg.setMaxIterations(100);

		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			PCL_ERROR("Could not estimate a planar model for the given dataset.");
			return;
		}

		pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
		// Extract plane inliers.
		extract.setInputCloud(cloud);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*cloud_plane);
		// Extract outliers as objects cloud (no plane).
		extract.setNegative(true);
		extract.filter(*cloud_objs);
	}

	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> Segmentation::computeClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {
		pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
		kdtree->setInputCloud(cloud);

		// Cluster point clouds using Euclidean distance.
		std::vector<pcl::PointIndices> cluster_indices; // Vector of multiple indices collections (clusters)
		pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> clustering;
		clustering.setClusterTolerance(0.04); // m
		clustering.setMinClusterSize(200);
		clustering.setMaxClusterSize(1000000);
		clustering.setSearchMethod(kdtree);
		clustering.setInputCloud(cloud);
		clustering.extract(cluster_indices);

		std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> clusters;
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
			for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
				cloud_cluster->points.push_back(cloud->points[*pit]);
			}
			cloud_cluster->width = cloud_cluster->points.size();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;
			clusters.push_back(cloud_cluster);
		}
		return clusters;
	}

	void Segmentation::segment(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_objs(new pcl::PointCloud<pcl::PointXYZRGBA>);

		// Down sample the dataset.
		if (SHOULD_DOWN_SAMPLE == 1) {
			pcl::VoxelGrid<pcl::PointXYZRGBA> vox;
			vox.setInputCloud(cloud);
			vox.setLeafSize(0.005f, 0.005f, 0.005f); // m
			vox.filter(*cloud_filtered);
		} else {
			cloud_filtered = cloud;
		}

		// Statistical outlier removal
		// TODO: should be handled in a sort of preprocessing stage
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
		sor.setInputCloud(cloud_filtered);
		sor.setMeanK(50);
		sor.setStddevMulThresh(1.0);
		sor.filter(*cloud_filtered);

		removePlane(cloud_filtered, cloud_plane, cloud_objs);
		plane = cloud_plane;
		mvecpCluster = computeClusters(cloud_objs);
	}
}
