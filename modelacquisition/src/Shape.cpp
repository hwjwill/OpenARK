#include "Shape.h"
#include <unordered_map>
#include <string>

namespace ark {
	Shape::Shape() {

	}

	Shape::~Shape() {

	}

	void Shaps::generateMeshFromPtnCld(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double smooth_radius) {
		// Denoise
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
		sor.setInputCloud(cloud);
		sor.setMeanK(100);
		sor.setStddevMulThresh(0.05);
		sor.filter(*cloud_filtered);
		// Translate to origin
		pcl::CentroidPoint<pcl::PointXYZRGB> centroid;
		for (int i = 0; i < cloud_filtered->points.size(); ++i)
			centroid.add(cloud_filtered->points[i]);
		pcl::PointXYZRGB pc;
		centroid.get(pc);
		Eigen::Matrix4f transform_mat = Eigen::Matrix4f::Identity();
		transform_mat(0, 3) = -pc.x;
		transform_mat(1, 3) = -pc.y;
		transform_mat(2, 3) = -pc.z;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, transform_mat);
		/*========================== Mesh Generation Part Starts =======================*/
		pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
		mls.setInputCloud(cloud_transformed);
		mls.setSearchRadius(smoothRadius);
		mls.setPolynomialFit(true);
		mls.setPolynomialOrder(2);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGB>());
		mls.process(*cloud_smoothed);

		// Normal estimation*
		pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> n;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
		tree->setInputCloud(cloud_smoothed);
		n.setInputCloud(cloud_smoothed);
		n.setSearchMethod(tree);
		n.setKSearch(20);
		n.compute(*normals);
		//* normals should contain the point normals + surface curvatures

		// Concatenate the XYZ and normal fields*
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed_copy(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*cloud_smoothed, *cloud_smoothed_copy);
		pcl::concatenateFields(*cloud_smoothed_copy, *normals, *cloud_with_normals);

		// Initialize objects
		pcl::Poisson<pcl::PointNormal> poisson;
		pcl::PolygonMesh mesh;
		poisson.setDepth(6);
		poisson.setInputCloud(cloud_with_normals);
		poisson.reconstruct(mesh);

		// Add color information
		pcl::PointCloud<pcl::PointXYZ> cloudData;
		pcl::fromPCLPointCloud2(mesh.cloud, cloudData);
		for (size_t i = 0; i < cloudData.points.size(); ++i) {
			pcl::PointXYZRGB p;
			p.x = cloudData.points[i].x;
			p.y = cloudData.points[i].y;
			p.z = cloudData.points[i].z;
			/* Edit this part to get true texture of the mesh */
			/*================================================*/
			double dis = DBL_MAX;
			int idx = 0;
			for (size_t j = 0; j < cloud_smoothed->points.size(); ++j) {
				double d = sqrt((p.x - cloud_smoothed->points[j].x) * (p.x - cloud_smoothed->points[j].x)
					+ (p.y - cloud_smoothed->points[j].y) * (p.y - cloud_smoothed->points[j].y)
					+ (p.z - cloud_smoothed->points[j].z) * (p.z - cloud_smoothed->points[j].z));
				if (dis > d) {
					dis = d;
					idx = j;
				}
			}
			p.r = cloud_smoothed->points[idx].r;
			p.g = cloud_smoothed->points[idx].g;
			p.b = cloud_smoothed->points[idx].b;
			/*================================================*/
			vertices.push_back(p);
		}
		for (auto f : mesh.polygons) {
			Face f_new;
			f_new.vertex_idx[0] = f.vertices[0];
			f_new.vertex_idx[1] = f.vertices[1];
			f_new.vertex_idx[2] = f.vertices[2];
			faces.push_back(f_new);
		}
		/*========================== Mesh Generation Part Ends =========================*/
		return;
	}
	
	void Shape::generateMeshFromTSDF(TSDFData input_tsdf) {
		int total_size = input_tsdf.voxel_grid_dim[0] * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2];
		std::unordered_map<std::string, int> vertices_idx;
		int vertex_count = 0;
		for (size_t i = 0; i < total_size; ++i) {
			if (i % 10000000 == 0)
				cout << i << " ";
			int xi = i / (input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2]);
			int yi = (i - xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2]) / input_tsdf.voxel_grid_dim[2];
			int zi = i - xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] - yi * input_tsdf.voxel_grid_dim[2];
			if (xi == input_tsdf.voxel_grid_dim[0] - 1 || yi == input_tsdf.voxel_grid_dim[1] - 1 || zi == input_tsdf.voxel_grid_dim[2] - 1)
				continue;
			GridCell grid;
			grid.p[0] = pcl::PointXYZRGB(xi, yi, zi);
			grid.p[1] = pcl::PointXYZRGB(xi, yi + 1, zi);
			grid.p[2] = pcl::PointXYZRGB(xi + 1, yi + 1, zi);
			grid.p[3] = pcl::PointXYZRGB(xi + 1, yi, zi);
			grid.p[4] = pcl::PointXYZRGB(xi, yi, zi + 1);
			grid.p[5] = pcl::PointXYZRGB(xi, yi + 1, zi + 1);
			grid.p[6] = pcl::PointXYZRGB(xi + 1, yi + 1, zi + 1);
			grid.p[7] = pcl::PointXYZRGB(xi + 1, yi, zi + 1);

			grid.val[0] = input_tsdf.tsdf[xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + yi * input_tsdf.voxel_grid_dim[2] + zi];
			grid.val[1] = input_tsdf.tsdf[xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + (yi + 1) * input_tsdf.voxel_grid_dim[2] + zi];
			grid.val[2] = input_tsdf.tsdf[(xi + 1) * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + (yi + 1) * input_tsdf.voxel_grid_dim[2] + zi];
			grid.val[3] = input_tsdf.tsdf[(xi + 1) * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + yi * input_tsdf.voxel_grid_dim[2] + zi];
			grid.val[4] = input_tsdf.tsdf[xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + yi * input_tsdf.voxel_grid_dim[2] + (zi + 1)];
			grid.val[5] = input_tsdf.tsdf[xi * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + (yi + 1) * input_tsdf.voxel_grid_dim[2] + (zi + 1)];
			grid.val[6] = input_tsdf.tsdf[(xi + 1) * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + (yi + 1) * input_tsdf.voxel_grid_dim[2] + (zi + 1)];
			grid.val[7] = input_tsdf.tsdf[(xi + 1) * input_tsdf.voxel_grid_dim[1] * input_tsdf.voxel_grid_dim[2] + yi * input_tsdf.voxel_grid_dim[2] + (zi + 1)];
			int cubeIndex = 0;
			if (grid.val[0] < 0) cubeIndex |= 1;
			if (grid.val[1] < 0) cubeIndex |= 2;
			if (grid.val[2] < 0) cubeIndex |= 4;
			if (grid.val[3] < 0) cubeIndex |= 8;
			if (grid.val[4] < 0) cubeIndex |= 16;
			if (grid.val[5] < 0) cubeIndex |= 32;
			if (grid.val[6] < 0) cubeIndex |= 64;
			if (grid.val[7] < 0) cubeIndex |= 128;
			pcl::PointXYZRGB vertlist[12];
			if (edge_table[cubeIndex] == 0)
				continue;

			/* Find the vertices where the surFace intersects the cube */
			if (edge_table[cubeIndex] & 1)
				vertlist[0] =
				interpolateVertex(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
			if (edge_table[cubeIndex] & 2)
				vertlist[1] =
				interpolateVertex(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
			if (edge_table[cubeIndex] & 4)
				vertlist[2] =
				interpolateVertex(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
			if (edge_table[cubeIndex] & 8)
				vertlist[3] =
				interpolateVertex(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
			if (edge_table[cubeIndex] & 16)
				vertlist[4] =
				interpolateVertex(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
			if (edge_table[cubeIndex] & 32)
				vertlist[5] =
				interpolateVertex(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
			if (edge_table[cubeIndex] & 64)
				vertlist[6] =
				interpolateVertex(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
			if (edge_table[cubeIndex] & 128)
				vertlist[7] =
				interpolateVertex(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
			if (edge_table[cubeIndex] & 256)
				vertlist[8] =
				interpolateVertex(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
			if (edge_table[cubeIndex] & 512)
				vertlist[9] =
				interpolateVertex(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
			if (edge_table[cubeIndex] & 1024)
				vertlist[10] =
				interpolateVertex(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
			if (edge_table[cubeIndex] & 2048)
				vertlist[11] =
				interpolateVertex(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

			/* Create the Triangle */
			for (int ti = 0; triangle_table[cubeIndex][ti] != -1; ti += 3) {
				Face f;
				Triangle t;
				t.p[0] = vertlist[triangle_table[cubeIndex][ti]];
				t.p[1] = vertlist[triangle_table[cubeIndex][ti + 1]];
				t.p[2] = vertlist[triangle_table[cubeIndex][ti + 2]];
				for (int pi = 0; pi < 3; ++pi) {
					std::string s = "x" + std::to_string(t.p[pi].x) + "y" + std::to_string(t.p[pi].y) + "z" + std::to_string(t.p[pi].z);
					if (vertices_idx.find(s) == vertices_idx.end()) {
						vertices_idx.insert(make_pair(s, vertex_count));
						f.vertex_idx[pi] = vertex_count++;
						t.p[pi].x = t.p[pi].x * input_tsdf.voxel_size + input_tsdf.voxel_grid_origin[0];
						t.p[pi].y = t.p[pi].y * input_tsdf.voxel_size + input_tsdf.voxel_grid_origin[1];
						t.p[pi].z = t.p[pi].z * input_tsdf.voxel_size + input_tsdf.voxel_grid_origin[2];
						t.p[pi].r = 255;
						t.p[pi].g = 255;
						t.p[pi].b = 255;
						vertices.push_back(t.p[pi]);
					}
					else
						f.vertex_idx[pi] = vertices_idx[s];
				}
				faces.push_back(f);
			}
		}
		return;
	}

	void Shape::exportToPly(std::string output_file_name) {
		std::ofstream ply_file;
		ply_file.open(output_file_name);
		ply_file << "ply\nformat ascii 1.0\ncomment stanford bunny\nelement vertex ";
		ply_file << vertices.size() << "\n";
		ply_file << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
		ply_file << "element face " << faces.size() << "\n";
		ply_file << "property list int int vertex_index\nend_header\n";
		for (auto v : vertices) {
			ply_file << v.x << " " << v.y << " " << v.z << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << "\n";
		}
		for (auto f : faces) {
			ply_file << "3 " << f.vertex_idx[0] << " " << f.vertex_idx[1] << " " << f.vertex_idx[2] << "\n";
		}
		ply_file.close();
		cout << "File saved." << endl;
		return;
	}

	void Shape::exportToObj(std::string output_file_name) {
		
		return;
	}

	void Shape::exportToJt(std::string output_file_name) {

		return;
	}

	pcl::PointXYZRGB Shape::interpolateVertex(float isolevel, pcl::PointXYZRGB p1, pcl::PointXYZRGB p2, float val1, float val2) {
		float mu;
		pcl::PointXYZRGB p;

		if (fabs(isolevel - val1) < 0.00001)
			return p1;
		if (fabs(isolevel - val2) < 0.00001)
			return p2;
		if (fabs(val1 - val2) < 0.00001)
			return p1;
		mu = (isolevel - val1) / (val2 - val1);
		p.x = p1.x + mu * (p2.x - p1.x);
		p.y = p1.y + mu * (p2.y - p1.y);
		p.z = p1.z + mu * (p2.z - p1.z);

		return p;
	}

}

