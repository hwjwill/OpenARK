#include "Shape.h"
#include <unordered_map>
#include <string>

namespace ark {
	Shape::Shape() {

	}

	Shape::~Shape() {

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
		ply_file << "property float x\nproperty float y\nproperty float z\n";// property uchar red\nproperty uchar green\nproperty uchar blue\n";
		ply_file << "element face " << faces.size() << "\n";
		ply_file << "property list int int vertex_index\nend_header\n";
		for (auto v : vertices) {
			ply_file << v.x << " " << v.y << " " << v.z << /*" " << (int)c.r << " " << (int)c.g << " " << (int)c.b <<*/ "\n";
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

