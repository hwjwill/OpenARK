#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

#include "tsdf.cuh"

// CUDA kernel function to integrate a TSDF voxel volume given depth images
namespace ark
{
    __global__
    void Integrate(float * cam_K, float * cam2base, float * depth_im,
                   int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                   float * voxel_grid_TSDF, float * voxel_grid_weight) {

        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

            // Convert voxel center from grid coordinates to base frame camera coordinates
            float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
            float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
            float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

            // Convert from base frame camera coordinates to current frame camera coordinates
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
            tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
            tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
            float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

            if (pt_cam_z <= 0)
                continue;

            int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
            int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
            if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
                continue;

            float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

            if (depth_val <= 0 || depth_val > 6)
                continue;

            float diff = depth_val - pt_cam_z;

            if (diff <= -trunc_margin)
                continue;

            // Integrate
            int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
            float dist = fmin(1.0f, diff / trunc_margin);
            float weight_old = voxel_grid_weight[volume_idx];
            float weight_new = weight_old + 1.0f;
            voxel_grid_weight[volume_idx] = weight_new;
            voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
        }
    }
__host__
GpuTsdfGenerator::GpuTsdfGenerator(int width, int height, float fx, float fy, float cx, float cy,
                                   float v_g_o_x = -1.5f, float v_g_o_y = -1.5f, float v_g_o_z = 0.5f,
                                   float v_size = 0.006f, float trunc_m = 0.03f, int v_g_d_x = 500, int v_g_d_y = 500, int v_g_d_z = 500){
    im_width_ = width;
    im_height_ = height;

    memset(p_cam_K_, 0.0f, sizeof(float) * 3*3);
    p_cam_K_[0] = fx;
    p_cam_K_[2] = cx;
    p_cam_K_[4] = fy;
    p_cam_K_[5] = cy;
    p_cam_K_[8] = 1.0f;

    voxel_grid_origin_x_ = v_g_o_x;
    voxel_grid_origin_y_ = v_g_o_y;
    voxel_grid_origin_z_ = v_g_o_z;

    voxel_grid_dim_x_ = v_g_d_x;
    voxel_grid_dim_y_ = v_g_d_y;
    voxel_grid_dim_z_ = v_g_d_z;

    voxel_size_ = v_size;

    trunc_margin_ = trunc_m;

    std::cout << "fx: " <<fx<<std::endl;
    std::cout << "cx: " <<cx<<std::endl;
    std::cout << "fy: " <<fy<<std::endl;
    std::cout << "cy: " <<cy<<std::endl;

    // Initialize voxel grid
    p_voxel_grid_TSDF_ = new float[voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_];
    p_voxel_grid_weight_ = new float[voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_];
    memset(p_voxel_grid_TSDF_, 1.0f, sizeof(float) * voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_);
    memset(p_voxel_grid_weight_, 0.0f, sizeof(float) * voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_);

    // Load variables to GPU memory
    cudaMalloc(&p_gpu_voxel_grid_TSDF_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float));
    cudaMalloc(&p_gpu_voxel_grid_weight_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(p_gpu_voxel_grid_TSDF_,p_voxel_grid_TSDF_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu_voxel_grid_weight_, p_voxel_grid_weight_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMalloc(&p_gpu_cam_K_, 3 * 3 * sizeof(float));
    cudaMemcpy(p_gpu_cam_K_,p_cam_K_, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&p_gpu_cam2base_, 4 * 4 * sizeof(float));
    cudaMalloc(&p_gpu_depth_im_, im_height_ * im_width_ * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
}

__host__
void GpuTsdfGenerator::processFrame(float *depth_im, float *cam2base)
{
    cudaMemcpy(p_gpu_cam2base_, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu_depth_im_, depth_im, im_height_ * im_width_ * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    Integrate <<< voxel_grid_dim_z_, voxel_grid_dim_y_ >>>(p_gpu_cam_K_, p_gpu_cam2base_, p_gpu_depth_im_,
          im_height_, im_width_, voxel_grid_dim_x_, voxel_grid_dim_y_, voxel_grid_dim_z_,
          voxel_grid_origin_x_, voxel_grid_origin_y_, voxel_grid_origin_z_, voxel_size_, trunc_margin_,
            p_gpu_voxel_grid_TSDF_, p_gpu_voxel_grid_weight_);
    checkCUDA(__LINE__, cudaGetLastError());
}

__host__
void GpuTsdfGenerator::Shutdown() {

}

__host__
void GpuTsdfGenerator::hello()
{
    std::cout << "Hello World" <<std::endl;
}

__host__
void GpuTsdfGenerator::SaveTSDF(std::string filename) {
    // Load TSDF voxel grid from GPU to CPU memory
    cudaMemcpy(p_voxel_grid_TSDF_, p_gpu_voxel_grid_TSDF_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_voxel_grid_weight_, p_gpu_voxel_grid_weight_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDA(__LINE__, cudaGetLastError());
    // Save TSDF voxel grid and its parameters to disk as binary file (float array)
    std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
    std::string voxel_grid_saveto_path = filename;
    std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
    float voxel_grid_dim_xf = (float) voxel_grid_dim_x_;
    float voxel_grid_dim_yf = (float) voxel_grid_dim_y_;
    float voxel_grid_dim_zf = (float) voxel_grid_dim_z_;
    outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
    outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
    outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_x_, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_y_, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_z_, sizeof(float));
    outFile.write((char*)&voxel_size_, sizeof(float));
    outFile.write((char*)&trunc_margin_, sizeof(float));
    for (int i = 0; i < voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_; ++i)
        outFile.write((char*)&p_voxel_grid_TSDF_[i], sizeof(float));
    outFile.close();
}

__host__
void GpuTsdfGenerator::SavePLY(std::string filename) {
    cudaMemcpy(p_voxel_grid_TSDF_, p_gpu_voxel_grid_TSDF_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaMemcpy(p_voxel_grid_weight_, p_gpu_voxel_grid_weight_, voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_ * sizeof(float), cudaMemcpyDeviceToHost);

    checkCUDA(__LINE__, cudaGetLastError());
    tsdf2mesh(filename);
}

    __host__
    XYZ GpuTsdfGenerator::VertexInterp(float isolevel, XYZ p1, XYZ p2, float valp1, float valp2)
    {
        float mu;
        XYZ p;

        if (fabs(isolevel - valp1) < 0.00001)
            return p1;
        if (fabs(isolevel - valp2) < 0.00001)
            return p2;
        if (fabs(valp1 - valp2) < 0.00001)
            return p1;
        mu = (isolevel - valp1) / (valp2 - valp1);
        p.x = p1.x + mu * (p2.x - p1.x);
        p.y = p1.y + mu * (p2.y - p1.y);
        p.z = p1.z + mu * (p2.z - p1.z);

        return p;
    }

    __host__
    void GpuTsdfGenerator::tsdf2mesh(std::string outputFileName) {
        int totalSize = voxel_grid_dim_x_ * voxel_grid_dim_y_ * voxel_grid_dim_z_;
        std::vector<FACE> faces;
        std::vector<XYZ> vertices;
        std::unordered_map<std::string, int> verticesIdx;
        int vertexCount = 0;
        for (size_t i = 0; i < totalSize; ++i) {
            int xi = i / (voxel_grid_dim_y_ * voxel_grid_dim_z_);
            int yi = (i - xi * voxel_grid_dim_y_ * voxel_grid_dim_z_) / voxel_grid_dim_z_;
            int zi = i - xi * voxel_grid_dim_y_ * voxel_grid_dim_z_ - yi * voxel_grid_dim_z_;
            if (xi == voxel_grid_dim_x_ - 1 || yi == voxel_grid_dim_y_ - 1 || zi == voxel_grid_dim_z_ - 1)
                continue;
            GRIDCELL grid;
            grid.p[0] = XYZ(xi, yi, zi);
            grid.p[1] = XYZ(xi, yi + 1, zi);
            grid.p[2] = XYZ(xi + 1, yi + 1, zi);
            grid.p[3] = XYZ(xi + 1, yi, zi);
            grid.p[4] = XYZ(xi, yi, zi + 1);
            grid.p[5] = XYZ(xi, yi + 1, zi + 1);
            grid.p[6] = XYZ(xi + 1, yi + 1, zi + 1);
            grid.p[7] = XYZ(xi + 1, yi, zi + 1);

            grid.val[0] = p_voxel_grid_TSDF_[xi * voxel_grid_dim_y_ * voxel_grid_dim_z_ + yi * voxel_grid_dim_z_ + zi];
            grid.val[1] = p_voxel_grid_TSDF_[xi * voxel_grid_dim_y_ * voxel_grid_dim_z_ + (yi + 1) * voxel_grid_dim_z_ + zi];
            grid.val[2] = p_voxel_grid_TSDF_[(xi + 1) * voxel_grid_dim_y_ * voxel_grid_dim_z_ + (yi + 1) * voxel_grid_dim_z_ + zi];
            grid.val[3] = p_voxel_grid_TSDF_[(xi + 1) * voxel_grid_dim_y_ * voxel_grid_dim_z_ + yi * voxel_grid_dim_z_ + zi];
            grid.val[4] = p_voxel_grid_TSDF_[xi * voxel_grid_dim_y_ * voxel_grid_dim_z_ + yi * voxel_grid_dim_z_ + (zi + 1)];
            grid.val[5] = p_voxel_grid_TSDF_[xi * voxel_grid_dim_y_ * voxel_grid_dim_z_ + (yi + 1) * voxel_grid_dim_z_ + (zi + 1)];
            grid.val[6] = p_voxel_grid_TSDF_[(xi + 1) * voxel_grid_dim_y_ * voxel_grid_dim_z_ + (yi + 1) * voxel_grid_dim_z_ + (zi + 1)];
            grid.val[7] = p_voxel_grid_TSDF_[(xi + 1) * voxel_grid_dim_y_ * voxel_grid_dim_z_ + yi * voxel_grid_dim_z_ + (zi + 1)];
            int cubeIndex = 0;
            if (grid.val[0] < 0) cubeIndex |= 1;
            if (grid.val[1] < 0) cubeIndex |= 2;
            if (grid.val[2] < 0) cubeIndex |= 4;
            if (grid.val[3] < 0) cubeIndex |= 8;
            if (grid.val[4] < 0) cubeIndex |= 16;
            if (grid.val[5] < 0) cubeIndex |= 32;
            if (grid.val[6] < 0) cubeIndex |= 64;
            if (grid.val[7] < 0) cubeIndex |= 128;
            XYZ vertlist[12];
            if (edgeTable[cubeIndex] == 0)
                continue;

            /* Find the vertices where the surface intersects the cube */
            if (edgeTable[cubeIndex] & 1)
                vertlist[0] =
                        VertexInterp(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (edgeTable[cubeIndex] & 2)
                vertlist[1] =
                        VertexInterp(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (edgeTable[cubeIndex] & 4)
                vertlist[2] =
                        VertexInterp(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (edgeTable[cubeIndex] & 8)
                vertlist[3] =
                        VertexInterp(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (edgeTable[cubeIndex] & 16)
                vertlist[4] =
                        VertexInterp(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (edgeTable[cubeIndex] & 32)
                vertlist[5] =
                        VertexInterp(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (edgeTable[cubeIndex] & 64)
                vertlist[6] =
                        VertexInterp(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (edgeTable[cubeIndex] & 128)
                vertlist[7] =
                        VertexInterp(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (edgeTable[cubeIndex] & 256)
                vertlist[8] =
                        VertexInterp(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (edgeTable[cubeIndex] & 512)
                vertlist[9] =
                        VertexInterp(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (edgeTable[cubeIndex] & 1024)
                vertlist[10] =
                        VertexInterp(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (edgeTable[cubeIndex] & 2048)
                vertlist[11] =
                        VertexInterp(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

            /* Create the triangle */
            for (int ti = 0; triTable[cubeIndex][ti] != -1; ti += 3) {
                FACE f;
                TRIANGLE t;
                t.p[0] = vertlist[triTable[cubeIndex][ti]];
                t.p[1] = vertlist[triTable[cubeIndex][ti + 1]];
                t.p[2] = vertlist[triTable[cubeIndex][ti + 2]];
                for (int pi = 0; pi < 3; ++pi) {
                    std::string s = "x" + std::to_string(t.p[pi].x) + "y" + std::to_string(t.p[pi].y) + "z" + std::to_string(t.p[pi].z);
                    if (verticesIdx.find(s) == verticesIdx.end()) {
                        verticesIdx.insert(std::make_pair(s, vertexCount));
                        f.vIdx[pi] = vertexCount++;
                        t.p[pi].x = t.p[pi].x * voxel_size_ + voxel_grid_origin_x_;
                        t.p[pi].y = t.p[pi].y * voxel_size_ + voxel_grid_origin_y_;
                        t.p[pi].z = t.p[pi].z * voxel_size_ + voxel_grid_origin_z_;
                        vertices.push_back(t.p[pi]);
                    }
                    else
                        f.vIdx[pi] = verticesIdx[s];
                }
                faces.push_back(f);
            }
        }
        std::cout << vertexCount << std::endl;
        std::ofstream plyFile;
        plyFile.open(outputFileName);
        plyFile << "ply\nformat ascii 1.0\ncomment stanford bunny\nelement vertex ";
        plyFile << vertices.size() << "\n";
        plyFile << "property float x\nproperty float y\nproperty float z\n";// property uchar red\nproperty uchar green\nproperty uchar blue\n";
        plyFile << "element face " << faces.size() << "\n";
        plyFile << "property list int int vertex_index\nend_header\n";
        for (auto v : vertices) {
            plyFile << v.x << " " << v.y << " " << v.z << /*" " << (int)c.r << " " << (int)c.g << " " << (int)c.b <<*/ "\n";
        }
        for (auto f : faces) {
            plyFile << "3 " << f.vIdx[0] << " " << f.vIdx[1] << " " << f.vIdx[2] << "\n";
        }
        plyFile.close();
        std::cout << "File saved" << std::endl;
    }
}


