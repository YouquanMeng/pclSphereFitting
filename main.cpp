#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>

// 定义常用的点云类型（以XYZ为例，可替换为PointXYZRGB等）
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct ErrorMetrics {
    float RMS_error = 0.0f;         // 均方根误差
    float StdDev = 0.0f;            // 误差方差
    float mean_error = 0.0f;        // 平均误差
    float max_error = 0.0f;         // 最大误差
    float Inlier_ratio = 0.0f;      // 内点比例
    float in_sphere_ratio = 0.0f;   // 球内部点比例
} ;

static bool load_xyz_file(const std::string &file_path, std::vector<std::vector<float>> &cloud) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        return false;
    }

    cloud.clear();
    float x, y, z;
    while (in >> x >> y >> z) { 
        cloud.push_back({x, y, z});
    }

    return !cloud.empty();
}

/**
 * @brief 点云绕Y轴旋转
 * @param input_cloud 输入点云（原始点云）
 * @param output_cloud 输出点云（旋转后的点云）
 * @param angle_deg 旋转角度（单位：度），顺时针为正（右手坐标系）
 */
void rotatePointCloudAroundYAxis(const PointCloudT::Ptr& input_cloud,
                                 PointCloudT::Ptr& output_cloud,
                                 float angle_deg)
{
    // 1. 将角度从度转换为弧度（PCL的三角函数使用弧度）
    // float angle_rad = pcl::deg2rad(angle_deg);
    float angle_rad = angle_deg * M_PI / 180.0f;

    // 2. 构造绕Y轴的旋转矩阵
    // 右手坐标系下绕Y轴旋转的矩阵公式：
    // [ cosθ   0   sinθ ]
    // [   0    1    0   ]
    // [-sinθ   0   cosθ ]
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitY()));

    // 3. 执行点云变换（应用旋转矩阵）
    pcl::transformPointCloud(*input_cloud, *output_cloud, transform);
}

/**
 * @brief 点云计算连通域（欧式聚类）
 * @param input_cloud 输入点云
 * @param cluster_indices 输出：每个连通域的点索引
 * @param cluster_tolerance 连通域阈值（两点距离≤此值则连通，单位：米）
 * @param min_size 最小连通域点数（过滤噪声）
 * @param max_size 最大连通域点数（过滤过大区域）
 */
void computeConnectedRegions(
    const PointCloudT::Ptr& input_cloud,
    std::vector<pcl::PointIndices>& cluster_indices,
    float cluster_tolerance = 5.0f,  // 2cm邻域阈值
    int min_size = 10,                 // 最小10个点为一个连通域
    int max_size = 300000)             // 最大10万个点
{
    // 1. 构建KdTree（加速邻域搜索）
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(input_cloud);

    // 2. 欧式聚类提取连通域
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance);  // 连通域距离阈值
    ec.setMinClusterSize(min_size);             // 最小连通域尺寸
    ec.setMaxClusterSize(max_size);             // 最大连通域尺寸
    ec.setSearchMethod(tree);                   // 邻域搜索方式
    ec.setInputCloud(input_cloud);              // 输入点云
    ec.extract(cluster_indices);                // 提取连通域索引
}

/**
 * @brief 根据连通域索引提取对应点云
 * @param input_cloud 输入点云
 * @param cluster_indices 连通域索引
 * @return 每个连通域对应的点云列表
 */
std::vector<PointCloudT::Ptr> extractConnectedClouds(
    const PointCloudT::Ptr& input_cloud,
    const std::vector<pcl::PointIndices>& cluster_indices)
{
    std::vector<PointCloudT::Ptr> connected_clouds;
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input_cloud);
    extract.setNegative(false);  // 提取索引对应的点

    for (size_t i = 0; i < cluster_indices.size(); ++i)
    {
        // 创建当前连通域的索引对象
        pcl::PointIndices::Ptr idx(new pcl::PointIndices);
        idx->indices = cluster_indices[i].indices;

        // 提取当前连通域的点云
        PointCloudT::Ptr cloud(new PointCloudT);
        extract.setIndices(idx);
        extract.filter(*cloud);

        connected_clouds.push_back(cloud);
        std::cout << "连通域 " << i << " 包含点数：" << cloud->size() << std::endl;
    }
    return connected_clouds;
}


/**
 * @brief 统计滤波去噪（推荐）
 * @param input_cloud 输入含噪声点云
 * @param output_cloud 输出去噪后点云
 * @param mean_k 邻域点数（计算均值/标准差的参考点数）
 * @param std_mul 标准差倍数（越大过滤越宽松，越小越严格）
 */
void removeNoiseByStatistical(const PointCloudT::Ptr& input_cloud,
                              PointCloudT::Ptr& output_cloud,
                              int mean_k = 50,          // 邻域参考点数（默认50）
                              float std_mul = 1.0f)     // 标准差倍数（默认1.0）
{
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(input_cloud);        // 设置输入点云
    sor.setMeanK(mean_k);                  // 设置邻域参考点数
    sor.setStddevMulThresh(std_mul);       // 标准差倍数阈值
    // sor.setNegative(true); // 若开启，会输出噪声点（用于调试）
    sor.filter(*output_cloud);             // 执行滤波
}

/**
 * @brief 最小二乘球拟合（无噪声点云专用）
 * @param input_cloud 输入点云
 * @param sphere_center 球心
 * @param sphere_radius 半径
 * @param err_rms 均方根误差（拟合质量指标）
 * @param error_variance 误差方差（拟合质量指标）
 * @return 是否拟合成功
 */
bool fitSphereLeastSquares(
    const PointCloudT::Ptr& input_cloud,
    Eigen::Vector3f& sphere_center,
    float& sphere_radius,  ErrorMetrics &error_metrics
)
{
    if (input_cloud->size() < 4)  // 拟合球面至少需要4个点
    {
        return false;
    }

    // 构建最小二乘矩阵
    Eigen::MatrixXf A(input_cloud->size(), 4);
    Eigen::VectorXf B(input_cloud->size());
    for (int i = 0; i < input_cloud->size(); ++i)
    {
        const auto& p = input_cloud->points[i];
        A(i, 0) = 2 * p.x;
        A(i, 1) = 2 * p.y;
        A(i, 2) = 2 * p.z;
        A(i, 3) = 1;
        B(i) = p.x*p.x + p.y*p.y + p.z*p.z;
    }

    // 求解线性方程组 Ax = B
    Eigen::VectorXf X = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    sphere_center << X(0), X(1), X(2);
    sphere_radius = sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2) + X(3));

    // 计算拟合误差（均方根误差）
    error_metrics.RMS_error = 0.0f;
    error_metrics.StdDev = 0.0f;
    std::vector<float> errors;
    errors.clear();

    for (int i = 0; i < input_cloud->size(); ++i)
    {
        const auto& p = input_cloud->points[i];
        float dist = sqrt((p.x - sphere_center.x())*(p.x - sphere_center.x()) +
                         (p.y - sphere_center.y())*(p.y - sphere_center.y()) +
                         (p.z - sphere_center.z())*(p.z - sphere_center.z()));
        float error = dist - sphere_radius;
        errors.push_back(error);
        error_metrics.RMS_error += error*error;
    }
    error_metrics.RMS_error = sqrt(error_metrics.RMS_error / input_cloud->size());
    // 计算误差的方差
    float mean_error = std::accumulate(errors.begin(), errors.end(), 0.0f) / errors.size();
    for (const auto& error : errors) {
        error_metrics.StdDev += (error - mean_error) * (error - mean_error);
    }
    error_metrics.StdDev = sqrt(error_metrics.StdDev / errors.size());
    // 计算平均误差
    error_metrics.mean_error = mean_error;
    // 计算最大误差
    error_metrics.max_error = *std::max_element(errors.begin(), errors.end());
    float min_error = *std::min_element(errors.begin(), errors.end());
    error_metrics.max_error=std::max(std::abs(error_metrics.max_error), std::abs(min_error));

    // 计算内点比例
    int inlier_count = std::count_if(errors.begin(), errors.end(), [&](float error) {
        return std::abs(error) <= error_metrics.RMS_error; // 以RMS误差作为内点阈值
    });
    error_metrics.Inlier_ratio = static_cast<float>(inlier_count) / input_cloud->size();

    // 计算球内部点比例
    int in_sphere_count = std::count_if(errors.begin(), errors.end(), [&](float error) {
        return error < 0; // 球内部点的误差为负
    });
    error_metrics.in_sphere_ratio = static_cast<float>(in_sphere_count) / input_cloud->size();
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd> [cluster_tolerance] [min_size] [max_size]" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    double cluster_tolerance = 0.02;
    int min_size = 100;
    int max_size = 500000;
    
    if (argc >= 3) cluster_tolerance = atof(argv[2]);
    if (argc >= 4) min_size = atoi(argv[3]);
    if (argc >= 5) max_size = atoi(argv[4]);

    pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>);
    std::vector<std::vector<float>> cloud_points;
    if (load_xyz_file(filename, cloud_points)) {
        std::cout << "Loaded " << cloud_points.size() << " points from " << filename << std::endl;
        for (const auto &pt : cloud_points) {
            if (pt.size() >= 3) {
                PointT p;
                p.x = pt[0];
                p.y = pt[1];
                p.z = pt[2];
                input_cloud->push_back(p);
            }
        }
        input_cloud->width = static_cast<std::uint32_t>(input_cloud->size());
        input_cloud->height = 1;
        input_cloud->is_dense = true;
    } else {
        std::cerr << "Failed to load point cloud from " << filename << std::endl;
        return -1;
    }

    std::cout << "原始点云点数：" << input_cloud->size() << std::endl;

    // 方式2：手动创建测试点云（无文件时用这个测试）
    // PointCloudT::Ptr input_cloud(new PointCloudT);
    // input_cloud->push_back(PointT(1.0, 0.0, 0.0));  // 点(1,0,0)绕Y轴转90度后应为(0,0,-1)
    // input_cloud->push_back(PointT(0.0, 0.0, 1.0));  // 点(0,0,1)绕Y轴转90度后应为(1,0,0)

    // ===================== 2. 执行绕Y轴旋转 =====================
    PointCloudT::Ptr rotated_cloud(new PointCloudT);
    float rotate_angle = -34.5f;  // 旋转角度：90度（可根据需求修改）
    rotatePointCloudAroundYAxis(input_cloud, rotated_cloud, rotate_angle);

    // ===================== 3. 保存旋转后的点云 =====================
    pcl::io::savePCDFileASCII("../data/result/rotated_cloud01.pcd", *rotated_cloud);
    std::cout << "旋转后的点云已保存，点数：" << rotated_cloud->size() << std::endl;

    // 打印测试点验证（可选）
    if (rotated_cloud->size() > 0)
    {
        std::cout << "第一个点旋转前：(" << input_cloud->points[0].x << ", " 
                  << input_cloud->points[0].y << ", " << input_cloud->points[0].z << ")" << std::endl;
        std::cout << "第一个点旋转后：(" << rotated_cloud->points[0].x << ", " 
                  << rotated_cloud->points[0].y << ", " << rotated_cloud->points[0].z << ")" << std::endl;
    }

    // 创建直通滤波器对象
    PointCloudT::Ptr filtered_cloud(new PointCloudT);
    pcl::PassThrough<PointT> pass;
    float z_min = 6.0f; // 设置Z轴的最小值（根据实际情况调整） 10.0f
    pass.setInputCloud(rotated_cloud);       // 设置输入点云
    pass.setFilterFieldName("z");          // 指定过滤的字段（Z轴）
    pass.setFilterLimits(z_min, FLT_MAX);  // 保留 Z ≥ z_min 的点（FLT_MAX表示无上限）
    // pass.setFilterLimitsNegative(true); // 若打开此注释，会过滤 Z ≥ z_min 的点（保留 Z < z_min）
    pass.filter(*filtered_cloud);            // 执行过滤，输出结果
    
    // 3. 保存过滤后的点云（PCD/PLY格式任选）
    pcl::io::savePCDFileASCII("../data/result/filtered_cloud01.pcd", *filtered_cloud);  // 保存为PCD
    pcl::io::savePLYFileBinary("../data/result/filtered_cloud01.ply", *filtered_cloud); // 保存为PLY（MeshLab可用）
    std::cout << "过滤后的点云已保存，点数：" << filtered_cloud->size() << std::endl;


    // 2. 计算连通域
    std::vector<pcl::PointIndices> cluster_indices;
    // computeConnectedRegions(filtered_cloud, cluster_indices, 0.5f, 1000, 300000);
    computeConnectedRegions(filtered_cloud, cluster_indices, 0.5f, filtered_cloud->size()/5, filtered_cloud->size());
    std::cout << "提取到的连通域数量：" << cluster_indices.size() << std::endl;

    // 3. 提取每个连通域的点云
    std::vector<PointCloudT::Ptr> connected_clouds = extractConnectedClouds(filtered_cloud, cluster_indices);
    // std::vector<PointCloudT::Ptr> connected_clouds = extractConnectedClouds(input_cloud, cluster_indices);



    // 4. 保存每个连通域（可选）
    for (size_t i = 0; i < connected_clouds.size(); ++i)
    {
        std::string filename = "connected_region_" + std::to_string(i) + ".pcd";
        std::string filename_ply = "connected_region_" + std::to_string(i) + ".ply";
        pcl::io::savePCDFileASCII("../data/result/"+filename, *connected_clouds[i]);
        pcl::io::savePLYFileBinary("../data/result/"+filename_ply, *connected_clouds[i]);
        std::cout << "连通域 " << i << " 已保存为：" << filename << std::endl;
    }

    // 2. 统计滤波去噪
    std::vector<PointCloudT::Ptr> denoised_clouds;
    for (size_t i = 0; i < connected_clouds.size(); ++i)
    {
        PointCloudT::Ptr denoised_cloud(new PointCloudT);
        removeNoiseByStatistical(connected_clouds[i], denoised_cloud, 50, 1.0f);
        denoised_clouds.push_back(denoised_cloud);
        std::cout << "连通域 " << i << " 统计滤波后点数：" << denoised_cloud->size() << std::endl;

        // 3. 保存去噪后的点云
        std::string filename = "denoised_connected_region_" + std::to_string(i) + ".pcd";
        std::string filename_ply = "denoised_connected_region_" + std::to_string(i) + ".ply";
        // pcl::io::savePCDFileASCII("../data/result/"+filename, *denoised_cloud);
        pcl::io::savePLYFileBinary("../data/result/"+filename_ply, *denoised_cloud);
        std::cout << "去噪后点云保存为 " << filename_ply << std::endl;
    }
    
    // PointCloudT::Ptr denoised_cloud(new PointCloudT);
    // removeNoiseByStatistical(connected_clouds[0], denoised_cloud, 50, 1.0f);
    // std::cout << "统计滤波后点数：" << denoised_cloud->size() << std::endl;

    // ===================== 球拟合 =====================
    std::vector<Eigen::Vector3f> sphere_centers;
    std::vector<float> sphere_radii;
    for (size_t i = 0; i < denoised_clouds.size(); ++i)
    {
        Eigen::Vector3f center;
        float radius(0.0f);
        ErrorMetrics error_metrics;
        if (fitSphereLeastSquares(denoised_clouds[i], center, radius, error_metrics))
        {
            sphere_centers.push_back(center);
            sphere_radii.push_back(radius);
            std::cout << "连通域 " << i << " 拟合球心: (" << center.x() << ", " << center.y() << ", " << center.z() << ")" 
                      << " 半径: " << radius 
                      << " 均方根误差: " << error_metrics.RMS_error << " 误差方差: " << error_metrics.StdDev 
                      << " 平均误差: " << error_metrics.mean_error << " 最大误差: " << error_metrics.max_error 
                      << " 内点比例: " << error_metrics.Inlier_ratio << " 球内部点比例: " << error_metrics.in_sphere_ratio << std::endl;
        }
        else
        {
            std::cout << "连通域 " << i << " 拟合失败（点数不足）" << std::endl;
        }
    }

    // 计算球心距
    if (sphere_centers.size() >= 2)
    {
        for (size_t i = 0; i < sphere_centers.size(); ++i)
            {
                for (size_t j = i + 1; j < sphere_centers.size(); ++j)
                {
                    float distance = (sphere_centers[i] - sphere_centers[j]).norm();
                    std::cout << "球心 " << i << " 和球心 " << j << " 的距离: " << distance << std::endl;
                }
            }
    }
    
    // 生成球面网格点（θ: 0~π, φ: 0~2π）
    for (size_t i = 0; i < sphere_centers.size(); ++i)
    {
        float cx = sphere_centers[i].x();
        float cy = sphere_centers[i].y();
        float cz = sphere_centers[i].z();
        float R = sphere_radii[i];
        PointCloudT::Ptr ideal_sphere(new PointCloudT);
        for (float theta = 0; theta <= M_PI; theta += 0.005) 
        {
            for (float phi = 0; phi <= 2*M_PI; phi += 0.02) 
            {
                PointT p;
                p.x = cx + R * sin(theta) * cos(phi);
                p.y = cy + R * sin(theta) * sin(phi);
                p.z = cz + R * cos(theta);
                ideal_sphere->push_back(p);
            }
        }
        ideal_sphere->width = static_cast<std::uint32_t>(ideal_sphere->size());
        ideal_sphere->height = 1;
        ideal_sphere->is_dense = true;
        std::string filename = "ideal_sphere_" + std::to_string(i) + ".pcd";
        std::string filename_ply = "ideal_sphere_" + std::to_string(i) + ".ply";
        pcl::io::savePCDFileASCII("../data/result/"+filename, *ideal_sphere);
        // pcl::io::savePLYFileBinary("../data/result/"+filename_ply, *ideal_sphere);
        std::cout << "理想球面点云已保存为 " << filename << std::endl;
    }

    // 显示拟合结果（可选）
    pcl::visualization::PCLVisualizer viewer("Sphere Fitting");

    viewer.setBackgroundColor(0, 0, 0); // 背景设为黑色

    // 3. 【核心】设置点云显示为单一颜色（红色，RGB范围0~1）
    
    for (size_t i = 0; i < denoised_clouds.size(); ++i)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> model_color(denoised_clouds[i], 0, 0, 255);
        viewer.addPointCloud<PointT>(denoised_clouds[i], "denoised_cloud_" + std::to_string(i));
    }

    // // PointCloudColorHandlerCustom：自定义单一颜色，无需点云带RGB字段
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> model_color(rotated_cloud, 0, 0, 255);
    // viewer.addPointCloud<PointT>(rotated_cloud, "input_cloud");
    for (size_t i = 0; i < denoised_clouds.size(); ++i)
    {
        // 2. 第一步：添加红色球体（先设置颜色，再调透明度）
        // 红色：R=1.0, G=0.0, B=0.0（addSphere颜色值范围0~1）
        viewer.addSphere(pcl::PointXYZ(sphere_centers[i].x(), sphere_centers[i].y(), sphere_centers[i].z()), sphere_radii[i], 1.0, 0.0, 0.0, "sphere_transparent_" + std::to_string(i));
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, "sphere_transparent_" + std::to_string(i));
    }
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }        

    return 0;
}
