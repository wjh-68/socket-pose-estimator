#include "StaticPoseOptimizer.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

namespace fs = std::filesystem;

static bool loadCameraIntrinsics(const std::string &path, cv::Mat &K, cv::Mat &dist)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }
    fs["k"] >> K;
    fs["d"] >> dist;
    return !K.empty() && !dist.empty();
}

static bool loadExtrinsics(const std::string &path, cv::Mat &eMc)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }
    cv::Mat R, t;
    fs["c2g_r"] >> R;
    fs["c2g_t"] >> t;
    if (R.empty() || t.empty())
    {
        return false;
    }
    eMc = cv::Mat::eye(4, 4, CV_64F);
    R.convertTo(eMc(cv::Rect(0, 0, 3, 3)), CV_64F);
    t.convertTo(eMc(cv::Rect(3, 0, 1, 3)), CV_64F);
    return true;
}

static bool loadPointsTxt(const std::string &path, std::vector<cv::Point2d> &points)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }
    points.clear();
    double x, y;
    std::vector<double> values;
    while (in >> x >> y)
    {
        values.push_back(x);
        values.push_back(y);
    }
    if (values.empty() || values.size() % 2 != 0)
    {
        return false;
    }
    for (size_t i = 0; i + 1 < values.size(); i += 2)
    {
        points.emplace_back(values[i], values[i + 1]);
    }
    return true;
}

static bool loadNpyMatrix(const std::string &path, cv::Mat &mat)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
    {
        return false;
    }
    char version[2];
    in.read(version, 2);
    uint16_t headerLen;
    if (version[0] == 1)
    {
        in.read(reinterpret_cast<char *>(&headerLen), 2);
    }
    else if (version[0] == 2)
    {
        uint32_t len32;
        in.read(reinterpret_cast<char *>(&len32), 4);
        headerLen = static_cast<uint16_t>(len32);
    }
    else
    {
        return false;
    }
    std::string header(headerLen, '\0');
    in.read(header.data(), headerLen);
    auto findToken = [&](const std::string &key) -> std::string
    {
        auto pos = header.find(key);
        if (pos == std::string::npos)
            return {};
        auto colon = header.find(':', pos + key.size());
        if (colon == std::string::npos)
            return {};
        auto quoteStart = header.find('"', colon);
        auto quoteSingleStart = header.find('\'', colon);
        size_t start = std::string::npos;
        char quoteChar = '\0';
        if (quoteSingleStart != std::string::npos && (quoteStart == std::string::npos || quoteSingleStart < quoteStart))
        {
            start = quoteSingleStart + 1;
            quoteChar = '\'';
        }
        else if (quoteStart != std::string::npos)
        {
            start = quoteStart + 1;
            quoteChar = '"';
        }
        if (start == std::string::npos)
            return {};
        auto end = header.find(quoteChar, start);
        if (end == std::string::npos)
            return {};
        return header.substr(start, end - start);
    };
    std::string descr = findToken("descr");
    if (descr.empty() || descr != "<f8")
    {
        return false;
    }
    bool fortranOrder = header.find("fortran_order': True") != std::string::npos ||
                        header.find("\"fortran_order\": True") != std::string::npos;
    if (fortranOrder)
    {
        return false;
    }
    auto shapePos = header.find("shape");
    if (shapePos == std::string::npos)
    {
        return false;
    }
    auto left = header.find('(', shapePos);
    auto right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos)
    {
        return false;
    }
    std::string shapeStr = header.substr(left + 1, right - left - 1);
    std::vector<int> shape;
    std::stringstream ss(shapeStr);
    int value;
    while (ss >> value)
    {
        shape.push_back(value);
        if (ss.peek() == ',')
            ss.ignore();
    }
    if (shape.empty())
    {
        return false;
    }
    int count = 1;
    for (int s : shape)
    {
        count *= s;
    }
    std::vector<double> buffer(count);
    in.read(reinterpret_cast<char *>(buffer.data()), sizeof(double) * count);
    if (!in)
    {
        return false;
    }
    if (shape.size() == 2)
    {
        mat = cv::Mat(shape[0], shape[1], CV_64F);
        std::memcpy(mat.data, buffer.data(), sizeof(double) * count);
    }
    else if (shape.size() == 1)
    {
        mat = cv::Mat(1, shape[0], CV_64F);
        std::memcpy(mat.data, buffer.data(), sizeof(double) * count);
    }
    else
    {
        return false;
    }
    return true;
}

static std::vector<int> sortedDataIds(const std::string &path)
{
    std::vector<int> ids;
    for (auto &entry : fs::directory_iterator(path))
    {
        if (!entry.is_regular_file())
            continue;
        auto name = entry.path().filename().string();
        if (entry.path().extension() == ".txt")
        {
            try
            {
                ids.push_back(std::stoi(entry.path().stem().string()));
            }
            catch (...)
            {
            }
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static bool loadDataset(const std::string &datasetPath,
                        std::vector<int> &frameIds,
                        std::vector<cv::Mat> &robotPoses,
                        std::vector<std::vector<cv::Point2d>> &framePoints)
{
    std::string dataPath = datasetPath + "/data";
    if (!fs::exists(dataPath))
    {
        std::cerr << "Dataset path not found: " << dataPath << std::endl;
        return false;
    }
    frameIds = sortedDataIds(dataPath);
    if (frameIds.empty())
    {
        std::cerr << "No txt frame files found in " << dataPath << std::endl;
        return false;
    }
    for (int id : frameIds)
    {
        std::string txtPath = dataPath + "/" + std::to_string(id) + ".txt";
        std::string npyPath = dataPath + "/" + std::to_string(id) + ".npy";
        if (!fs::exists(txtPath) || !fs::exists(npyPath))
        {
            std::cerr << "Missing pair for id " << id << std::endl;
            return false;
        }
        std::vector<cv::Point2d> pts2d;
        if (!loadPointsTxt(txtPath, pts2d))
        {
            std::cerr << "Failed to load points from " << txtPath << std::endl;
            return false;
        }
        cv::Mat robotPose;
        if (!loadNpyMatrix(npyPath, robotPose))
        {
            std::cerr << "Failed to load robot pose from " << npyPath << std::endl;
            return false;
        }
        if (robotPose.rows != 4 || robotPose.cols != 4)
        {
            std::cerr << "Robot pose matrix shape invalid: " << npyPath << std::endl;
            return false;
        }
        framePoints.push_back(pts2d);
        robotPoses.push_back(robotPose);
    }
    return true;
}

static void printMatrix(const cv::Mat &mat)
{
    for (int r = 0; r < mat.rows; ++r)
    {
        for (int c = 0; c < mat.cols; ++c)
        {
            std::cout << mat.at<double>(r, c) << (c + 1 == mat.cols ? "" : " ");
        }
        std::cout << std::endl;
    }
}

void runRandomExample()
{
    std::cout << "=== Random test example ===" << std::endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 1015.4, 0.0, 638.5,
                 0.0, 1015.4, 386.8,
                 0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<double>(1, 5) << 0.1, -0.2, 0.0, 0.0, 0.07);
    StaticPoseOptimizer optimizer(K, dist);

    std::vector<cv::Point3d> objPts = {
        {-8.0, 11.2, 0.0}, {8.0, 11.2, 0.0}, {-16.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {16.0, 0.0, 0.0}, {-8.0, -13.9, 0.0}, {8.0, -13.9, 0.0}};
    optimizer.setObjectPoints(objPts);
    optimizer.setExtrinsics(cv::Mat::eye(4, 4, CV_64F));

    cv::Mat bMoInit = cv::Mat::eye(4, 4, CV_64F);
    bMoInit.at<double>(0, 3) = 348.0;
    bMoInit.at<double>(1, 3) = -1084.0;
    bMoInit.at<double>(2, 3) = 498.0;
    optimizer.setInitialPose(bMoInit);

    for (int i = 0; i < 5; ++i)
    {
        cv::Mat robotPose = cv::Mat::eye(4, 4, CV_64F);
        robotPose.at<double>(0, 3) = i * 10.0;
        robotPose.at<double>(1, 3) = i * 5.0;
        robotPose.at<double>(2, 3) = 100.0 + i * 50.0;

        std::vector<cv::Point2d> pts2d(7);
        for (int j = 0; j < 7; ++j)
        {
            pts2d[j] = cv::Point2d(400.0 + std::rand() % 200, 300.0 + std::rand() % 200);
        }
        optimizer.addFrame(i, robotPose, pts2d);
    }

    bool ok = optimizer.optimize();
    std::cout << "Random example optimization " << (ok ? "succeeded" : "failed") << std::endl;
    std::cout << "Optimized pose:" << std::endl;
    printMatrix(optimizer.getPose());
    std::cout << "Average error: " << optimizer.getAverageError() << std::endl;
}

void runDatasetExample(const std::string &datasetPath, int maxFrames)
{
    std::cout << "=== Dataset example ===" << std::endl;
    std::vector<int> frameIds;
    std::vector<cv::Mat> robotPoses;
    std::vector<std::vector<cv::Point2d>> framePoints;

    if (!loadDataset(datasetPath, frameIds, robotPoses, framePoints))
    {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return;
    }

    cv::Mat K, dist, eMc;
    std::string camIntrinsicPath = datasetPath + "/camPrms/cam_intrisic.xml";
    std::string extrinsicsPath = datasetPath + "/camPrms/cam_2_gripper.xml";
    if (!loadCameraIntrinsics(camIntrinsicPath, K, dist))
    {
        std::cerr << "Failed to load camera intrinsics from " << camIntrinsicPath << std::endl;
        return;
    }
    if (!loadExtrinsics(extrinsicsPath, eMc))
    {
        std::cerr << "Failed to load extrinsics from " << extrinsicsPath << std::endl;
        return;
    }

    StaticPoseOptimizer optimizer(K, dist);
    std::vector<cv::Point3d> objPts = {
        {-8.0, 11.2, 0.0}, {8.0, 11.2, 0.0}, {-16.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {16.0, 0.0, 0.0}, {-8.0, -13.9, 0.0}, {8.0, -13.9, 0.0}};
    optimizer.setObjectPoints(objPts);
    optimizer.setExtrinsics(eMc);

    int framesToUse = std::min(maxFrames, static_cast<int>(frameIds.size()));
    bool initialPoseSet = false;
    for (int idx = 0; idx < framesToUse; ++idx)
    {
        cv::Mat rvec, tvec;
        bool solved = cv::solvePnP(objPts, framePoints[idx], K, dist, rvec, tvec, false, cv::SOLVEPNP_IPPE);
        if (!solved)
        {
            std::cerr << "SolvePnP failed for frame " << frameIds[idx] << std::endl;
            continue;
        }
        cv::Mat cMo = StaticPoseOptimizer::poseFromRodrigues(rvec, tvec);
        cv::Mat bMoInit = robotPoses[idx] * eMc * cMo;
        if (!initialPoseSet)
        {
            optimizer.setInitialPose(bMoInit);
            initialPoseSet = true;
        }
        optimizer.addFrame(frameIds[idx], robotPoses[idx], framePoints[idx]);
    }

    if (!initialPoseSet)
    {
        std::cerr << "No valid initial pose computed." << std::endl;
        return;
    }

    bool ok = optimizer.optimize();
    std::cout << "Dataset example optimization " << (ok ? "succeeded" : "failed") << std::endl;
    std::cout << "Optimized pose:" << std::endl;
    printMatrix(optimizer.getPose());
    std::cout << "Average error: " << optimizer.getAverageError() << std::endl;
}

int main(int argc, char *argv[])
{
    std::string datasetPath = "/home/byd/work/socket-pose-estimator/dataset/save_data3/chb_20260511_120244";
    int maxFrames = 32;
    bool runDataset = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg.rfind("--dataset_path=", 0) == 0)
        {
            datasetPath = arg.substr(std::string("--dataset_path=").size());
            runDataset = true;
        }
        else if (arg.rfind("--max_frames=", 0) == 0)
        {
            maxFrames = std::stoi(arg.substr(std::string("--max_frames=").size()));
        }
        else if (arg == "--run_dataset")
        {
            runDataset = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: static_pose_optimizer_example [--run_dataset] [--dataset_path=PATH] [--max_frames=N]" << std::endl;
            return 0;
        }
    }

    runRandomExample();
    if (runDataset)
    {
        runDatasetExample(datasetPath, maxFrames);
    }
    else
    {
        std::cout << "Dataset example skipped. Use --run_dataset or --dataset_path=PATH to enable." << std::endl;
    }
    return 0;
}
