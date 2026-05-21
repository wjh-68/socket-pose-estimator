#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct FrameData
{
    int frameIndex;
    cv::Mat robotPose; // 4x4
    std::vector<cv::Point2d> pts2d;
    std::vector<cv::Point3d> pts3d;
    std::vector<double> perPointErrorsPNP;
};

class StaticPoseOptimizer
{
public:
    StaticPoseOptimizer(const cv::Mat &K,
                        const cv::Mat &dist,
                        const std::vector<double> &priorSigma = {},
                        const std::vector<double> &pointSigmas = {});

    void setExtrinsics(const cv::Mat &eMc);
    void setObjectPoints(const std::vector<cv::Point3d> &objPts);
    void setPointSigmas(const std::vector<double> &sigmas);
    void setInitialPose(const cv::Mat &pose);
    void addFrame(int frameIndex,
                  const cv::Mat &robotPose,
                  const std::vector<cv::Point2d> &pts2d,
                  const std::vector<cv::Point3d> &pts3d = {},
                  const std::vector<double> &perPointErrorsPNP = {});
    void removeOldestFrame();

    bool optimize(int maxIters = 100, double lambdaInit = 1e-3);

    cv::Mat getPose() const;
    double getAverageError() const;
    double getFrameError(int frameIndex) const;
    int getFrameCount() const;

    static cv::Mat poseFromRodrigues(const cv::Mat &rvec, const cv::Mat &tvec);
    static void poseToRodrigues(const cv::Mat &pose, cv::Mat &rvec, cv::Mat &tvec);
    static cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &R);
    static void poseToEulerTvec(const cv::Mat &pose, cv::Vec3d &eulerDeg, cv::Vec3d &tvec);

private:
    cv::Mat K_;
    cv::Mat dist_;
    cv::Mat eMc_;
    cv::Mat eMcInv_;
    cv::Mat bMo_;
    cv::Mat bMoParams_;
    cv::Mat lastParams_;
    std::vector<double> priorSigma_;
    std::vector<double> pointSigmas_;
    std::vector<FrameData> frames_;
    std::vector<cv::Point3d> objPts_;

    bool isPoseSet_ = false;
    bool isObjPtsSet_ = false;

    cv::Mat computeCMo(int frameIndex, const cv::Mat &params) const;
    std::vector<double> computeResiduals(const cv::Mat &params) const;
    void computeJacobian(const cv::Mat &params, cv::Mat &J, double eps = 1e-6) const;
    double computeReprojectionError(const cv::Mat &params) const;
    void ensurePointSigmas();
    int framePosition(int frameIndex) const;
    cv::Point2d projectPoint(const cv::Point3d &pt, const cv::Mat &R, const cv::Mat &tvec) const;
    static cv::Mat applyPerturbation(const cv::Mat &T, const cv::Mat &delta);
};
