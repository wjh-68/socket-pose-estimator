#include "StaticPoseOptimizer.h"
#include <cmath>
#include <numeric>

StaticPoseOptimizer::StaticPoseOptimizer(const cv::Mat &K,
                                         const cv::Mat &dist,
                                         const std::vector<double> &priorSigma,
                                         const std::vector<double> &pointSigmas)
    : K_(K.clone()), dist_(dist.clone()), eMc_(cv::Mat::eye(4, 4, CV_64F)), eMcInv_(cv::Mat::eye(4, 4, CV_64F))
{
    if (K_.empty() || K_.rows != 3 || K_.cols != 3)
    {
        throw std::runtime_error("K must be a 3x3 matrix");
    }
    if (dist_.empty() || dist_.total() != 5)
    {
        throw std::runtime_error("dist must be a 1x5 vector or matrix");
    }

    if (!priorSigma.empty())
    {
        priorSigma_ = priorSigma;
    }
    else
    {
        priorSigma_ = {5.0 * CV_PI / 180.0, 5.0 * CV_PI / 180.0, 1.0 * CV_PI / 180.0, 0.5, 0.5, 10.0};
    }

    if (!pointSigmas.empty())
    {
        setPointSigmas(pointSigmas);
    }
}

void StaticPoseOptimizer::setExtrinsics(const cv::Mat &eMc)
{
    if (eMc.empty() || eMc.rows != 4 || eMc.cols != 4)
    {
        throw std::runtime_error("eMc must be a 4x4 matrix");
    }
    eMc_ = eMc.clone();
    eMcInv_ = eMc_.inv(cv::DECOMP_SVD);
}

void StaticPoseOptimizer::setObjectPoints(const std::vector<cv::Point3d> &objPts)
{
    objPts_ = objPts;
    isObjPtsSet_ = true;
    if (pointSigmas_.empty())
    {
        pointSigmas_.assign(objPts_.size(), 1.0);
    }
}

void StaticPoseOptimizer::setPointSigmas(const std::vector<double> &sigmas)
{
    if (!objPts_.empty() && static_cast<int>(sigmas.size()) != static_cast<int>(objPts_.size()))
    {
        throw std::runtime_error("pointSigmas length does not match object points length");
    }
    for (double sigma : sigmas)
    {
        if (sigma <= 0.0)
        {
            throw std::runtime_error("pointSigmas must be positive");
        }
    }
    pointSigmas_ = sigmas;
}

void StaticPoseOptimizer::setInitialPose(const cv::Mat &pose)
{
    if (pose.empty() || pose.rows != 4 || pose.cols != 4)
    {
        throw std::runtime_error("Initial pose must be a 4x4 matrix");
    }
    bMo_ = pose.clone();
    bMoParams_ = cv::Mat(6, 1, CV_64F);
    cv::Mat rvec, tvec;
    poseToRodrigues(bMo_, rvec, tvec);
    rvec.copyTo(bMoParams_.rowRange(0, 3));
    tvec.copyTo(bMoParams_.rowRange(3, 6));
    isPoseSet_ = true;
}

void StaticPoseOptimizer::addFrame(int frameIndex,
                                   const cv::Mat &robotPose,
                                   const std::vector<cv::Point2d> &pts2d,
                                   const std::vector<cv::Point3d> &pts3d,
                                   const std::vector<double> &perPointErrorsPNP)
{
    if (robotPose.empty() || robotPose.rows != 4 || robotPose.cols != 4)
    {
        throw std::runtime_error("robotPose must be a 4x4 matrix");
    }
    if (pts2d.empty())
    {
        throw std::runtime_error("pts2d must not be empty");
    }
    std::vector<cv::Point3d> objectPoints = pts3d;
    if (objectPoints.empty())
    {
        if (!isObjPtsSet_)
        {
            throw std::runtime_error("Object points are not set");
        }
        objectPoints = objPts_;
    }
    if (static_cast<int>(objectPoints.size()) != static_cast<int>(pts2d.size()))
    {
        throw std::runtime_error("pts2d and pts3d size mismatch");
    }
    if (!perPointErrorsPNP.empty() && perPointErrorsPNP.size() != pts2d.size())
    {
        throw std::runtime_error("perPointErrorsPNP length does not match pts2d length");
    }

    FrameData frame;
    frame.frameIndex = frameIndex;
    frame.robotPose = robotPose.clone();
    frame.pts2d = pts2d;
    frame.pts3d = objectPoints;
    frame.perPointErrorsPNP = perPointErrorsPNP;
    frames_.push_back(frame);
}

void StaticPoseOptimizer::removeOldestFrame()
{
    if (!frames_.empty())
    {
        frames_.erase(frames_.begin());
    }
}

bool StaticPoseOptimizer::optimize(int maxIters, double lambdaInit)
{
    if (frames_.empty())
    {
        throw std::runtime_error("No frames added for optimization");
    }
    if (!isPoseSet_)
    {
        throw std::runtime_error("Initial pose has not been set");
    }
    if (pointSigmas_.empty())
    {
        ensurePointSigmas();
    }

    int nFrames = static_cast<int>(frames_.size());
    int nParams = 6 + 6 * nFrames;
    cv::Mat params = cv::Mat::zeros(nParams, 1, CV_64F);
    bMoParams_.copyTo(params.rowRange(0, 6));

    double lambda = lambdaInit;
    double prevError = computeReprojectionError(params);

    for (int iter = 0; iter < maxIters; ++iter)
    {
        std::vector<double> residuals = computeResiduals(params);
        if (residuals.empty())
        {
            return false;
        }
        cv::Mat r = cv::Mat(residuals).reshape(1, static_cast<int>(residuals.size()));
        cv::Mat J;
        computeJacobian(params, J);

        cv::Mat H = J.t() * J;
        H += lambda * cv::Mat::eye(H.size(), H.type());
        cv::Mat g = J.t() * r;
        cv::Mat dx;
        bool solved = cv::solve(H, -g, dx, cv::DECOMP_CHOLESKY);
        if (!solved)
        {
            solved = cv::solve(H, -g, dx, cv::DECOMP_SVD);
            if (!solved)
            {
                return false;
            }
        }

        if (dx.rows != nParams || dx.cols != 1)
        {
            return false;
        }

        cv::Mat nextParams = params + dx;
        double nextError = computeReprojectionError(nextParams);

        if (nextError < prevError)
        {
            params = nextParams;
            prevError = nextError;
            lambda *= 0.1;
            if (cv::norm(dx) < 1e-6)
            {
                break;
            }
        }
        else
        {
            lambda *= 10.0;
        }
    }

    bMo_ = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat rvec = params.rowRange(0, 3).clone();
    cv::Mat tvec = params.rowRange(3, 6).clone();
    bMo_.rowRange(0, 3).colRange(0, 3) = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    R.copyTo(bMo_.rowRange(0, 3).colRange(0, 3));
    tvec.copyTo(bMo_.rowRange(0, 3).col(3));
    bMoParams_ = params.rowRange(0, 6).clone();
    lastParams_ = params.clone();
    return true;
}

cv::Mat StaticPoseOptimizer::getPose() const
{
    return bMo_.clone();
}

double StaticPoseOptimizer::getAverageError() const
{
    if (!isPoseSet_ || frames_.empty())
    {
        return -1.0;
    }
    if (!lastParams_.empty())
    {
        return computeReprojectionError(lastParams_);
    }
    cv::Mat params = cv::Mat::zeros(6 + 6 * static_cast<int>(frames_.size()), 1, CV_64F);
    bMoParams_.copyTo(params.rowRange(0, 6));
    return computeReprojectionError(params);
}

double StaticPoseOptimizer::getFrameError(int frameIndex) const
{
    int pos = framePosition(frameIndex);
    if (pos < 0)
    {
        throw std::runtime_error("Frame index not found");
    }
    if (!isPoseSet_)
    {
        throw std::runtime_error("Initial pose has not been set");
    }
    cv::Mat params;
    if (!lastParams_.empty())
    {
        params = lastParams_.clone();
    }
    else
    {
        params = cv::Mat::zeros(6 + 6 * static_cast<int>(frames_.size()), 1, CV_64F);
        bMoParams_.copyTo(params.rowRange(0, 6));
    }
    cv::Mat cMo = computeCMo(pos, params);
    cv::Mat R = cMo.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = cMo.rowRange(0, 3).col(3).clone();

    const FrameData &frame = frames_[pos];
    double sumError = 0.0;
    for (size_t i = 0; i < frame.pts2d.size(); i++)
    {
        cv::Point2d proj = projectPoint(frame.pts3d[i], R, t);
        double dx = proj.x - frame.pts2d[i].x;
        double dy = proj.y - frame.pts2d[i].y;
        sumError += std::sqrt(dx * dx + dy * dy);
    }
    return sumError / static_cast<double>(frame.pts2d.size());
}

cv::Mat StaticPoseOptimizer::poseFromRodrigues(const cv::Mat &rvec, const cv::Mat &tvec)
{
    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    R.copyTo(pose.rowRange(0, 3).colRange(0, 3));
    tvec.copyTo(pose.rowRange(0, 3).col(3));
    return pose;
}

void StaticPoseOptimizer::poseToRodrigues(const cv::Mat &pose, cv::Mat &rvec, cv::Mat &tvec)
{
    cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
    cv::Rodrigues(R, rvec);
    tvec = pose.rowRange(0, 3).col(3).clone();
}

cv::Vec3d StaticPoseOptimizer::rotationMatrixToEulerAngles(const cv::Mat &R)
{
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0.0;
    }
    return cv::Vec3d(x, y, z);
}

void StaticPoseOptimizer::poseToEulerTvec(const cv::Mat &pose, cv::Vec3d &eulerDeg, cv::Vec3d &tvec)
{
    cv::Vec3d euler = rotationMatrixToEulerAngles(pose.rowRange(0, 3).colRange(0, 3));
    eulerDeg = cv::Vec3d(euler[0] * 180.0 / CV_PI,
                         euler[1] * 180.0 / CV_PI,
                         euler[2] * 180.0 / CV_PI);
    tvec = cv::Vec3d(pose.at<double>(0, 3), pose.at<double>(1, 3), pose.at<double>(2, 3));
}

cv::Mat StaticPoseOptimizer::computeCMo(int frameIndex, const cv::Mat &params) const
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames_.size()))
    {
        throw std::runtime_error("Invalid frame position");
    }
    cv::Mat bMo = poseFromRodrigues(params.rowRange(0, 3), params.rowRange(3, 6));
    const FrameData &frame = frames_[frameIndex];
    cv::Mat bMe = frame.robotPose;
    cv::Mat eMb = bMe.inv(cv::DECOMP_SVD);
    cv::Mat cMoNominal = eMcInv_ * eMb * bMo;
    cv::Mat delta = params.rowRange(6 + 6 * frameIndex, 6 + 6 * frameIndex + 6).clone();
    return applyPerturbation(cMoNominal, delta);
}

std::vector<double> StaticPoseOptimizer::computeResiduals(const cv::Mat &params) const
{
    if (frames_.empty())
    {
        return {};
    }
    std::vector<double> residuals;
    cv::Mat bMo = poseFromRodrigues(params.rowRange(0, 3), params.rowRange(3, 6));
    for (int frameIndex = 0; frameIndex < static_cast<int>(frames_.size()); ++frameIndex)
    {
        const FrameData &frame = frames_[frameIndex];
        cv::Mat bMe = frame.robotPose;
        cv::Mat eMb = bMe.inv(cv::DECOMP_SVD);
        cv::Mat cMoNominal = eMcInv_ * eMb * bMo;
        cv::Mat delta = params.rowRange(6 + 6 * frameIndex, 6 + 6 * (frameIndex + 1)).clone();
        cv::Mat cMo = applyPerturbation(cMoNominal, delta);
        cv::Mat R = cMo.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = cMo.rowRange(0, 3).col(3).clone();

        for (size_t i = 0; i < frame.pts2d.size(); ++i)
        {
            cv::Point2d proj = projectPoint(frame.pts3d[i], R, t);
            double sigma = 1.0;
            if (!pointSigmas_.empty())
            {
                sigma = pointSigmas_[static_cast<int>(i)];
            }
            if (!frame.perPointErrorsPNP.empty())
            {
                double clipped = std::min(frame.perPointErrorsPNP[i], 3.0);
                sigma *= 1.0 + 0.2 * clipped;
                if (frame.perPointErrorsPNP[i] > 20.0)
                {
                    sigma = 1e6;
                }
            }
            residuals.push_back((proj.x - frame.pts2d[i].x) / sigma);
            residuals.push_back((proj.y - frame.pts2d[i].y) / sigma);
        }

        for (int k = 0; k < 6; ++k)
        {
            residuals.push_back(params.at<double>(6 + 6 * frameIndex + k, 0) / priorSigma_[k]);
        }
    }
    return residuals;
}

void StaticPoseOptimizer::computeJacobian(const cv::Mat &params, cv::Mat &J, double eps) const
{
    std::vector<double> base = computeResiduals(params);
    int nRes = static_cast<int>(base.size());
    int nParams = params.rows;

    J = cv::Mat(nRes, nParams, CV_64F);
    for (int i = 0; i < nParams; ++i)
    {
        cv::Mat perturbed = params.clone();
        perturbed.at<double>(i, 0) += eps;
        std::vector<double> r2 = computeResiduals(perturbed);
        for (int j = 0; j < nRes; ++j)
        {
            double deriv = (r2[j] - base[j]) / eps;
            J.at<double>(j, i) = deriv;
        }
    }
}

double StaticPoseOptimizer::computeReprojectionError(const cv::Mat &params) const
{
    std::vector<double> residuals = computeResiduals(params);
    double sumSquares = 0.0;
    for (double v : residuals)
    {
        sumSquares += v * v;
    }
    return sumSquares / static_cast<double>(residuals.size());
}

void StaticPoseOptimizer::ensurePointSigmas()
{
    if (pointSigmas_.empty() && !objPts_.empty())
    {
        pointSigmas_.assign(objPts_.size(), 1.0);
    }
}

int StaticPoseOptimizer::framePosition(int frameIndex) const
{
    for (int i = 0; i < static_cast<int>(frames_.size()); ++i)
    {
        if (frames_[i].frameIndex == frameIndex)
        {
            return i;
        }
    }
    return -1;
}

int StaticPoseOptimizer::getFrameCount() const
{
    return static_cast<int>(frames_.size());
}

cv::Point2d StaticPoseOptimizer::projectPoint(const cv::Point3d &pt, const cv::Mat &R, const cv::Mat &tvec) const
{
    cv::Mat pt3d = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
    cv::Mat cam = R * pt3d + tvec;
    double x = cam.at<double>(0, 0) / cam.at<double>(2, 0);
    double y = cam.at<double>(1, 0) / cam.at<double>(2, 0);
    double r2 = x * x + y * y;
    double k1 = dist_.at<double>(0, 0);
    double k2 = dist_.at<double>(0, 1);
    double p1 = dist_.at<double>(0, 2);
    double p2 = dist_.at<double>(0, 3);
    double k3 = dist_.at<double>(0, 4);
    double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    double xDist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double yDist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    double u = K_.at<double>(0, 0) * xDist + K_.at<double>(0, 2);
    double v = K_.at<double>(1, 1) * yDist + K_.at<double>(1, 2);
    return cv::Point2d(u, v);
}

cv::Mat StaticPoseOptimizer::applyPerturbation(const cv::Mat &T, const cv::Mat &delta)
{
    cv::Mat dR;
    cv::Rodrigues(delta.rowRange(0, 3), dR);
    cv::Mat dt = delta.rowRange(3, 6).clone();
    cv::Mat Tdelta = cv::Mat::eye(4, 4, CV_64F);
    dR.copyTo(Tdelta.rowRange(0, 3).colRange(0, 3));
    dt.copyTo(Tdelta.rowRange(0, 3).col(3));
    return T * Tdelta;
}
