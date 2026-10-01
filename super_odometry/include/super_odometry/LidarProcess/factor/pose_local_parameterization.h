// Copyright (C) 2026 Shibo Zhao. All rights reserved.
// Contact: shibowing@gmail.com; shiboz@andrew.cmu.edu

//
// Created by shiboz on 2021-02-06.
//

#ifndef ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H
#define ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H


#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../../utils/utility.h"
// Ceres 2.2 removed LocalParameterization in favor of Manifold (added in 2.1).
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
class PoseLocalParameterization : public ceres::Manifold
{
public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
    bool PlusJacobian(const double *x, double *jacobian) const override;
    bool Minus(const double *y, const double *x, double *y_minus_x) const override;
    bool MinusJacobian(const double *x, double *jacobian) const override;
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
};
#else
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
#endif






#endif //ARISE_SLAM_MID360_POSE_LOCAL_PARAMETERIZATION_H
