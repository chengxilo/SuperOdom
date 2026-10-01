// Copyright (C) 2026 Shibo Zhao. All rights reserved.
// Contact: shibowing@gmail.com; shiboz@andrew.cmu.edu

//
// Created by shiboz on 2021-02-06.
//

#include "super_odometry/LidarProcess/factor/pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

      p = _p + dp;
      q = (_q * dq).normalized();

    return true;
}
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
// The cost functions already return their Jacobian w.r.t. the 6-dof tangent in
// the first 6 columns, so the lift is [I; 0] as with the old ComputeJacobian.
bool PoseLocalParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    (void)x;
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

// Inverse of Plus: dp = p_y - p_x, dtheta = 2 * vec(q_x^-1 * q_y).
bool PoseLocalParameterization::Minus(const double *y, const double *x, double *y_minus_x) const
{
    Eigen::Map<const Eigen::Vector3d> p_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);
    Eigen::Map<const Eigen::Vector3d> p_y(y);
    Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);

    Eigen::Quaterniond dq = q_x.conjugate() * q_y;
    if (dq.w() < 0.0)
        dq.coeffs() = -dq.coeffs();
    dq.coeffs() /= dq.w();

    Eigen::Map<Eigen::Matrix<double, 6, 1>> delta(y_minus_x);
    delta.head<3>() = p_y - p_x;
    delta.tail<3>() = 2.0 * dq.vec();

    return true;
}

bool PoseLocalParameterization::MinusJacobian(const double *x, double *jacobian) const
{
    (void)x;
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(jacobian);
    j.leftCols<6>().setIdentity();
    j.rightCols<1>().setZero();

    return true;
}
#else
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
#endif
