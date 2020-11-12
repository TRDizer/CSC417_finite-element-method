#include <dphi_linear_tetrahedron_dX.h>
#include <phi_linear_tetrahedron.h>
#include <iostream>
void dphi_linear_tetrahedron_dX(Eigen::Matrix43d &dphi, Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, Eigen::Ref<const Eigen::Vector3d> X) {
    //        dphi0/dx  dphi0/dy  dphi0/dz
    // dphi = dphi1/dx  dphi1/dy  dphi1/dz = | -1^T * T^-1 |
    //        dphi2/dx  dphi2/dy  dphi2/dz   |    T^-1     |
    //        dphi3/dx  dphi3/dy  dphi3/dz    
    dphi.setZero();

    Eigen::Vector3d X0 = V.row(element(0));
    Eigen::Vector3d X1 = V.row(element(1));
    Eigen::Vector3d X2 = V.row(element(2));
    Eigen::Vector3d X3 = V.row(element(3));
    Eigen::Matrix3d T;
    T << (X1 - X0), (X2 - X0), (X3 - X0);

    dphi.block<3,3>(1,0) = T.inverse();
    // I think Eigen is a bit more optimized with its built-in reduction method than multiplication sum
    // dphi.block<1,3>(0,0) = -1.0 * Eigen::RowVector3d::Ones() * dphi.block<3,3>(1,0);
    dphi.block<1,3>(0,0) = dphi.block<3,3>(1,0).colwise().sum() * -1.0;
}