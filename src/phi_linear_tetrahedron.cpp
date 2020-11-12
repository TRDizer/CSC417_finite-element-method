#include <phi_linear_tetrahedron.h>

void phi_linear_tetrahedron(Eigen::Vector4d &phi, Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, Eigen::Ref<const Eigen::Vector3d> x) {
    phi.setZero();
    
    Eigen::Vector3d X0 = V.row(element(0));
    Eigen::Vector3d X1 = V.row(element(1));
    Eigen::Vector3d X2 = V.row(element(2));
    Eigen::Vector3d X3 = V.row(element(3));

    Eigen::Matrix3d T;
    T << (X1 - X0), (X2 - X0), (X3 - X0);

    phi.tail<3>() = T.inverse() * (x - X0);
    phi(0) = 1 - phi.tail<3>().sum();
}