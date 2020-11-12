#include <T_linear_tetrahedron.h>
#include <mass_matrix_linear_tetrahedron.h>
#include <iostream>

void T_linear_tetrahedron(double &T, Eigen::Ref<const Eigen::VectorXd> qdot, Eigen::Ref<const Eigen::RowVectorXi> element, double density, double volume) {
    // std::cout << "qdot size in T linear tetrahedron: " << qdot.size() << std::endl;
    Eigen::Matrix1212d M;
    Eigen::Vector12d target_qdot;
    target_qdot << qdot.segment<3>(element(0) * 3),
                   qdot.segment<3>(element(1) * 3),
                   qdot.segment<3>(element(2) * 3),
                   qdot.segment<3>(element(3) * 3);
    mass_matrix_linear_tetrahedron(M, target_qdot, element, density, volume);
    T = 0.5 * target_qdot.transpose() * M * target_qdot;
}