#include <build_skinning_matrix.h>
#include <phi_linear_tetrahedron.h>
#include <vector>
#include <iostream>
#include <igl/AABB.h>
#include <igl/in_element.h>

void build_skinning_matrix(Eigen::SparseMatrixd &N, Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> T, 
                                                   Eigen::Ref<const Eigen::MatrixXd> V_skin) {
    // Reference https://libigl.github.io/tutorial/#point-location
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init((Eigen::MatrixXd) V, (Eigen::MatrixXi) T);
    Eigen::VectorXi I;
    igl::in_element((Eigen::MatrixXd) V, (Eigen::MatrixXi) T, (Eigen::MatrixXd) V_skin, tree, I);

    N.resize(V_skin.rows(), V.rows());
    N.setZero();
    std::vector<Eigen::Triplet<double>> N_entries;

    for (int i = 0; i < V_skin.rows(); i++) {
        Eigen::Vector4d phi;
        Eigen::RowVector4i bounding_tetrahedra = T.row(I(i));
        phi_linear_tetrahedron(phi, V, bounding_tetrahedra, V_skin.row(i).transpose());
        for (int vertex_i = 0; vertex_i < 4; vertex_i++) {
            N_entries.push_back(Eigen::Triplet<double>(
                i,
                bounding_tetrahedra(vertex_i),
                phi(vertex_i)
            ));
        }
    }

    N.setFromTriplets(N_entries.begin(), N_entries.end());
}