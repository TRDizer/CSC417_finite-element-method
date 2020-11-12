#include <mass_matrix_mesh.h>
#include <mass_matrix_linear_tetrahedron.h>
#include <vector>

//Input:
//  qdot - generalized velocity for the FEM system
//  T - the mx4 vertex indices for tet mesh
//  density - density of material
//  v0 - the undeformed tetrahedra volumes
//Output:
//  M - Sparse mass matrix for the whole mesh.
void mass_matrix_mesh(Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::VectorXd> qdot, Eigen::Ref<const Eigen::MatrixXi> T, double density, Eigen::Ref<const Eigen::VectorXd> v0) {
    M.resize(qdot.size(), qdot.size());
    M.setZero();
    std::vector<Eigen::Triplet<double>> M_entries;

    for (int i = 0; i < T.rows(); i++) {
        Eigen::RowVector4i current_tetrahedron = T.row(i);

        // dummy step for sanity check as mass_matrix_linear_tetrahedron does not need qdot at all
        Eigen::Vector12d current_tetrahedron_qdot;
        current_tetrahedron_qdot << qdot.segment<3>(current_tetrahedron(0) * 3),
                                    qdot.segment<3>(current_tetrahedron(1) * 3),
                                    qdot.segment<3>(current_tetrahedron(2) * 3),
                                    qdot.segment<3>(current_tetrahedron(3) * 3);

        Eigen::Matrix1212d current_tetrahedron_M;
        mass_matrix_linear_tetrahedron(current_tetrahedron_M, current_tetrahedron_qdot, T.row(i), density, v0(i));
        
        // Iterate to populate 16 total d2V/d(corner_i)(corner_j) blocks
        for (int phi_i = 0; phi_i < 4; phi_i++) {
            for (int phi_j = 0; phi_j < 4; phi_j++) {
                // Fill up diagonal entry of each block
                for (int qdot_i = 0; qdot_i < 3; qdot_i++) {
                    M_entries.push_back(
                        Eigen::Triplet<double>(
                            current_tetrahedron(phi_i) * 3 + qdot_i, 
                            current_tetrahedron(phi_j) * 3 + qdot_i, 
                            current_tetrahedron_M(phi_i * 3 + qdot_i, phi_j * 3 + qdot_i)
                        )
                    );
                }
            }
        }
    }

    M.setFromTriplets(M_entries.begin(), M_entries.end());
}