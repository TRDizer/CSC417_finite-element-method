#include <assemble_forces.h>
#include <iostream>

void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::MatrixXd> qdot, 
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> T, Eigen::Ref<const Eigen::VectorXd> v0,
                     double C, double D) { 
    f.resize(q.size());
    f.setZero();
        
    for (int i = 0; i < T.rows(); i++) {
        Eigen::RowVector4i current_tetrahedron = T.row(i);

        // // dummy step for sanity check as dV_linear_tetrahedron_dq does not need q at all
        // Eigen::Vector12d current_tetrahedron_q;
        // current_tetrahedron_q << q.segment<3>(current_tetrahedron(0) * 3),
        //                          q.segment<3>(current_tetrahedron(1) * 3),
        //                          q.segment<3>(current_tetrahedron(2) * 3),
        //                          q.segment<3>(current_tetrahedron(3) * 3);

        Eigen::Vector12d dV;
        dV_linear_tetrahedron_dq(dV, q, V, current_tetrahedron, v0(i), C, D);

        for (int vertex_i = 0; vertex_i < 4; vertex_i++) {
            f.segment<3>(current_tetrahedron(vertex_i) * 3) -= dV.segment<3>(vertex_i * 3);
        }
    }
};