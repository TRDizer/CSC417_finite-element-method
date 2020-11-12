#include <V_linear_tetrahedron.h>

#include <dphi_linear_tetrahedron_dX.h>
#include <psi_neo_hookean.h>
#include <quadrature_single_point.h>

void V_linear_tetrahedron(double &energy, Eigen::Ref<const Eigen::VectorXd> q, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double volume,
                          double C, double D) {

    // The reason for removing X from parameters is because dphi_linear_tetrahedron_dX yields a constant gradient hence invarient 
    // to the choice of X as long as it belongs to the tetrahedra, but X will be provided as a sanity check for grasp of concept 
    // by selecting the undeformed centroid to be it.
    auto neohookean_linear_tet = [&](double &e, Eigen::Ref<const Eigen::VectorXd>q, Eigen::Ref<const Eigen::RowVectorXi> element) {
        // Eigen::Vector3d X = Eigen::Vector3d::Zero();
        // for (int i = 0; i < element.size(); i++) {
        //     X += V.row(element(i)).transpose();
        // }
        // X /= (double) element.size();

        // Speed up the code by simply choosing X0
        Eigen::Vector3d X = V.row(element(0)).transpose();


        Eigen::Matrix34d x;
        Eigen::Matrix43d dphi;
        x << q.segment<3>(element(0) * 3),
             q.segment<3>(element(1) * 3),
             q.segment<3>(element(2) * 3),
             q.segment<3>(element(3) * 3);
        dphi_linear_tetrahedron_dX(dphi, V, element, X);
        // F = x * dphi
        psi_neo_hookean(e, x*dphi, C, D);
    };

    quadrature_single_point(energy, q, element, volume, neohookean_linear_tet);  
    
}