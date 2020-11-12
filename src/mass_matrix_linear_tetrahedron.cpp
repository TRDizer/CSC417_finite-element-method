 
 #include <mass_matrix_linear_tetrahedron.h>

 void mass_matrix_linear_tetrahedron(Eigen::Matrix1212d &M, Eigen::Ref<const Eigen::VectorXd> qdot, Eigen::Ref<const Eigen::RowVectorXi> element, double density, double volume) {
    //                   [1/10, 1/20, 1/20, 1/20]
    // M_0 = rho * vol * [1/20, 1/10, 1/20, 1/20]
    //                   [1/20, 1/20, 1/10, 1/20]
    //                   [1/20, 1/20, 1/20, 1/10]
    M.setZero();

    Eigen::Matrix4d M_0;
    M_0 << 1.0/10.0, 1.0/20.0, 1.0/20.0, 1.0/20.0,
           1.0/20.0, 1.0/10.0, 1.0/20.0, 1.0/20.0,
           1.0/20.0, 1.0/20.0, 1.0/10.0, 1.0/20.0,
           1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/10.0;
    M_0 *= density * volume;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M.block<3,3>(i*3, j*3) = Eigen::Matrix3d::Identity() * M_0(i,j);
        }
    }
 }