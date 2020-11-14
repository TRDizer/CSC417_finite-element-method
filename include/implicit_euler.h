#include <Eigen/Dense>
#include <EigenTypes.h>
#include <newtons_method.h>
#include <iostream>

//Input:
//  q - generalized coordinates for the FEM system
//  qdot - generalized velocity for the FEM system
//  dt - the time step in seconds
//  mass - the mass matrix
//  energy(q, qdot) -  a function that computes the energy of the FEM system. This takes q and qdot as parameters, returns the energy value.
//  force(f, q, qdot) - a function that computes the force acting on the FEM system. This takes q and qdot as parameters, returns the force in f.
//  stiffness(K, q, qdot) - a function that computes the stiffness (negative second derivative of the potential energy). This takes q and qdot as parameters, returns the stiffness matrix in K.  
//  tmp_qdot - scratch space for storing velocities 
//  tmp_force - scratch space to collect forces
//  tmp_stiffness - scratch space to collect stiffness matrix
//Output:
//  q - set q to the updated generalized coordinate using linearly implicit time integration
//  qdot - set qdot to the updated generalized velocity using linearly implicit time integration
template<typename ENERGY, typename FORCE, typename STIFFNESS> 
inline void implicit_euler(Eigen::VectorXd &q, Eigen::VectorXd &qdot, double dt, 
                            const Eigen::SparseMatrixd &mass,  ENERGY &energy, FORCE &force, STIFFNESS &stiffness, 
                            Eigen::VectorXd &tmp_qdot, Eigen::VectorXd &tmp_force, Eigen::SparseMatrixd &tmp_stiffness) {
    // Define cost function, its gradient and hessian
    auto cost = [&](const Eigen::VectorXd &v) {
        return 0.5 * (v - qdot).transpose() * mass * (v - qdot) + energy(v);
    };

    auto grad_cost = [&](Eigen::VectorXd &d_cost, Eigen::VectorXd &v) {
        tmp_force.setZero();
        force(tmp_force, q + dt * v, v);
        d_cost = mass * (v - qdot) - dt * tmp_force;
    };

    auto hess_cost = [&](Eigen::SparseMatrixd &d2_cost, Eigen::VectorXd &v) {
        tmp_stiffness.setZero();
        stiffness(tmp_stiffness, q + dt * v, v);
        d2_cost = mass - dt * dt * tmp_stiffness;
    };
    
    // std::cout << "========Begin Newton's Method========" << std::endl;

    // // Use linearly implicit euler solution as initial guess
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    // stiffness(tmp_stiffness, q, qdot);
    // force(tmp_force, q, qdot);
    // solver.compute(mass - dt * dt * tmp_stiffness);
    // tmp_qdot = solver.solve(mass * qdot + dt * tmp_force);
    // double error = newtons_method(tmp_qdot, cost, grad_cost, hess_cost, 5, tmp_force, tmp_stiffness);

    // Use current qdot as initial guess
    tmp_qdot = qdot;
    double error = newtons_method(tmp_qdot, cost, grad_cost, hess_cost, 5, Eigen::VectorXd(), Eigen::SparseMatrixd());
    // std::cout << "Current update error: " << error << std::endl;

    // std::cout << "========After Newton's Method========" << std::endl;

    qdot = tmp_qdot;
    q = q + dt * qdot;
}
