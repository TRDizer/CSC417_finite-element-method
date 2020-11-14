#include <Eigen/Dense>
#include <EigenTypes.h>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>

//Input:
//  x0 - initial point for newtons search
//  f(x) - function that evaluates and returns the cost function at x
//  g(dx, x) - function that evaluates and returns the gradient of the cost function in dx
//  H(dH, x) - function that evaluates and returns the Hessian in dH (as a sparse matrix).
//  max steps - the maximum newton iterations to take
//  tmp_g and tmp_H are scratch space to store gradients and hessians
//Output: 
//  x0 - update x0 to new value

// alpha (initial step length) = 1, p (scaling factor) = 0.5, c (ensure sufficient decrease) = 1e-8.
static const double alpha_max = 1.0;
static const double p = 0.5;
static const double c = 1e-8;
static const double cost_tol = 1e-8;
// line search is overall less costy than the full newton method, so a lower tolerance is given
static const double line_search_tol = 1e-12;

template<typename Objective, typename Jacobian, typename Hessian>
double newtons_method(Eigen::VectorXd &x0, Objective &f, Jacobian &g, Hessian &H, unsigned int maxSteps, Eigen::VectorXd &tmp_g, Eigen::SparseMatrixd &tmp_H) {
   double error;
   // Begin Newton's method iteration
   for (unsigned int newton_step = 0; newton_step < maxSteps; newton_step++) {
      // Calculate convergence error for the current guess
      tmp_g.setZero();
      g(tmp_g, x0);
      error = tmp_g.norm();
      if (error < cost_tol) {
         // Enters here if current guess is good enough
         // std::cout << "convergence reached after " << newton_step << " steps!" << std::endl;
         return error;
      }

      // Current guess does not converge, perhaps too ambitious
      // Proceed to line search to tune it down 
      tmp_H.setZero();
      H(tmp_H, x0);
      // Quadratic minimization: H_i * d = -g_i
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(tmp_H);
      Eigen::VectorXd d = solver.solve(-tmp_g);

      // Begin line search iteration
      double alpha = alpha_max;
      const double line_search_obj = f(x0) + c * d.dot(tmp_g);
      // std::cout << "Target decrease: " << (c * d.dot(tmp_g)) << std::endl;
      // std::cout << "Decrease is negative? : " << (c * d.dot(tmp_g) <=0 ? "y" : "n" ) << std::endl;

      // While decrease is not sufficient and we still have space to further shrink alpha
      while (
         (f(x0 + alpha * d) > line_search_obj) && (alpha >= line_search_tol)
         ) {
         alpha *= p;
      }

      // std::cout << "Post line search alpha: " << alpha << std::endl;
      x0 += alpha * d;
   }

   // get error of the last step if things fall through
   // std::cout << "Failed to converge after " << maxSteps << "steps!" << std::endl;
   error = tmp_g.norm();
   return error;
}
