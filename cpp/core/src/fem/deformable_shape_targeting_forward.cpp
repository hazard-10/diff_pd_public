#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"

// Below is related to the Shape Targeting Forward
// contains the following functions:
// 1. SetupShapeTargetingSolver
// 2. ShapeTargetingForward

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupShapeTargetingSolver(const real dt, const std::map<std::string, real>& options) const{
    if (pd_solver_ready_) return; // only setup once
    CheckError(options.find("thread_ct") != options.end(), "Missing parameter thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    // inv_h2m + w_i * S'A'AS + w_i * S'A'M'MAS.
    // Assemble and pre-factorize the left-hand-side matrix.
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    const int vertex_num = mesh_.NumOfVertices();
    const real mass = density_ * element_volume_;
    const real inv_h2m = mass / (dt * dt);
    std::array<SparseMatrixElements, vertex_dim> nonzeros;
    // Part I: Add inv_h2m.
    #pragma omp parallel for
    for (int k = 0; k < vertex_dim; ++k) {
        for (int i = 0; i < vertex_num; ++i) {
            const int dof = i * vertex_dim + k;
            if (dirichlet_.find(dof) != dirichlet_.end())
                nonzeros[k].push_back(Eigen::Triplet<real>(i, i, 1));
            else
                nonzeros[k].push_back(Eigen::Triplet<real>(i, i, inv_h2m));
        }
    }
    
    // Part II: PD element energy: w_i * S'A'AS.
    // TODO: define energy
    
    // Assemble and pre-factorize the matrix.
    for (int i = 0; i < vertex_dim; ++i) {
        pd_lhs_[i] = ToSparseMatrix(vertex_num, vertex_num, nonzeros[i]);
        pd_eigen_solver_[i].compute(pd_lhs_[i]);
        CheckError(pd_eigen_solver_[i].info() == Eigen::Success, "Cholesky solver failed to factorize the matrix.");
    }
    
    pd_solver_ready_ = true;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetingForward(const VectorXr& q, const VectorXr& v, const VectorXr& act, const real dt,
        const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const{
    // skip reading param for now

    // 1. do SetupProjectiveDynamicsSolver(method, dt, options);
    //    1.1 The og Setup consider element, vertex, and muscle energies. Here we only need element. Sine the Gc matrix of Shape
    //        Targeting is the same as the Gc matrix of the corotated.
    //    1.2 The og Setup also consider the dirclet boundary condition. Which we also need to consider.
    //    1.3 The og Setup considers the frictional boundary condition. Which we don't need to consider.
    //    The final procedure is 
    //    a. do Part I: Add inv_h2m.
    //    b. do Part II: PD element energy: w_i * S'A'AS.
    //    c. do // Assemble and pre-factorize the matrix.
    //    var to write: pd_lhs_, pd_eigen_solver_, solver_ready_
    
    // still need to define energy stifness for part 1

    // 2. setup the rhs

    // 3. similar to the fixed contact, we need to call PdNonlinearSolve
    
    // update 2/21 
    // just realized that the a matrix is only used as actuation force / energy computation. It is not used otherwise in the PD solver.
    // And this concludes the forward pass. Only modification is the setup of a new energy and update how the accumulated force and energy is computed.

    // as test, set all entries of q_next to 112
    q_next = VectorXr::Constant(q.size(), 112);
}


template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

