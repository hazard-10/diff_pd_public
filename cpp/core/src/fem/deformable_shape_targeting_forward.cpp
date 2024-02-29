#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"

// ------- Below is related to the Shape Targeting Forward pass -------
// compared to pd_forward, we made the following changes:
// 1. PDLocalStep is skipped because we use bfgs acceleration


// 1. do SetupProjectiveDynamicsSolver(method, dt, options);
//    1.1 The og Setup consider element, vertex, and muscle energies. Here we only need element. Sine the Gc matrix of Shape
//        Targeting is the same as the Gc matrix of the corotated.
//    1.2 The og Setup also consider the dirclet boundary condition. Which we also need to consider.
//    1.3 The og Setup considers the frictional boundary condition. Which we don't need to consider.
//    The final procedure is 
//        a. Add inv_h2m.
//        b. PD element energy: w_i * S'A'AS.
//        c. Assemble and pre-factorize the matrix.
//    var to write: 
//        pd_lhs_ : 'A' matrix from Ax = b
//        pd_eigen_solver_: prefactorized A matrix, used for x = A^-1 * b
//        solver_ready_: bool, if the solver is ready
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
    std::array<SparseMatrixElements, vertex_dim> nonzeros; // for each vertex_dim, 
                                                        // contains a vector of triplets for the sparse matrix,
                                                        // vector size is vertex_num
                                                        // triplets are (i, i, value). Value is used for A matrix
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
    real w = shape_target_stiffness_;
    for (int i = 0; i < element_num; ++i) { // for each hex element
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i); // get the 8 global vertex indices of the hex
        std::array<int, vertex_dim * element_dim> remap_idx; // construct a 3 x 8 array
        for (int j = 0; j < element_dim; ++j) // for each index of the 8 vertex
            for (int k = 0; k < vertex_dim; ++k) // for each dimension of the vertex dim x, y, z
                remap_idx[j * vertex_dim + k] = vertex_dim * vi[j] + k; // stors the global index of the vertex into remap_idx
                                                                        // both times vertex_dim because both is a one dim array
        for (int j = 0; j < sample_num; ++j) { // for each sample point inside the hex
            // Add w * SAAS to nonzeros.
            const SparseMatrixElements pd_AtA_nonzeros = FromSparseMatrix(finite_element_samples_[i][j].pd_AtA());
            for (const auto& triplet: pd_AtA_nonzeros) {
                const int row = triplet.row();
                const int col = triplet.col();
                const real val = triplet.value() * w * element_volume_ / sample_num; // triplet.value() is S'A'AS at the sample point
                // Skip dofs that are fixed by dirichlet boundary conditions.
                if (dirichlet_.find(remap_idx[row]) == dirichlet_.end() &&
                    dirichlet_.find(remap_idx[col]) == dirichlet_.end()) {
                    const int r = remap_idx[row];
                    const int c = remap_idx[col];
                    CheckError((r - c) % vertex_dim == 0, "AtA violates the assumption that x, y, and z are decoupled.");
                    nonzeros[r % vertex_dim].push_back(Eigen::Triplet<real>(r / vertex_dim, c / vertex_dim, val));
                }
            }
        }
    }
    
    //  Part III: Assemble and pre-factorize the matrix.
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
    // Key Parameters go through
    // q, v, q_next, v_next : # of vertices * vertex_dim
    // act :                  # of elements * 6

    // 0. Read options     
    CheckShapeTargetParam(options);
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = 0;
    int max_ls_iter = 0;
    if (use_bfgs) {
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    }
    
    omp_set_num_threads(thread_ct);

    // 1. do SetupProjectiveDynamicsSolver(method, dt, options); 
    // still need to define energy stifness for part 1
    SetupShapeTargetingSolver(dt, options);

    // 2. setup the rhs    
    const real h = dt;
    // TODO: this mass is incorrect for tri or tet meshes.
    const real mass = element_volume_ * density_;
    const real h2m = dt * dt / mass;
    const real inv_h2m = mass / (dt * dt);
    const VectorXr rhs = q + h * v + h2m * ForwardStateForce(q, v);

    // 3. original solver takes multiple iterations only to update contact_index, 
    //    the q and v is not updated in the loop. so we only call the solver once
    
    // update 2/21 
    // just realized that the act matrix is only used as actuation force / energy computation. It is not used otherwise in the PD solver.
    // And this concludes the forward pass. Only modification is the setup of a new energy and update how the accumulated force and energy is computed.
 
}




// ------- Below is contains Shape Target Energy related functions -------


// Precompute the deformation gradient auxiliary
// Compare to ComputeDeformationGradientAuxiliaryDataAndProjection, we don't need projectToManifold
// as we are dealing with R * A, not R alone
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetComputeAuxiliaryDeformationGradient(const VectorXr& q) const{
    // exact same as ComputeDeformationGradientAuxiliaryDataAndProjection
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    F_auxiliary_.resize(element_num);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        F_auxiliary_[i].resize(sample_num);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            F_auxiliary_[i][j].Initialize(F); // Extract U, V, sig, R, S
        }
    }
    // skip the projection as R * A is not provided until later
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyGetShapeTargetSMatrixFromDeformation(const std::vector<real>& q, std::vector<real>& S) const{
    // same logic as ShapeTargetComputeAuxiliaryDeformationGradient
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    VectorXr q_eigen = ToEigenVector(q);
    F_auxiliary_.resize(element_num);
    S.resize(element_num * sample_num * 6);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q_eigen, i);
        F_auxiliary_[i].resize(sample_num);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            F_auxiliary_[i][j].Initialize(F); // Extract U, V, sig, R, S
            // S is a 6x1 vector
            // transform a symmetric 3x3 matrix to a 6x1 vector
            auto S_mat = F_auxiliary_[i][j].S(); 
            S[i * sample_num * 6 + j * 6 + 0] = S_mat(0, 0);
            S[i * sample_num * 6 + j * 6 + 1] = S_mat(0, 1);
            S[i * sample_num * 6 + j * 6 + 2] = S_mat(0, 2);
            S[i * sample_num * 6 + j * 6 + 3] = S_mat(1, 1);
            S[i * sample_num * 6 + j * 6 + 4] = S_mat(1, 2);
            S[i * sample_num * 6 + j * 6 + 5] = S_mat(2, 2); 
            // this would be the ideal action for each element
        }
    }
}

// Design choice: pass actuation data everywhere with a 1d vector, 
// only assemble the matrix before solving

// ShapeTarget Energy. Modified from deformation_actuation.cpp/AcuationEnergy
template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ShapeTargetingEnergy(const VectorXr& q, const VectorXr& act) const{
    auto EnergyDensity = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& F, 
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& R,
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& A) -> real {
        return 0.5 * shape_target_stiffness_ * (F - R * A).squaredNorm();
    };
    return 0;  
}
// ShapeTarget Force
template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ShapeTargetingForce(const VectorXr& q, const VectorXr& act) const{
    auto StressTensor = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& F, 
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& R,
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& A) -> Eigen::Matrix<real, vertex_dim, vertex_dim> {
        return shape_target_stiffness_ * (F - R * A);
    };

    return VectorXr::Zero(q.size());
}




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

