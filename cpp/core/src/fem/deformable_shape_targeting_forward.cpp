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
void Deformable<vertex_dim, element_dim>::SetupShapeTargetingSolver(const std::map<std::string, real>& options) const{
    if (pd_solver_ready_) return; // only setup once
    CheckError(options.find("thread_ct") != options.end(), "Missing parameter thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    // inv_h2m + w_i * S'A'AS + w_i * S'A'M'MAS.
    // Assemble and pre-factorize the left-hand-side matrix.
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    const int vertex_num = mesh_.NumOfVertices();
    std::array<SparseMatrixElements, vertex_dim> nonzeros; // for each vertex_dim, 
                                                        // contains a vector of triplets for the sparse matrix,
                                                        // vector size is vertex_num
                                                        // triplets are (i, i, value). Value is used for A matrix
    // Part I: Add inv_h2m.
    // removed because quasistatic simulation doesn't involve mass nor acceleration
    
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
void Deformable<vertex_dim, element_dim>::ShapeTargetingForward(const VectorXr& q, const VectorXr& act, 
        const std::map<std::string, real>& options, VectorXr& q_next ) const{
    // Key Parameters go through
    // q, v, q_next, v_next : # of vertices * vertex_dim
    // act :                  # of elements * 6

    // 0. Read options     
    CheckShapeTargetParam(options);
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    // const int verbose_level = static_cast<int>(options.at("verbose"));
    // re-use verbose when needed
    
    omp_set_num_threads(thread_ct);

    // 1. do SetupProjectiveDynamicsSolver(method, dt, options); 
    // still need to define energy stifness for part 1
    SetupShapeTargetingSolver(options);

    // 2. setup the rhs     
    const VectorXr rhs = q; // + h * v + h2m * ForwardStateForce(q, v);

    // 3. original solver takes multiple iterations only to update contact_index, 
    //    the q and v is not updated in the loop. so we only call the solver once
    
    std::map<int, real> additional_dirichlet; 
    // Initial guess.
    const VectorXr q_sol = ShapeTargetNonlinearSolve(q, act, rhs, additional_dirichlet, options); 
    q_next = q_sol;
    return;
    // update 2/21 
    // just realized that the act matrix is only used as actuation force / energy computation. It is not used otherwise in the PD solver.
    // And this concludes the forward pass. Only modification is the setup of a new energy and update how the accumulated force and energy is computed.
 
}


template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ShapeTargetNonlinearSolve(const VectorXr& q_init, const VectorXr& act, 
        const VectorXr& rhs, const std::map<int, real>& additional_dirichlet,
        const std::map<std::string, real>& options) const {

    // 0. Read options, prepare variables
    // print the first 10 entries of the q_init
    CheckShapeTargetParam(options);
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    // const int verbose_level = static_cast<int>(options.at("verbose"));
    const bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = 0;
    int max_ls_iter = 0;
    if (use_bfgs) {
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    }
    bool use_acc = true;
    bool use_sparse = false;
    const std::string method = "pd_eigen";

    std::map<int, real> augmented_dirichlet = dirichlet_;
    for (const auto& pair : additional_dirichlet)
        augmented_dirichlet[pair.first] = pair.second;

    // 1. initialize variables
    VectorXr q_sol = q_init; 
    // Enforce dirichlet boundary conditions.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : augmented_dirichlet) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    } 
    ShapeTargetComputeAuxiliaryDeformationGradient(q_sol); // do this every time q_sol is updated
 
    VectorXr force_sol = ShapeTargetingForce(q_sol, act);
    real energy_sol = ShapeTargetingEnergy(q_sol, act);    
    auto eval_obj = [&](const VectorXr& q_cur, const real energy_cur) {
        return energy_cur;
        // return 0.5 * (q_cur - rhs).dot(inv_h2m * (q_cur - rhs)) + energy_cur;
    };
    real obj_sol = eval_obj(q_sol, energy_sol);
    VectorXr grad_sol = -1 * force_sol.array() * selected.array();
    bool success = false;
    // Initialize queues for BFGS. Better explained in the PDNonlinearSolve function
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history; 
    for (int i = 0; i < max_pd_iter; ++i) {
        if (use_bfgs) {
            // BFGS's direction: quasi_newton_direction = B * grad_sol.
            VectorXr quasi_newton_direction = VectorXr::Zero(dofs_); 
            // Current solution: q_sol.
            // Current gradient: grad_sol.
            const int bfgs_size = static_cast<int>(xi_history.size());
            if (bfgs_size == 0) {
                // Initially, the queue is empty. We use A as our initial guess of Hessian (not the inverse!).
                xi_history.push_back(q_sol);
                gi_history.push_back(grad_sol);
                quasi_newton_direction = PdLhsSolve(method, grad_sol, additional_dirichlet, use_acc, use_sparse); 
            } else {
                const VectorXr q_sol_last = xi_history.back();
                const VectorXr grad_sol_last = gi_history.back();
                xi_history.push_back(q_sol);
                gi_history.push_back(grad_sol);
                si_history.push_back(q_sol - q_sol_last);
                yi_history.push_back(grad_sol - grad_sol_last);
                if (bfgs_size == bfgs_history_size + 1) {
                    xi_history.pop_front();
                    gi_history.pop_front();
                    si_history.pop_front();
                    yi_history.pop_front();
                }
                VectorXr q = grad_sol;
                std::deque<real> rhoi_history, alphai_history;
                for (auto sit = si_history.crbegin(), yit = yi_history.crbegin(); sit != si_history.crend(); ++sit, ++yit) {
                    const VectorXr& yi = *yit;
                    const VectorXr& si = *sit;
                    const real rhoi = 1 / yi.dot(si);
                    const real alphai = rhoi * si.dot(q);
                    rhoi_history.push_front(rhoi);
                    alphai_history.push_front(alphai);
                    q -= alphai * yi;
                }
                // H0k = PdLhsSolve(I);
                VectorXr z = PdLhsSolve(method, q, additional_dirichlet, use_acc, use_sparse);
                auto sit = si_history.cbegin(), yit = yi_history.cbegin();
                auto rhoit = rhoi_history.cbegin(), alphait = alphai_history.cbegin();
                for (; sit != si_history.cend(); ++sit, ++yit, ++rhoit, ++alphait) {
                    const real rhoi = *rhoit;
                    const real alphai = *alphait;
                    const VectorXr& si = *sit;
                    const VectorXr& yi = *yit;
                    const real betai = rhoi * yi.dot(z);
                    z += si * (alphai - betai);
                }
                quasi_newton_direction = z;
            } 
            if (quasi_newton_direction.dot(grad_sol) <= 0)
                quasi_newton_direction = grad_sol.array() * selected.array();
            // Line search --- keep in mind that grad/newton_direction points to the direction that *increases* the objective. 
            real step_size = 1;
            VectorXr q_sol_next = q_sol - step_size * quasi_newton_direction; 
            ShapeTargetComputeAuxiliaryDeformationGradient(q_sol_next);
            real energy_next = ShapeTargetingEnergy(q_sol_next, act); 
            real obj_next = eval_obj(q_sol_next, energy_next);
            const real gamma = ToReal(1e-4);
            bool ls_success = false;    
            for (int j = 0; j < max_ls_iter; ++j) {
                // Directional gradient: obj(q_sol - step_size * newton_direction)
                //                     = obj_sol - step_size * newton_direction.dot(grad_sol)
                const real obj_cond = obj_sol - gamma * step_size * grad_sol.dot(quasi_newton_direction);
                const bool descend_condition = !std::isnan(obj_next) && obj_next < obj_cond + std::numeric_limits<real>::epsilon();
                if (descend_condition) {
                    ls_success = true; 
                    break;
                }
                step_size /= 2;
                q_sol_next = q_sol - step_size * quasi_newton_direction;
                ShapeTargetComputeAuxiliaryDeformationGradient(q_sol_next); 
                energy_next = ShapeTargetingEnergy(q_sol_next, act); 
                obj_next = eval_obj(q_sol_next, energy_next);
            }
            if (!ls_success) {                
                PrintWarning("Line search fails after " + std::to_string(max_ls_iter) + " trials.");
            }
            // update 
            q_sol = q_sol_next;  
            energy_sol = energy_next; 
            obj_sol = obj_next;
        } else {
            // we will skip non-accelerated version. If this is necessary afeter bfgs failed testing, we can add it later.
        } 
        force_sol = ShapeTargetingForce(q_sol, act); 
        grad_sol = -1 * force_sol.array() * selected.array(); 
        // check convergence
        const real abs_error = grad_sol.norm();
        const real rhs_norm = VectorXr(selected.array() * (rhs).array()).norm(); 
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            success = true; 
            return q_sol;
        } 
    }
    CheckError(success, "PD method fails to converge.");
    return VectorXr::Zero(dofs_);
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
    int s_vec_dim = 6;
    S.resize(element_num * sample_num * s_vec_dim);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q_eigen, i);       
        F_auxiliary_[i].resize(sample_num);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            F_auxiliary_[i][j].Initialize(F); // Extract U, V, sig, R, S
            // S is a 6x1 vector
            // transform a symmetric 3x3 matrix to a 6x1 vector
            const Eigen::Matrix<real, vertex_dim, vertex_dim>& S_mat = F_auxiliary_[i][j].S(); 
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 0] = S_mat(0, 0);
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 1] = S_mat(0, 1);
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 2] = S_mat(0, 2);
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 3] = S_mat(1, 1);
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 4] = S_mat(1, 2);
            S[i * sample_num * s_vec_dim + j * s_vec_dim + 5] = S_mat(2, 2);
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

    real total_energy = 0; 
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    std::vector<real> element_energy(element_num, 0);

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        // const auto deformed = ScatterToElement(q, i); Not used because we enforce precomputed F_auxiliary_
        for (int j = 0; j < sample_num; ++j) {
            const auto F = F_auxiliary_[i][j].F();
            const auto R = F_auxiliary_[i][j].R();
            // A is 1x6 per sample point, assemble it into a symmetric 3x3 matrix
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat;
            A_mat(0, 0) = act[i * sample_num * 6 + j * 6 + 0];
            A_mat(0, 1) = act[i * sample_num * 6 + j * 6 + 1];
            A_mat(0, 2) = act[i * sample_num * 6 + j * 6 + 2];
            A_mat(1, 0) = act[i * sample_num * 6 + j * 6 + 1];
            A_mat(1, 1) = act[i * sample_num * 6 + j * 6 + 3];
            A_mat(1, 2) = act[i * sample_num * 6 + j * 6 + 4];
            A_mat(2, 0) = act[i * sample_num * 6 + j * 6 + 2];
            A_mat(2, 1) = act[i * sample_num * 6 + j * 6 + 4];
            A_mat(2, 2) = act[i * sample_num * 6 + j * 6 + 5];
            element_energy[i] += EnergyDensity(F, R, A_mat) * element_volume_ / sample_num;
        }
    }
    for (const auto& e : element_energy) total_energy += e;
    return total_energy;
}
// ShapeTarget Force
template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ShapeTargetingForce(const VectorXr& q, const VectorXr& act) const{
    auto StressTensor = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& F, 
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& R,
                            const Eigen::Matrix<real, vertex_dim, vertex_dim>& A) -> Eigen::Matrix<real, vertex_dim, vertex_dim> {
        return shape_target_stiffness_ * (F - R * A);
    };

    VectorXr total_force = VectorXr::Zero(q.size());
    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    std::vector<Eigen::Matrix<real, vertex_dim, 1>> f_ints(element_num * element_dim,
        Eigen::Matrix<real, vertex_dim, 1>::Zero());
    
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        // const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = F_auxiliary_[i][j].F();
            const auto R = F_auxiliary_[i][j].R();
            // A is 1x6 per sample point, assemble it into a symmetric 3x3 matrix
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat;
            A_mat(0, 0) = act[i * sample_num * 6 + j * 6 + 0];
            A_mat(0, 1) = act[i * sample_num * 6 + j * 6 + 1];
            A_mat(0, 2) = act[i * sample_num * 6 + j * 6 + 2];
            A_mat(1, 0) = act[i * sample_num * 6 + j * 6 + 1];
            A_mat(1, 1) = act[i * sample_num * 6 + j * 6 + 3];
            A_mat(1, 2) = act[i * sample_num * 6 + j * 6 + 4];
            A_mat(2, 0) = act[i * sample_num * 6 + j * 6 + 2];
            A_mat(2, 1) = act[i * sample_num * 6 + j * 6 + 4];
            A_mat(2, 2) = act[i * sample_num * 6 + j * 6 + 5];
            
            Eigen::Matrix<real, vertex_dim, vertex_dim> P = StressTensor(F, R, A_mat);
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> f_kd =
                -Flatten(P).transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k) {
                f_ints[i * element_dim + k] += Eigen::Matrix<real, vertex_dim, 1>(f_kd.segment(k * vertex_dim, vertex_dim));
            }
        }
    }
    for (int i = 0; i < element_num; ++i) {
        const auto vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                total_force(vi(j) * vertex_dim + k) += f_ints[i * element_dim + j](k);
            }
        }
    }
    return total_force;
}





template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

