#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "solver/deformable_preconditioner.h"
#include "Eigen/SparseCholesky"

// Notation for the code: 
// (the paper used a series of different notations which is annoying, I will declare them here
// to avoid confusion). The format is 
// paper , original code, then this code

// q, pos | q_cur, q_next | q before and after forward pass
// A , the "Left hand side " of PD equation, w * Gc.T * Gc, also the first part of hessian_q
//      original code :pd_lhs from, multiply computed from PdLhsMatrixOp, prefactored in pd_forward setup
//      this code     : stick to pd_lhs, as it is not exposed in this file
// ΔA, the second part of hessian_q, - w * Gc.T * dProjectoin_dq
//      original code : pd_backward_local_muscle_matrices, computed from ApplyProjectiveDynamicsLocalStepDifferential
//      this code     : used dA
// Z, the adjoint variable of dLoss_dq_next * hessian_q.inverse
//      original code : x_sol, optimized with bfgs
//      this code     : stick to Z, same as paper
// Hess_q, the hessian of q, A - ΔA
//      original code : S, implicitly computed with Sx_sol
//      this code     : stick to Hess_q
// Sx_sol, Hess_q * Z, which is just dLoss_dq_next. see eq 14
//      original code : Sx_sol
//      this code     : Hess_q_Z


// SetUp and Apply is used to compute dA * Z
// SetUp construct dA and store in pd_backward_local_muscle_matrices
// Apply compute dA * Z
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupShapeTargetingLocalStepDifferential(const VectorXr& q_cur, const VectorXr& act,
        std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& dA
    ) const {    
    for (const auto& pair : dirichlet_) CheckError(q_cur(pair.first) == pair.second, "Boundary conditions violated.");    
    
    const int sample_num = GetNumOfSamplesInElement();
    const int element_num = mesh_.NumOfElements();
    const real w = shape_target_stiffness_ * element_volume_ / sample_num;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> ones = Eigen::Matrix<real, vertex_dim, vertex_dim>::Ones();
    // From paper Implicit Neural Representation for Physics-driven Actuated Soft Bodies, eq 14
    // dA = w * Gc.T * A * dR_dq. 
    // if use_FA_not_F, then dA = w * Gc.T * act * Hess_rot_AF * act * Gc
    // if Not,          then dA = w * Gc.T * act * Hess_rot_F * Gc

    auto rotationGradiant = [&](const DeformationGradientAuxiliaryData<vertex_dim>& F){
        // following code compute with SVD of St version in aux data
        const Eigen::Matrix<real, vertex_dim, 1>& sig = F.sigst();

        real lambda0 = 2 / (sig(0) + sig(1));
        real lambda1 = 2 / (sig(1) + sig(2));
        real lambda2 = 2 / (sig(0) + sig(2));

        Eigen::Matrix<real, vertex_dim, vertex_dim> q_0 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> q_1 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 0, 0, 0, 1, 0, -1, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> q_2 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 1, 0, 0, 0, -1, 0, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q0 = (1/sqrt(2)) * F.Ust() * q_0 * F.Vst().transpose();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q1 = (1/sqrt(2)) * F.Ust() * q_1 * F.Vst().transpose();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q2 = (1/sqrt(2)) * F.Ust() * q_2 * F.Vst().transpose();

        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ0 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q0.data(), vertex_dim*vertex_dim, 1);
        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ1 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q1.data(), vertex_dim*vertex_dim, 1);
        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ2 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q2.data(), vertex_dim*vertex_dim, 1);
        
        Eigen::Matrix<real, vertex_dim*vertex_dim, vertex_dim*vertex_dim> dR_dF = (lambda0 * vecQ0 * vecQ0.transpose() + 
                                            lambda1 * vecQ1 * vecQ1.transpose() + 
                                            lambda2 * vecQ2 * vecQ2.transpose());
        return dR_dF;
    };

    dA.resize(element_num);
    // #pragma omp parallel for
    for (int i = 0; i < element_num; i++) {
        Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim> w_GT_A_HrAF_A_G; w_GT_A_HrAF_A_G.setZero();
        for (int j = 0; j < sample_num; ++j) {
            DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j];
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F.A();
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
            CheckError(vertex_dim == 3, "Only 3d is supported in shape targeting now");
            A_expand.block(0, 0, vertex_dim, vertex_dim).noalias() = A_mat;
            A_expand.block(3, 3, vertex_dim, vertex_dim).noalias() = A_mat;
            A_expand.block(6, 6, vertex_dim, vertex_dim).noalias() = A_mat;
 
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF = rotationGradiant(F);

            w_GT_A_HrAF_A_G += w * finite_element_samples_[i][j].pd_At() * A_expand * dR_dF * A_expand * finite_element_samples_[i][j].pd_A();             
        }
        dA[i] = w_GT_A_HrAF_A_G;
    }

}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ApplyShapeTargetingLocalStepDifferential(const VectorXr& q_cur, 
        const VectorXr& act,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& dA, const VectorXr& Z) const{    
    // Implements dA * Z
    const int element_num = mesh_.NumOfElements();
    VectorXr dA_Z = VectorXr::Zero(dofs_);
    // Project PdElementEnergy.
    for (int i = 0; i < element_num; ++i) {
        const auto ddeformed = ScatterToElementFlattened(Z, i);
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const Eigen::Matrix<real, vertex_dim * element_dim, 1> da_z_ele = dA[i] * ddeformed;
        for (int k = 0; k < element_dim; ++k)
            dA_Z.segment(vertex_dim * vi(k), vertex_dim) += da_z_ele.segment(k * vertex_dim, vertex_dim);
    }     
    return dA_Z;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetingBackward(const VectorXr& q, const VectorXr& act, const VectorXr& q_next,
    const VectorXr& dl_dq_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dact, 
    VectorXr& dl_dpd_mat_w, VectorXr& dl_dact_w) const {
    // 0.a check params, initialize options input
    CheckShapeTargetParam(options);
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
    int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    const std::string method = "pd_eigen";
    bool use_acc = true;
    bool use_sparse = false;

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupShapeTargetingSolver(options);
    // 0.b initialize the variables to write
    dl_dq = VectorXr::Zero(q.size());
    dl_dact = VectorXr::Zero(act.size());
    dl_dpd_mat_w = VectorXr::Zero(1); // this is derivative of the w term in pd, since we 
                                      // only has one energy, the size is 1
    dl_dact_w = VectorXr::Zero(1); // will delete this later

    // todo：initialize local muscle / element matrices in setupShapeTargetingLocalStepDifferential
    // line 182 - 192 in pd_backward function

    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> dA;
    ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act);
    SetupShapeTargetingLocalStepDifferential(q_next, act, dA);

    std::map<int, real> augmented_dirichlet = dirichlet_;

    const VectorXr dl_dq_next_agg = dl_dq_next;
    VectorXr dl_drhs_intermediate;


    VectorXr Z = VectorXr::Zero(dofs_); 
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : augmented_dirichlet) {
        Z(pair.first) = 0;
        selected(pair.first) = 0;
    }
    VectorXr hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
    VectorXr grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();
    real obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z);


    // prepare for PD backward iteration
    // 1. compute hessian_q_Z = A * Z - ΔA * Z 
    bool success = false; 

    int iter_num = 8;
    // // conventional pd backward 
    // Z = VectorXr::Zero(dofs_);
    // selected = VectorXr::Ones(dofs_);
    // for (const auto& pair : augmented_dirichlet) {
    //     Z(pair.first) = 0;
    //     selected(pair.first) = 0;
    // }
    // hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
    // grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();
    // obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z);
    // for(int i = 0; i < iter_num; ++i){ // diffpd
    //     const VectorXr b = (dl_dq_next_agg + ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z)).array() * selected.array();
    //     // Global step:
    //     Z = (PdLhsSolve(method, b, augmented_dirichlet, use_acc, use_sparse).array() * selected.array());
    //     hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
    //     grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();   
    //     if (verbose_level > 1){ 
    //         std::cout << ", abs_error = " << grad_sol.norm()
    //                 << ", iter " << i
    //                 << std::endl;
    //     }
    // }

    // now do the 2021 paper version    
    Z = VectorXr::Zero(dofs_);
    selected = VectorXr::Ones(dofs_);
    for (const auto& pair : augmented_dirichlet) {
        Z(pair.first) = 0;
        selected(pair.first) = 0;
    }
    hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
    grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();
    obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z); 
    for(int i = 0; i < iter_num; ++i){ // 2021 paper
        hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
        const VectorXr r = dl_dq_next_agg - hessian_q_Z;
        VectorXr dZ = PdLhsSolve(method, r, augmented_dirichlet, use_acc, use_sparse);
        // if (verbose_level > 1) std::cout << "iter " << j << " dZ.norm() = " << dZ.norm() << std::endl;
        Z += dZ;
        grad_sol = r.array() * selected.array();  
        if (verbose_level > 1){ 
            std::cout << ", abs_error = " << grad_sol.norm()
                    << ", iter " << i
                    << std::endl;
        }
    }
    success = true;

    // next is bfgs    
    // Z = VectorXr::Zero(dofs_);
    // selected = VectorXr::Ones(dofs_);
    // for (const auto& pair : augmented_dirichlet) {
    //     Z(pair.first) = 0;
    //     selected(pair.first) = 0;
    // }
    // hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
    // grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();
    // obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z);
    
    // Initialize queues for BFGS.
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history;
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("PD backward iteration: " + std::to_string(i));
        if (false){
            VectorXr quasi_newton_direction = VectorXr::Zero(dofs_);
            const int bfgs_size = static_cast<int>(xi_history.size());
            if (bfgs_size == 0) {
                // Initially, the queue is empty. We use A as our initial guess of Hessian (not the inverse!).
                xi_history.push_back(Z);
                gi_history.push_back(grad_sol);
                quasi_newton_direction = PdLhsSolve(method, grad_sol, augmented_dirichlet, use_acc, use_sparse);
            } else {
                const VectorXr Z_last = xi_history.back();
                const VectorXr grad_last = gi_history.back();
                xi_history.push_back(Z);
                gi_history.push_back(grad_sol);
                si_history.push_back(Z - Z_last);
                yi_history.push_back(grad_sol - grad_last);
                if (bfgs_size == bfgs_history_size + 1) {
                    xi_history.pop_front();
                    gi_history.pop_front();
                    si_history.pop_front();
                    yi_history.pop_front();
                }
                VectorXr bfgs_q = grad_sol;
                std::deque<real> rhoi_history, alphai_history;
                for (auto sit = si_history.crbegin(), yit = yi_history.crbegin(); sit != si_history.crend(); ++sit, ++yit) {
                    const VectorXr& yi = *yit;
                    const VectorXr& si = *sit;
                    const real rhoi = 1 / yi.dot(si);
                    const real alphai = rhoi * si.dot(bfgs_q);
                    rhoi_history.push_front(rhoi);
                    alphai_history.push_front(alphai);
                    bfgs_q -= alphai * yi;
                }
                VectorXr z_ = PdLhsSolve(method, bfgs_q, augmented_dirichlet, use_acc, use_sparse);
                auto sit = si_history.cbegin(), yit = yi_history.cbegin();
                auto rhoit = rhoi_history.cbegin(), alphait = alphai_history.cbegin();
                for (; sit != si_history.cend(); ++sit, ++yit, ++rhoit, ++alphait) {
                    const real rhoi = *rhoit;
                    const real alphai = *alphait;
                    const VectorXr& si = *sit;
                    const VectorXr& yi = *yit;
                    const real betai = rhoi * yi.dot(z_);
                    z_ += si * (alphai - betai);
                }
                quasi_newton_direction = z_;
            }
            quasi_newton_direction = quasi_newton_direction.array() * selected.array();
            if (quasi_newton_direction.dot(grad_sol) < -ToReal(1e-4)) { // TODO: replace 1e-4 with a relative threshold.
                // This implies the (inverse of) Hessian is indefinite, which means the objective to be minimized will
                // become unbounded below. In this case, we choose to switch back to Newton's method.
                success = false;
                PrintWarning("Indefinite Hessian. BFGS is minimizing an unbounded objective.");
                break;
            }
            // Line search.
            real step_size = 1;
            VectorXr Z_next = Z - step_size * quasi_newton_direction;
            VectorXr hessian_q_Z_next = PdLhsMatrixOp(Z_next, augmented_dirichlet) - 
                                ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z_next);
            VectorXr grad_sol_next = (hessian_q_Z_next - dl_dq_next_agg).array() * selected.array();
            real obj_next = 0.5 * Z_next.dot(hessian_q_Z_next) - dl_dq_next_agg.dot(Z_next);
            const real gamma = 1e-4;
            bool ls_success = false;
            for (int j = 0; j < max_ls_iter; ++j) {
                const real obj_cond = obj_sol + gamma * step_size * grad_sol.dot(quasi_newton_direction);
                const bool descend_condition = !std::isnan(obj_next) && obj_next < obj_cond + std::numeric_limits<real>::epsilon();
                if (descend_condition) {
                    ls_success = true;
                    break;
                }
                step_size *= 0.5;
                Z_next = Z - step_size * quasi_newton_direction;
                hessian_q_Z_next = PdLhsMatrixOp(Z_next, augmented_dirichlet) - 
                                ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z_next);
                grad_sol_next = (hessian_q_Z_next - dl_dq_next_agg).array() * selected.array();
                obj_next = 0.5 * Z_next.dot(hessian_q_Z_next) - dl_dq_next_agg.dot(Z_next);
                if (verbose_level > 0) PrintInfo("Line search iteration: " + std::to_string(j));
                if (verbose_level > 1) {
                    std::cout << "step size: " << step_size << std::endl;
                    std::cout << "obj_sol: " << obj_sol << ", "
                        << "obj_cond: " << obj_cond << ", "
                        << "obj_next: " << obj_next << ", "
                        << "obj_cond - obj_sol: " << obj_cond - obj_sol << ", "
                        << "obj_next - obj_sol: " << obj_next - obj_sol << std::endl;
                }
            }
            if (!ls_success) {
                PrintWarning("Line search fails after " + std::to_string(max_ls_iter) + " trials.");
            }
            // update
            Z = Z_next;
            hessian_q_Z = hessian_q_Z_next;
            grad_sol = grad_sol_next;
            obj_sol = obj_next;
        }else{
            break;
            // // conventional pd backward 
            // // Local step:
            // const VectorXr pd_rhs = (dl_dq_next_agg + ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z)).array() * selected.array();
            // // Global step:
            // Z = (PdLhsSolve(method, pd_rhs, augmented_dirichlet, use_acc, use_sparse).array() * selected.array());
            // hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
            // grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();  
            // obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z);

            // // from 2021 paper
            // int iter_num = 10;
            // for (int j = 0; j < iter_num; ++j) {
            //     hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
            //     const VectorXr r = dl_dq_next_agg - hessian_q_Z;
            //     VectorXr dZ = PdLhsSolve(method, r, augmented_dirichlet, use_acc, use_sparse);
            //     // if (verbose_level > 1) std::cout << "iter " << j << " dZ.norm() = " << dZ.norm() << std::endl;
            //     Z += dZ;
            // }            
            // hessian_q_Z = PdLhsMatrixOp(Z, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_next, act, dA, Z);
            // grad_sol = (hessian_q_Z - dl_dq_next_agg).array() * selected.array();
            // obj_sol = 0.5 * Z.dot(hessian_q_Z) - dl_dq_next_agg.dot(Z);
        }
        // check convergence
        const real abs_error = grad_sol.norm();
        const real rhs_norm = dl_dq_next_agg.norm();
        if (verbose_level > 1){ 
            std::cout << "obj_sol = " << obj_sol 
                      << ", abs_error = " << abs_error 
                      << ", rhs_norm = " << rhs_norm
                      << ", rel_tol * rhs_norm + abs_tol = " << rel_tol * rhs_norm + abs_tol 
                      << ", Z size = " << Z.size() << ", hessian_q_Z size = " << hessian_q_Z.size() << ", dl_dq_next_agg size = " << dl_dq_next_agg.size() << std::endl;
        }
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            success = true;
            for (const auto& pair : augmented_dirichlet) Z(pair.first) = dl_dq_next_agg(pair.first);
            break;
        }
    }
    dl_drhs_intermediate = Z;
    if (!success) { 
        PrintError("PD Shape Targeting backward: switching to Cholesky decomposition");
        Eigen::SimplicialLDLT<SparseMatrix> cholesky;
        // const SparseMatrix op = NewtonMatrix(q_next, musc_act, inv_h2m, augmented_dirichlet, use_precomputed_data)
        SparseMatrix hessian_q_og;

        cholesky.compute(hessian_q_og);
        dl_drhs_intermediate = cholesky.solve(dl_dq_next_agg); // hess * z = dl_dq_next_agg
        CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    } 

    // not sure what these lines do
    VectorXr dl_drhs = dl_drhs_intermediate;
    for (const auto& pair: augmented_dirichlet) dl_drhs(pair.first) = dl_dq_next_agg(pair.first);
    VectorXr adjoint = dl_drhs_intermediate;
    for (const auto& pair : augmented_dirichlet) adjoint(pair.first) = 0;
    // const VectorXr dfixed = NewtonMatrixOp(q_next, musc_act, inv_h2m, {}, -adjoint);
    // for (const auto& pair : augmented_dirichlet) dl_drhs(pair.first) += dfixed(pair.first);

    // next step would be imploy the dForce_dAct, use individual matrix
    // to perform from 2021 paper sec4.2  
    // will directly perform in actuationDiff and write to dl_dact
    ShapeTargetingForceDifferential(q_next, act, dl_drhs, dl_dact); // perform Z*dForce_dAct
    // dl_dact is the only output we need
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetingForceDifferential(const VectorXr& q_next, 
            const VectorXr& act, const VectorXr& Z, VectorXr& dl_dact) const {
    // return dForce_dAct
    // final dLoss_dAct = dLoss_dq * Hess_q * dForce_dAct

    // according to 2021 paper, can simply compute dForce_dAct on individual element, then 
    // directly apply to the Z (aka dl_dact from input)
    
    // for each hex element, get the 3x8 matrixs of from vector Z
    // then compute the dForce_dAct, a 24x24 matrix, then compute a 24x1 vector, then add each vertex_dim 
    // contribution.
    // Very similar to ApplyShapeTargetingLocalStepDifferential. Only that we also implement 
    // SetupShapeTargetingLocalStepDifferential for ForceDifferential right here

    const int sample_num = GetNumOfSamplesInElement();
    const int element_num = mesh_.NumOfElements();
    const real w = shape_target_stiffness_ * element_volume_ / sample_num;  
    dl_dact = VectorXr::Zero( 6 * element_num * sample_num); 
    std::vector<Eigen::Matrix<real, vertex_dim, element_dim>> sp_remap(element_num,
        Eigen::Matrix<real, vertex_dim, element_dim>::Zero());  // remap from 3x8 in element space to vertex dim    
    
    auto rotationGradiant = [&](const DeformationGradientAuxiliaryData<vertex_dim>& F){
        // following code compute with SVD of St version in aux data
        const Eigen::Matrix<real, vertex_dim, 1>& sig = F.sigst();

        real lambda0 = 2 / (sig(0) + sig(1));
        real lambda1 = 2 / (sig(1) + sig(2));
        real lambda2 = 2 / (sig(0) + sig(2));

        Eigen::Matrix<real, vertex_dim, vertex_dim> q_0 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> q_1 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 0, 0, 0, 1, 0, -1, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> q_2 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 1, 0, 0, 0, -1, 0, 0).finished();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q0 = (1/sqrt(2)) * F.Ust() * q_0 * F.Vst().transpose();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q1 = (1/sqrt(2)) * F.Ust() * q_1 * F.Vst().transpose();
        Eigen::Matrix<real, vertex_dim, vertex_dim> Q2 = (1/sqrt(2)) * F.Ust() * q_2 * F.Vst().transpose();

        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ0 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q0.data(), vertex_dim*vertex_dim, 1);
        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ1 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q1.data(), vertex_dim*vertex_dim, 1);
        Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ2 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q2.data(), vertex_dim*vertex_dim, 1);
        assert(q_0(0, 1) == vecQ0(1));
        assert(q_1(1, 2) == vecQ1(5));
        assert(q_2(2, 0) == vecQ2(6));
        assert(q_0(2, 1) == vecQ0(7));
        Eigen::Matrix<real, vertex_dim*vertex_dim, vertex_dim*vertex_dim> dR_dF = (lambda0 * vecQ0 * vecQ0.transpose() + 
                                            lambda1 * vecQ1 * vecQ1.transpose() + 
                                            lambda2 * vecQ2 * vecQ2.transpose());
        return dR_dF;
    };

    auto matExpand = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& mat){
        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> expand; expand.setZero();
        for (int i = 0; i < vertex_dim; ++i){
            for (int j = 0; j < vertex_dim; ++j){
                for (int k = 0; k < vertex_dim; ++k){ 
                    expand(i * vertex_dim + k, j * vertex_dim + k) = mat(i, j);
                }
            }   
        }
        return expand;
    };
    
    // #pragma omp parallel for
    for (int i = 0; i < element_num; i++) {
        Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * vertex_dim> dF_dact; dF_dact.setZero();
        for (int j = 0; j < sample_num; ++j) {
            DeformationGradientAuxiliaryData<vertex_dim>& F_auxi_curr = F_auxiliary_[i][j];
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F_auxi_curr.A();
            Eigen::Matrix<real, vertex_dim, vertex_dim> F_mat = F_auxi_curr.F();
            Eigen::Matrix<real, vertex_dim, vertex_dim> R = F_auxi_curr.R();
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
            CheckError(vertex_dim == 3, "Only 3d is supported in shape targeting now");
            A_expand.block(0, 0, vertex_dim, vertex_dim).noalias() = A_mat;
            A_expand.block(3, 3, vertex_dim, vertex_dim).noalias() = A_mat;
            A_expand.block(6, 6, vertex_dim, vertex_dim).noalias() = A_mat;
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> R_expand = matExpand(R);
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> F_expand = matExpand(F_mat);

            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF = rotationGradiant(F_auxi_curr);

            dF_dact += -1 * w * finite_element_samples_[i][j].pd_At() * R_expand;
            // 24x9 = 1 * 1 * 24x9 * 9x9            
            dF_dact += -1 * w * finite_element_samples_[i][j].pd_At() * A_expand * dR_dF * F_expand;
            // 24x9 = 1 * 1 * 24x9 * 9x9 * 9x9
             
            // todo, Z needs to be negated
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> z_deformed = ScatterToElementFlattened(Z, i); // 24x1
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> z_deformed_T = z_deformed.transpose(); // 1x24
            const Eigen::Matrix<real, 1, vertex_dim * vertex_dim> dF_dact_Z = z_deformed_T * dF_dact; // 1x9 = 1x24 * 24x9
            // write the non-symmetric 6 value comes from entry [0, 1, 2, 4, 5, 8] 
            dl_dact[i * sample_num * 6 + j * 6 + 0] += dF_dact_Z(0, 0);
            dl_dact[i * sample_num * 6 + j * 6 + 1] += dF_dact_Z(0, 1);
            dl_dact[i * sample_num * 6 + j * 6 + 2] += dF_dact_Z(0, 2);
            dl_dact[i * sample_num * 6 + j * 6 + 3] += dF_dact_Z(0, 4);
            dl_dact[i * sample_num * 6 + j * 6 + 4] += dF_dact_Z(0, 5);
            dl_dact[i * sample_num * 6 + j * 6 + 5] += dF_dact_Z(0, 8);
            // std::cout << "z_deformed: " << std::endl << z_deformed << std::endl;
            // std::cout << "w: " << std::endl << w << std::endl
            // << "R_expand: " << std::endl << R_expand << std::endl << std::endl
            // << "A_expand: " << std::endl << A_expand << std::endl << std::endl
            // << "dR_dF: " << std::endl << dR_dF << std::endl
            // << "dF_dact: " << std::endl << dF_dact << std::endl
            // << "dF_dact_Z: " << std::endl << dF_dact_Z << std::endl;
        }
    } 
}


// Energy specific functions


template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
