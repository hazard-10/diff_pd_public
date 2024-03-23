#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "solver/deformable_preconditioner.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetGradientCheck(const VectorXr& q_next, const VectorXr& act,
        std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& dA) const {
    std::map<int, real> augmented_dirichlet = dirichlet_;
    const real eps = 1e-4;
    int a_vec_dim = 6;
    VectorXr eps_vec = VectorXr::Ones(dofs_) * eps;
    Eigen::Matrix<real, vertex_dim, vertex_dim> eps_mat = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity() * eps; 
    const int element_num = mesh_.NumOfElements();
    const real w_ = shape_target_stiffness_ * element_volume_ / element_dim;
    auto q_plus = q_next + eps_vec;
    auto q_minus = q_next - eps_vec;
    auto relativeDiff = [&](const VectorXr& a, const VectorXr& b) {
        return (a - b).norm();
    }; 
    auto dE_dq = [&](const VectorXr& q, const VectorXr& act) {
        VectorXr dE = VectorXr::Zero(q.size());
        for (int i = 0; i < element_num; i++) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            Eigen::Matrix<real, vertex_dim * element_dim, 1> dE_dq_local_sum; dE_dq_local_sum.setZero();
            for (int j = 0; j < element_dim; ++j) {
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[i][j].dF_dxkd_flattened();
                DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j];
                Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F.A();

                // experiments result TLDR
                // for axb = a * b
                // it is equivalent to construct 9x9 A_expand a_exp_1(i * vertex_dim + 0, j * vertex_dim + 0) = a(i, j);
                // times a flattend b with a_flatten(i * vertex_dim + j) = a(i, j);
                // (if you use the ordinary flatten of Eigen::Map, you will get fucked big time)
                // or expand b to 9x9 with b(0,0), b(3,3), b(6,6) = b(0,0), with a_flatten the same way
                // THE RETURNED VECTOR WILL BE ROW MAJOR sample of a*b
                if (false){ 
                    Eigen::Matrix<real, vertex_dim, vertex_dim> a;
                    a << 0, 1, 2, 3, 4, 5, 6, 7, 8;
                    Eigen::Matrix<real, vertex_dim, vertex_dim> b;
                    b << 9, 10, 11, 12, 13, 14, 15, 16, 17;
                    std::cout << "a" << std::endl << a << std::endl << "b" << std::endl << b << std::endl;
                    std::cout << a * b << std::endl;
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> a_exp_1; a_exp_1.setZero();
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> b_exp_1; b_exp_1.setZero();
                    for(int i = 0; i < vertex_dim; ++i){
                        for(int j = 0; j < vertex_dim; ++j){
                            a_exp_1(i * vertex_dim + 0, j * vertex_dim + 0) = a(i, j);
                            a_exp_1(i * vertex_dim + 1, j * vertex_dim + 1) = a(i, j);
                            a_exp_1(i * vertex_dim + 2, j * vertex_dim + 2) = a(i, j);  } }
                    VectorXr b_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                    VectorXr b_flatten_2 = Eigen::Map<VectorXr>(b.data(), vertex_dim * vertex_dim);
                    for(int i = 0; i < vertex_dim; ++i){
                        for(int j = 0; j < vertex_dim; ++j){
                            b_flatten(i * vertex_dim + j) = b(i, j); }
                    }
                    VectorXr ans1 = a_exp_1 * b_flatten;
                    VectorXr ans1_2 = a_exp_1 * b_flatten_2;
                    std::cout << "ans1: "<< std::endl << ans1 << std::endl;
                    std::cout << "ans1_2: "<< std::endl << ans1_2 << std::endl;
                    for(int i = 0; i < vertex_dim; ++i){
                        for(int j = 0; j < vertex_dim; ++j){
                            b_exp_1(i, j) = b(j, i);
                            b_exp_1(i + vertex_dim, j + vertex_dim) = b(j, i);
                            b_exp_1(i + 2 * vertex_dim, j + 2 * vertex_dim) = b(j, i); }}
                    VectorXr a_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                    for(int i = 0; i < vertex_dim; ++i){
                        for(int j = 0; j < vertex_dim; ++j){
                            a_flatten(i * vertex_dim + j) = a(i, j); }
                    }
                    VectorXr ans2 = b_exp_1 * a_flatten;
                    std::cout <<"ans2: " << std::endl << ans2 << std::endl;
                    Eigen::Matrix<real, vertex_dim, vertex_dim> axb = a * b;
                    VectorXr axb_flatten = Eigen::Map<VectorXr>(axb.data(), vertex_dim * vertex_dim);
                    std::cout << "axb_flatten = " << std::endl << axb_flatten << std::endl;
                    for (int j = 0; j < vertex_dim * vertex_dim; ++j) {
                        std::cout << "axb_flatten(j) = " << axb_flatten(j) << std::endl;
                        std::cout << "axb(j) = " << axb(j) << std::endl;
                    }
                }  
                
                // Expansion of A and Rst
                // Originally, we think Rst * A is A_expand * R, which is correct
                // but since everything is col wise flattened to vectors, we need to take the transpose of RstA
                // Therefore, switching the expansion and flatten order into Rst.T expansion and A.T flatten
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> Rst_T_expand; Rst_T_expand.setZero();
                CheckError(vertex_dim == 3, "Only 3d is supported in shape targeting now");
                for(int k = 0; k < vertex_dim; ++k){
                    for(int l = 0; l < vertex_dim; ++l){
                        A_expand(k, l) = A_mat(l, k);
                        A_expand(k + vertex_dim, l + vertex_dim) = A_mat(l, k);
                        A_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = A_mat(l, k);
                    }
                }
                for(int k = 0; k < vertex_dim; ++k){
                    for(int l = 0; l < vertex_dim; ++l){
                        Rst_T_expand(k, l) = F.Rst().transpose()(l, k);
                        Rst_T_expand(k + vertex_dim, l + vertex_dim) = F.Rst().transpose()(l, k);
                        Rst_T_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = F.Rst().transpose()(l, k);
                    }
                }
                // row wise flatten Rst and A
                Eigen::Matrix<real, vertex_dim, vertex_dim> A = F.A();
                Eigen::Matrix<real, vertex_dim, vertex_dim> Rst = F.Rst();
                VectorXr A_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                VectorXr rst_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                for(int k = 0; k < vertex_dim; ++k){
                    rst_flatten.segment(k * vertex_dim, vertex_dim) = Rst.row(k);
                    A_flatten.segment(k * vertex_dim, vertex_dim) = A.row(k); // A.T flatten, but A is symmetric so it is the same
                }  
                // compute A_expand_r_flatten
                VectorXr A_expand_r_flatten = A_expand * rst_flatten;
                VectorXr Rst_T_expand_A_flatten = Rst_T_expand * A_flatten;
                // compare with Rst * A_mat 
                Eigen::Matrix<real, vertex_dim, vertex_dim> RA = Rst * A_mat;
                VectorXr RA_flatten = Eigen::Map<VectorXr>(RA.data(), vertex_dim * vertex_dim); // col wise
                for(int k = 0; k < vertex_dim * vertex_dim; ++k) { 
                    CheckError( abs(RA_flatten(k) - Rst_T_expand_A_flatten(k) ) <= 1e-10, "RA_flatten(k) == rstt_expand_a_flatten(k)");
                        // CheckError( abs(RA_flatten(k) - A_expand_r_flatten(k) ) <= 1e-10, "RA_flatten(k) == A_expand_r_flatten(k)");
                        // Ar equal to row wise flatten of Rst * A_mat
                }
                // compute Gc_q
                const auto q_flatten = ScatterToElementFlattened(q, i);
                VectorXr Gc_q = Gc * q_flatten; // column wise 
                // compare with F.F()
                Eigen::Matrix<real, vertex_dim, vertex_dim> deformG = F.F(); // checked to be row wise
                VectorXr F_flatten = Eigen::Map<VectorXr>(deformG.data(), vertex_dim * vertex_dim); // col wise
                for(int k = 0; k < vertex_dim * vertex_dim; ++k){ 
                        CheckError( abs(F_flatten(k ) - Gc_q(k)) <= 1e-10, "deformG(k, l) == Gc_q(k * vertex_dim + l)"); 
                }
                // Then compare with stress
                VectorXr local_dE = Gc_q - Rst_T_expand * A_flatten;  // 9x24 * 24x1 - 9x9 * 9x1
                Eigen::Matrix<real, vertex_dim, vertex_dim> stress = F.F() - Rst * A_mat; // 3x3 - 3x3 * 3x3
                VectorXr stressFlatten = Eigen::Map<VectorXr>(stress.data(), vertex_dim * vertex_dim); // col wise
                for(int k = 0; k < vertex_dim * vertex_dim; ++k){ 
                        CheckError( abs(stressFlatten(k) - local_dE(k)) <= 1e-10, "stressFlatten(k) == local_dE(k)");
                        // local_dE equal to row wise flatten of F.F() - Rst * A_mat 
                }
                const auto local_dE_dq = w_ * finite_element_samples_[i][j].dF_dxkd_flattened().transpose() * local_dE; // 24x9 * 9x1                 
                
                dE_dq_local_sum = dE_dq_local_sum + local_dE_dq; 
                // for reasons that ShapeTargetingForce uses -Flatten(P).transpose() * Gc, the differentiation is different from
                // force. But I will leave it here for now.
            }

            for( int j = 0; j < element_dim; ++j){
                dE.segment(vertex_dim * vi(j), vertex_dim) += dE_dq_local_sum.segment(j * vertex_dim, vertex_dim);
            }
        }
        return dE;                
    };
    
    auto Compare_prefactor = [&](const VectorXr q, const VectorXr& act){
        // previous dE_dq is computed as w * Gc * (Gc * q_flatten - A_expand * Rst_flatten)
        // this function use the prefactor w * Gc * Gc^T (or A_star). so A_star * q_flatten - w * Gc * (A_expand * Rst_flatten)
        VectorXr A_star_q = PdLhsMatrixOp(q, augmented_dirichlet); // 3 * vertex_num
        VectorXr pref_lhs_F = VectorXr::Zero(q.size());
        VectorXr pref_lhs_Gcq = VectorXr::Zero(q.size());
        for(int i=0 ; i < element_num; ++i){
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            Eigen::Matrix<real, vertex_dim * element_dim, 1> local_Gc_F_sum; local_Gc_F_sum.setZero();
            for( int j =0; j<element_dim; j++){
                DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j]; 
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[i][j].dF_dxkd_flattened();
                VectorXr F_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                VectorXr q_flatten = ScatterToElementFlattened(q, i);
                for(int k = 0; k < vertex_dim; ++k){
                    F_flatten.segment(k * vertex_dim, vertex_dim) = F.F().col(k);
                }
                VectorXr Gc_q = Gc * q_flatten; // column wise 
                for (int k = 0; k < vertex_dim * vertex_dim; ++k) {
                    CheckError( abs(Gc_q(k)-F_flatten(k)) <= 1e-10, "Gc_q(k) == F_flatten(k)");
                }
                local_Gc_F_sum += w_ * Gc.transpose() * F_flatten; // 24x9 * 9x1
            }
            for( int j = 0; j < element_dim; ++j){
                pref_lhs_F.segment(vertex_dim * vi(j), vertex_dim) += local_Gc_F_sum.segment(j * vertex_dim, vertex_dim);
            }
        }
        auto pref_diff = (A_star_q - pref_lhs_F).norm();
        auto pref_norm = A_star_q.norm();
        auto pref_lhs_F_norm = pref_lhs_F.norm();
        std::cout << "pref_norm = " << pref_norm << ", pref_lhs_F_norm = " << pref_lhs_F_norm << std::endl;
        std::cout << "pref_diff = " << pref_diff << std::endl;
        CheckError(pref_diff < 1e-10, "pref_diff < 1e-10");
        // This shows that PdLhsMatrixOp works with column wise flatten
    };
    auto rotationGradiant = [&](const DeformationGradientAuxiliaryData<vertex_dim>& F, bool use_st = false){
            // following code compute with SVD of St version in aux data
            Eigen::Matrix<real, vertex_dim, 1> sig;
            Eigen::Matrix<real, vertex_dim, vertex_dim> U;
            Eigen::Matrix<real, vertex_dim, vertex_dim> V;
            if (use_st){
                sig = F.sigst();
                U = F.Ust();
                V = F.Vst();
            } else {
                sig = F.sig();
                U = F.U();
                V = F.V();
            }
                

            real lambda0 = 2 / (sig(0) + sig(1));
            real lambda1 = 2 / (sig(1) + sig(2));
            real lambda2 = 2 / (sig(0) + sig(2));

            Eigen::Matrix<real, vertex_dim, vertex_dim> q_0 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished();
            Eigen::Matrix<real, vertex_dim, vertex_dim> q_1 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 0, 0, 0, 1, 0, -1, 0).finished();
            Eigen::Matrix<real, vertex_dim, vertex_dim> q_2 = (Eigen::Matrix<real, vertex_dim, vertex_dim>() << 0, 0, 1, 0, 0, 0, -1, 0, 0).finished();
            Eigen::Matrix<real, vertex_dim, vertex_dim> Q0 = (1/sqrt(2)) * U * q_0 * V.transpose();
            Eigen::Matrix<real, vertex_dim, vertex_dim> Q1 = (1/sqrt(2)) * U * q_1 * V.transpose();
            Eigen::Matrix<real, vertex_dim, vertex_dim> Q2 = (1/sqrt(2)) * U * q_2 * V.transpose();

            Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ0 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q0.data(), vertex_dim*vertex_dim, 1);
            Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ1 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q1.data(), vertex_dim*vertex_dim, 1);
            Eigen::Matrix<real, vertex_dim*vertex_dim, 1> vecQ2 = Eigen::Map<Eigen::Matrix<real, vertex_dim*vertex_dim, 1>>(Q2.data(), vertex_dim*vertex_dim, 1);
            
            Eigen::Matrix<real, vertex_dim*vertex_dim, vertex_dim*vertex_dim> dR_dF = (lambda0 * vecQ0 * vecQ0.transpose() + 
                                                lambda1 * vecQ1 * vecQ1.transpose() + 
                                                lambda2 * vecQ2 * vecQ2.transpose());
            return dR_dF;
        }; 

    // first test first order gradient
    bool check_firstOrder_gradient = false;
    if(check_firstOrder_gradient){
        // first order gradient check. Alread passed
        bool check_dE_dq = false;
        if (check_dE_dq) {  
            auto dE_dq_rhs = [&](const VectorXr& q, const VectorXr& act) { // w_ * Gc.T * RA
                VectorXr dE_rhs = VectorXr::Zero(q.size());
                for (int i = 0; i < element_num; i++) {
                    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
                    Eigen::Matrix<real, vertex_dim * element_dim, 1> dE_dq_local_sum; dE_dq_local_sum.setZero();
                    for (int j = 0; j < element_dim; ++j) {
                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[i][j].dF_dxkd_flattened();
                        DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j];
                        Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F.A(); 
                        
                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> Rst_T_expand; Rst_T_expand.setZero();
                        CheckError(vertex_dim == 3, "Only 3d is supported in shape targeting now");
                        for(int k = 0; k < vertex_dim; ++k){
                            for(int l = 0; l < vertex_dim; ++l){
                                A_expand(k, l) = A_mat(l, k);
                                A_expand(k + vertex_dim, l + vertex_dim) = A_mat(l, k);
                                A_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = A_mat(l, k);
                            }
                        }
                        for(int k = 0; k < vertex_dim; ++k){
                            for(int l = 0; l < vertex_dim; ++l){
                                Rst_T_expand(k, l) = F.Rst().transpose()(l, k);
                                Rst_T_expand(k + vertex_dim, l + vertex_dim) = F.Rst().transpose()(l, k);
                                Rst_T_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = F.Rst().transpose()(l, k);
                            }
                        }
                        // row wise flatten Rst and A
                        Eigen::Matrix<real, vertex_dim, vertex_dim> A = F.A();
                        Eigen::Matrix<real, vertex_dim, vertex_dim> Rst = F.Rst();
                        VectorXr A_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                        VectorXr rst_flatten = VectorXr::Zero(vertex_dim * vertex_dim);
                        for(int k = 0; k < vertex_dim; ++k){
                            rst_flatten.segment(k * vertex_dim, vertex_dim) = Rst.row(k);
                            A_flatten.segment(k * vertex_dim, vertex_dim) = A.row(k); // A.T flatten, but A is symmetric so it is the same
                        }  
                        // compute A_expand_r_flatten
                        VectorXr A_expand_r_flatten = A_expand * rst_flatten;
                        VectorXr Rst_T_expand_A_flatten = Rst_T_expand * A_flatten;
                        // compare with Rst * A_mat 
                        Eigen::Matrix<real, vertex_dim, vertex_dim> RA = Rst * A_mat;
                        VectorXr RA_flatten = Eigen::Map<VectorXr>(RA.data(), vertex_dim * vertex_dim); // col wise
                        for(int k = 0; k < vertex_dim * vertex_dim; ++k) { 
                            CheckError( abs(RA_flatten(k) - Rst_T_expand_A_flatten(k) ) <= 1e-10, "RA_flatten(k) == rstt_expand_a_flatten(k)");
                                // CheckError( abs(RA_flatten(k) - A_expand_r_flatten(k) ) <= 1e-10, "RA_flatten(k) == A_expand_r_flatten(k)");
                                // Ar equal to row wise flatten of Rst * A_mat
                        } 
                        // Then compare with stress
                        VectorXr local_dE =  Rst_T_expand * A_flatten;  // 9x1
                        const auto local_dE_dq = w_ * Gc.transpose() * local_dE; // 24x9 * 9x1                 
                        
                        dE_dq_local_sum = dE_dq_local_sum + local_dE_dq; 
                        // for reasons that ShapeTargetingForce uses -Flatten(P).transpose() * Gc, the differentiation is different from
                        // force. But I will leave it here for now.
                    }

                    for( int j = 0; j < element_dim; ++j){
                        dE_rhs.segment(vertex_dim * vi(j), vertex_dim) += dE_dq_local_sum.segment(j * vertex_dim, vertex_dim);
                    }
                }
                return dE_rhs;                
            };
    
            int element_id = 300; 
            int sample_id = 5; // 0-7
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_id);
            int vertex_id = vi(sample_id);
            Eigen::Matrix<real, vertex_dim, 1> delta_xyz = Eigen::Matrix<real, vertex_dim, 1>::Ones() * eps;
            std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> hess_at_q; 
            VectorXr q_pertubated_plus = q_next;
            q_pertubated_plus.segment(vertex_dim * vertex_id, vertex_dim) += delta_xyz;
            VectorXr q_pertubated_minus = q_next;
            q_pertubated_minus.segment(vertex_dim * vertex_id, vertex_dim) -= delta_xyz;
            ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act);
            auto force_q_act = ShapeTargetingForce(q_next, act);
            auto dE_dq_current = dE_dq(q_next, act);
            // compare with A - dA, prepare for hessian check
            VectorXr dE_dq_current_lhs = PdLhsMatrixOp(q_next, augmented_dirichlet);
            VectorXr dE_dq_current_rhs = dE_dq_rhs(q_next, act);
            VectorXr dE_dq_current_op = dE_dq_current_lhs - dE_dq_current_rhs;
            ShapeTargetComputeAuxiliaryDeformationGradient(q_pertubated_plus, act);
            auto force_q_plus_act = ShapeTargetingForce(q_pertubated_plus, act);
            auto energy_q_plus = ShapeTargetingEnergy(q_pertubated_plus, act);
            auto dE_dq_plus = dE_dq(q_pertubated_plus, act); 
            ShapeTargetComputeAuxiliaryDeformationGradient(q_pertubated_minus, act);
            auto force_q_minus_act = ShapeTargetingForce(q_pertubated_minus, act);
            auto energy_q_minus = ShapeTargetingEnergy(q_pertubated_minus, act);
            auto dE_dq_minus = dE_dq(q_pertubated_minus, act);  
            // first compare with shape targeting force
            std::cout << "diff w op = " << force_q_act.norm() - dE_dq_current_op.norm() << std::endl;
            CheckError(force_q_act.norm() - dE_dq_current.norm() < 1e-10, "force_q_act.norm() - dE_dq_current.norm() < 1e-10");            
            CheckError(force_q_plus_act.norm() - dE_dq_plus.norm() < 1e-10, "force_q_plus_act.norm() - dE_dq_plus.norm() < 1e-10");            
            CheckError(force_q_minus_act.norm() - dE_dq_minus.norm() < 1e-10, "force_q_minus_act.norm() - dE_dq_minus.norm() < 1e-10");
            CheckError(dE_dq_current_op.norm() - dE_dq_current.norm() < 1e-10, "dE_dq_current_op.norm() - dE_dq_current.norm() < 1e-10");
            // next check numerical gradient
            real energy_diff = energy_q_plus - energy_q_minus;
            real energy_from_force = force_q_act.dot(q_pertubated_plus - q_pertubated_minus);
            std::cout << "energy_diff = " << energy_diff << ", energy_from_force = " << energy_from_force << ", diff = " << energy_diff - energy_from_force << std::endl;
        } 
        // check dR_dF
        // first pick a random element id and sample id
        bool check_dR_dF = false;
        if(check_dR_dF){
            int element_start = 300;
            int element_range = 1;
            for( int element_id = element_start; element_id < element_start + element_range; element_id++){
                // int element_id = 300; // 0-2500 
                int sample_id = 5; // 0-7
                // get vi 
                const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_id);
                int vertex_id = vi(sample_id);
                // get current deformation gradient
                auto F_auxi = F_auxiliary_[element_id][sample_id]; // has been initialized with current q_next
                Eigen::Matrix<real, vertex_dim, 1> delta_xyz = Eigen::Matrix<real, vertex_dim, 1>::Ones() * eps;
                VectorXr q_pertubated_plus = q_next;
                q_pertubated_plus.segment(vertex_dim * vertex_id, vertex_dim) += delta_xyz;
                Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_plus = ScatterToElement(q_pertubated_plus, element_id);
                VectorXr q_pertubated_minus = q_next;
                q_pertubated_minus.segment(vertex_dim * vertex_id, vertex_dim) -= delta_xyz;
                Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_minus = ScatterToElement(q_pertubated_minus, element_id);
                Eigen::Matrix<real, vertex_dim, vertex_dim> f_default = F_auxi.F();
                Eigen::Matrix<real, vertex_dim, vertex_dim> f_plus = DeformationGradient(element_id, q_scatter_plus, sample_id);
                Eigen::Matrix<real, vertex_dim, vertex_dim> f_minus = DeformationGradient(element_id, q_scatter_minus, sample_id);
                Eigen::Matrix<real, vertex_dim, vertex_dim> f_diff = f_plus - f_minus;
                VectorXr f_diff_flatten = Eigen::Map<VectorXr>(f_diff.data(), vertex_dim * vertex_dim); // col wise

                F_auxi.Initialize(f_default);
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF_default = dRFromdF(F_auxi.F(), F_auxi.R(), F_auxi.S());
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF_rotationGradiant = rotationGradiant(F_auxi, false);
                Eigen::Matrix<real, vertex_dim * vertex_dim, 1> estiamted_dR_default = dR_dF_default * f_diff_flatten;
                Eigen::Matrix<real, vertex_dim * vertex_dim, 1> estiamted_dR_rotationGradiant = dR_dF_rotationGradiant * f_diff_flatten;
                F_auxi.Initialize(f_plus);
                Eigen::Matrix<real, vertex_dim, vertex_dim> R_plus = F_auxi.R();
                F_auxi.Initialize(f_minus);
                Eigen::Matrix<real, vertex_dim, vertex_dim> R_minus = F_auxi.R();
                Eigen::Matrix<real, vertex_dim, vertex_dim> R_diff = R_plus - R_minus;
                VectorXr R_diff_flatten = Eigen::Map<VectorXr>(R_diff.data(), vertex_dim * vertex_dim); // col wise
                bool print_verbose = false;
                if(print_verbose){
                    std::cout << "Error of dR_dF: " << estiamted_dR_default.norm() - R_diff_flatten.norm() 
                                << ", with estiamted_dR_default.norm() = " << estiamted_dR_default.norm() 
                                << ", R_diff_flatten.norm() = " << R_diff_flatten.norm()
                                << std::endl;
                    std::cout << "dR_dF non zero check: " << dR_dF_default.norm() << std::endl;
                    std::cout << "Error of dR_dF_rotationGradiant: " << estiamted_dR_rotationGradiant.norm() - R_diff_flatten.norm() 
                                << ", with estiamted_dR_rotationGradiant.norm() = " << estiamted_dR_rotationGradiant.norm()
                                << ", R_diff_flatten.norm() = " << R_diff_flatten.norm()
                                << std::endl;
                    std::cout << "dR_dF_rotationGradiant non zero check: " << dR_dF_rotationGradiant.norm() << std::endl;
                }
                CheckError(estiamted_dR_default.norm() - R_diff_flatten.norm() < 1e-8, "estiamted_dR_default.norm() - R_diff_flatten.norm() < 1e-8");
                CheckError(estiamted_dR_rotationGradiant.norm() - R_diff_flatten.norm() < 1e-8, "estiamted_dR_rotationGradiant.norm() - R_diff_flatten.norm() < 1e-8");
                CheckError(estiamted_dR_default.norm() - estiamted_dR_rotationGradiant.norm() < 1e-8, "estiamted_dR_default.norm() == estiamted_dR_rotationGradiant.norm()");
            }   
        }
        // check dR_dq as dR_dF * dF_dq. Pretty much equivalent to dR_dF * Gc
        bool check_dR_dq_through_dF_dq = false;
        if(check_dR_dq_through_dF_dq){
            int element_id = element_num - 1; 
            int sample_id = 5; // 0-7
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_id);
            int vertex_id = vi(sample_id);
            // get current deformation gradient
            auto F_auxi = F_auxiliary_[element_id][sample_id]; // has been initialized with current q_next
            Eigen::Matrix<real, vertex_dim, 1> delta_xyz = Eigen::Matrix<real, vertex_dim, 1>::Ones() * eps;
            VectorXr q_pertubated_plus = q_next;
            q_pertubated_plus.segment(vertex_dim * vertex_id, vertex_dim) += delta_xyz;
            Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_plus = ScatterToElement(q_pertubated_plus, element_id);
            Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_plus = ScatterToElementFlattened(q_pertubated_plus, element_id);
            VectorXr q_pertubated_minus = q_next;
            q_pertubated_minus.segment(vertex_dim * vertex_id, vertex_dim) -= delta_xyz;
            Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_minus = ScatterToElement(q_pertubated_minus, element_id);
            Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_minus = ScatterToElementFlattened(q_pertubated_minus, element_id);
            Eigen::Matrix<real, vertex_dim, vertex_dim> f_default = F_auxi.F();
            Eigen::Matrix<real, vertex_dim, vertex_dim> f_plus = DeformationGradient(element_id, q_scatter_plus, sample_id);
            Eigen::Matrix<real, vertex_dim, vertex_dim> f_minus = DeformationGradient(element_id, q_scatter_minus, sample_id);
            Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_diff = q_flatten_plus - q_flatten_minus;
            
            F_auxi.Initialize(f_default);
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[element_id][sample_id].dF_dxkd_flattened();
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF_default = dRFromdF(F_auxi.F(), F_auxi.R(), F_auxi.S());
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF_rotationGradiant = rotationGradiant(F_auxi, false);
            Eigen::Matrix<real, vertex_dim * vertex_dim, 1> estiamted_dR_dq = dR_dF_default * Gc * q_flatten_diff; // 9x9 * 9x24 * 24x1
            Eigen::Matrix<real, vertex_dim * vertex_dim, 1> estiamted_dR_dq_rotationGradiant = dR_dF_rotationGradiant * Gc * q_flatten_diff;
            F_auxi.Initialize(f_plus);
            Eigen::Matrix<real, vertex_dim, vertex_dim> R_plus = F_auxi.R();
            F_auxi.Initialize(f_minus);
            Eigen::Matrix<real, vertex_dim, vertex_dim> R_minus = F_auxi.R();
            Eigen::Matrix<real, vertex_dim, vertex_dim> R_diff = R_plus - R_minus;
            VectorXr R_diff_flatten = Eigen::Map<VectorXr>(R_diff.data(), vertex_dim * vertex_dim); // col wise

            std::cout << "Error of dR_dq: " << estiamted_dR_dq.norm() - R_diff_flatten.norm() 
                        << ", with estiamted_dR_dq.norm() = " << estiamted_dR_dq.norm() 
                        << ", R_diff_flatten.norm() = " << R_diff_flatten.norm()
                        << std::endl;
            std::cout << "dR_dq non zero check: " << dR_dF_default.norm() << std::endl;
            std::cout << "Error of dR_dq_rotationGradiant: " << estiamted_dR_dq_rotationGradiant.norm() - R_diff_flatten.norm() 
                        << ", with estiamted_dR_dq_rotationGradiant.norm() = " << estiamted_dR_dq_rotationGradiant.norm()
                        << ", R_diff_flatten.norm() = " << R_diff_flatten.norm()
                        << std::endl;
            std::cout << "dR_dq_rotationGradiant non zero check: " << dR_dF_rotationGradiant.norm() << std::endl;
        }
        // check d(Rst * A) / dq
        bool check_dRstA_dq = true;
        if(check_dRstA_dq){
            int controller = 12;
            for(int i=0; i< controller; i++){
                int six_controller = i / 2;
                int remainder = i % 2;
                int two_controller = remainder % 2;
                // main loop
                int element_start = 0;
                int element_range = element_num;
                real total_error = 0;
                for( int element_id = element_start; element_id < element_start + element_range; element_id++){
                    int sample_id = 5; // 0-7
                    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_id);
                    int vertex_id = vi(sample_id);
                    // get current deformation gradient
                    auto F_auxi = F_auxiliary_[element_id][sample_id]; // has been initialized with current q_next
                    Eigen::Matrix<real, vertex_dim, 1> delta_xyz = Eigen::Matrix<real, vertex_dim, 1>::Ones() * eps;
                    VectorXr q_pertubated_plus = q_next;
                    q_pertubated_plus.segment(vertex_dim * vertex_id, vertex_dim) += delta_xyz;
                    Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_plus = ScatterToElement(q_pertubated_plus, element_id);
                    Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_plus = ScatterToElementFlattened(q_pertubated_plus, element_id);
                    VectorXr q_pertubated_minus = q_next;
                    q_pertubated_minus.segment(vertex_dim * vertex_id, vertex_dim) -= delta_xyz;
                    Eigen::Matrix<real, vertex_dim, element_dim> q_scatter_minus = ScatterToElement(q_pertubated_minus, element_id);
                    Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_minus = ScatterToElementFlattened(q_pertubated_minus, element_id);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Fst_default = F_auxi.Fst();
                    Eigen::Matrix<real, vertex_dim, vertex_dim> f_default = F_auxi.F();
                    Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F_auxi.A(); 
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
                    for(int k = 0; k < vertex_dim; ++k){
                        for(int l = 0; l < vertex_dim; ++l){
                            if(two_controller == 0){
                                A_expand(k, l) = A_mat(l, k);
                                A_expand(k + vertex_dim, l + vertex_dim) = A_mat(l, k);
                                A_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = A_mat(l, k);
                            } else {
                                A_expand( vertex_dim * k + 0, vertex_dim * l + 0) = A_mat(l, k);    
                                A_expand( vertex_dim * k + 1, vertex_dim * l + 1) = A_mat(l, k);    
                                A_expand( vertex_dim * k + 2, vertex_dim * l + 2) = A_mat(l, k);    
                            }
                        }
                    }
                    Eigen::Matrix<real, vertex_dim, vertex_dim> f_plus = DeformationGradient(element_id, q_scatter_plus, sample_id);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> f_minus = DeformationGradient(element_id, q_scatter_minus, sample_id);
                    Eigen::Matrix<real, vertex_dim * element_dim, 1> q_flatten_diff = q_flatten_plus - q_flatten_minus;

                    // F_auxi as f_default and A
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[element_id][sample_id].dF_dxkd_flattened();
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dRst_dFst_default = dRFromdF(F_auxi.Fst(), F_auxi.Rst(), F_auxi.Sst());
                    Eigen::Matrix<real, vertex_dim * vertex_dim, 1> estiamted_dRst_dq =  dRst_dFst_default * A_expand * Gc * q_flatten_diff; // 9x9 * 9x24 * 24x1
                    
                    if(      six_controller == 0){
                        estiamted_dRst_dq = A_expand * dRst_dFst_default * A_expand * Gc * q_flatten_diff;
                    }else if(six_controller == 1){
                        estiamted_dRst_dq = dRst_dFst_default * A_expand * A_expand * Gc * q_flatten_diff;
                    }else if(six_controller == 2){
                        estiamted_dRst_dq = A_expand * A_expand * dRst_dFst_default * Gc * q_flatten_diff;
                    }else if(six_controller == 3){  
                        estiamted_dRst_dq = dRst_dFst_default * A_expand * Gc * q_flatten_diff;
                    }else if(six_controller == 4){
                        estiamted_dRst_dq = A_expand * dRst_dFst_default * Gc * q_flatten_diff;
                    }else if(six_controller == 5){
                        estiamted_dRst_dq = dRst_dFst_default * Gc * q_flatten_diff;
                    }
                    // No seriously, I tried these for dRst_dq:
                    // A_expand with two different order as a few lines above
                    // A_expand * Gc * A_expand, A_expand * Gc, Gc * A_expand, Gc. And this so far is the only comb that give me -e9 level error, 
                    // which is about the same as dR_dF
                    F_auxi.Initialize(f_plus, A_mat);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_plus = F_auxi.Rst();
                    F_auxi.Initialize(f_minus, A_mat);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_minus = F_auxi.Rst();
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_diff = Rst_plus - Rst_minus;
                    VectorXr Rst_diff_flatten = Eigen::Map<VectorXr>(Rst_diff.data(), vertex_dim * vertex_dim); // col wise
                    bool verbose = false;
                    if(verbose){
                        std::cout << "Error of dRst_dq: " << abs(estiamted_dRst_dq.norm() - Rst_diff_flatten.norm()) 
                                    << ", with estiamted_dRst_dq.norm() = " << estiamted_dRst_dq.norm() 
                                    << ", Rst_diff_flatten.norm() = " << Rst_diff_flatten.norm()
                                    << std::endl;
                        std::cout << "dRst_dq non zero check: " << dRst_dFst_default.norm() << std::endl;
                    }
                    total_error += abs(estiamted_dRst_dq.norm() - Rst_diff_flatten.norm());
                }
                std::cout << "six_controller = " << six_controller << ", remainder = " << remainder << ", two_controller = " << two_controller << std::endl;
                std::cout << "Avg error of dRst_dq: " << total_error / element_range << std::endl;
                // six_controller = 0, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 7.15113e-05
                // six_controller = 0, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 4.76686e-05
                // six_controller = 1, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 9.63161e-05
                // six_controller = 1, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 6.15487e-05
                // six_controller = 2, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 5.44799e-05
                // six_controller = 2, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 5.1868e-05
                // six_controller = 3, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 5.89349e-05
                // six_controller = 3, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 5.74394e-09
                // six_controller = 4, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 3.15952e-05
                // six_controller = 4, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 3.00405e-05
                // six_controller = 5, remainder = 0, two_controller = 0
                // Avg error of dRst_dq: 6.06458e-05
                // six_controller = 5, remainder = 1, two_controller = 1
                // Avg error of dRst_dq: 6.06458e-05
            }
        }
        // Compare MatrixOp with w * Gc.T * Gc
        bool check_MatrixOp = false;
        if(check_MatrixOp){
            Compare_prefactor(q_next, act);
        }
    }
    // gradiant check END
    bool check_secondOrder_energy_wrt_q = false;
    if(check_secondOrder_energy_wrt_q){    
        auto HessAtQ = [&](const VectorXr& q, const VectorXr& act, std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& hess_q, int rhs_choice, int transpose_choice, int dR_choice = 0){
            ShapeTargetComputeAuxiliaryDeformationGradient(q, act); // update deformation gradient 
            for (int i = 0; i < element_num; i++) {
                const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
                Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim> hess_local; hess_local.setZero();
                for (int j = 0; j < element_dim; j++) {
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[i][j].dF_dxkd_flattened();
                    DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j];
                    Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F.A();
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF = dRFromdF(F.Fst(), F.Rst(), F.Sst());
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand; A_expand.setZero();
                    for(int k = 0; k < vertex_dim; ++k){
                        for(int l = 0; l < vertex_dim; ++l){
                            if(transpose_choice == 0){ 
                                A_expand(k, l) = A_mat(l, k);
                                A_expand(k + vertex_dim, l + vertex_dim) = A_mat(l, k);
                                A_expand(k + 2 * vertex_dim, l + 2 * vertex_dim) = A_mat(l, k);
                            }else if(transpose_choice == 1){
                                A_expand( vertex_dim * k + 0, vertex_dim * l + 0) = A_mat(l, k);    
                                A_expand( vertex_dim * k + 1, vertex_dim * l + 1) = A_mat(l, k);    
                                A_expand( vertex_dim * k + 2, vertex_dim * l + 2) = A_mat(l, k);   
                            }
                        }
                    }
                    if(dR_choice == 0){
                        
                    }else if(dR_choice == 1){
                        dR_dF = dR_dF.transpose();
                    }


                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> rhs; rhs.setZero();
                    if(rhs_choice == 0){
                        rhs = A_expand * dR_dF * A_expand * Gc;
                    }else if(rhs_choice == 1){
                        rhs = dR_dF * A_expand * A_expand * Gc;
                    }else if(rhs_choice == 2){
                        rhs = A_expand * A_expand * dR_dF * Gc;
                    }else if(rhs_choice == 3){
                        rhs = dR_dF * A_expand * Gc;
                    }else if(rhs_choice == 4){
                        rhs = A_expand * dR_dF * Gc;
                    }else if(rhs_choice == 5){
                        rhs = dR_dF * Gc;
                    }
                    hess_local += w_ * Gc.transpose() * (Gc - rhs); // w * Gc.T *(Gc - A_expand * dR_dF * A_expand * Gc)
                }
                hess_q[i] = hess_local;
            }
            return hess_q;
        };
        auto ApplyHessToQ = [&](const VectorXr& delta_q, std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& hess){
            VectorXr hess_Q = VectorXr::Zero(delta_q.size());
            // std::cout << "hess size is: " << hess.size() << std::endl;
            // std::cout << "hess_Q size is: " << hess_Q.size() << std::endl;
            int total_addition = 0;
            int non_zero_addition = 0;
            int non_zero_element = 0;
            CheckError(hess.size() == element_num, "hess.size() == element_num"); 
            for (int i = 0; i < element_num; i++) {
                const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i); 
                auto delta_q_flatten = ScatterToElementFlattened(delta_q, i); // 24x1
                Eigen::Matrix<real, vertex_dim * element_dim, 1> local_hess_Q = hess[i] * delta_q_flatten; // 24x24 * 24x1
                
                total_addition += 24;
                bool this_element_non_zero = false;
                for (int j = 0; j < element_dim; j++) {
                    for (int k = 0; k < vertex_dim; k++) {
                        if(local_hess_Q(j * vertex_dim + k) != 0){
                            non_zero_addition++;
                            this_element_non_zero = true;
                        }
                    }
                }
                if(this_element_non_zero){
                    non_zero_element++; // when idx is 200, non_zero_element = 2, all entries of local_hess_Q are non zero
                }

                for (int j = 0; j < element_dim; j++) {
                    // hess_Q.segment(vertex_dim * vi(j), vertex_dim) += local_hess_Q.segment(j * vertex_dim, vertex_dim);
                    // check indice within bound
                    if(vertex_dim * vi(j) + vertex_dim > hess_Q.size()){
                        // std::cout << "vertex_dim * vi(j) + vertex_dim = " << vertex_dim * vi(j) + vertex_dim 
                        //             << ", while hess_Q.size() = " << hess_Q.size() << std::endl;
                    }
                    // CheckError(vertex_dim * vi(j) + vertex_dim <= hess_Q.size(), "vertex_dim * vi(j) + vertex_dim <= hess_Q.size()");
                    // CheckError(j * vertex_dim + vertex_dim <= local_hess_Q.size(), "j * vertex_dim + vertex_dim <= local_hess_Q.size()");
                    hess_Q.segment(vertex_dim * vi(j), vertex_dim) += local_hess_Q.segment(j * vertex_dim, vertex_dim);
                }
            }
            // std::cout << "total_addition = " << total_addition << ", non_zero_addition = " << non_zero_addition << ", non zero element = " << non_zero_element << std::endl;
            return hess_Q;
        };  

        // 1. pick one vertex 
        // 2. Compute its current hessian (8 adjacent 24x24 square matrix), and current force (3x1)
        // 3. pertubate it by delta_xyz compute the force_1 (a new 3x1), then pertubate it by -delta_xyz compute the forc2 (another new 3x1)
        // 4. Use 2 * delta_xyz and multiply default hessian, add up the relevant entries from the 3x8 matrix to get the estimated force
        // 5. Compare the estimated force with the difference of force_1 and force_2

        int controller = 24;
        for(int i = 2 ; i < 3; ++i){
            // divide into 6, 2, 2 options
            int six_choice = i / 4;
            int remainder = i % 4;
            int first_two_choice = remainder / 2;
            int last_two_choice = remainder % 2;
            
            // main loop
            int vertex_start = 800;
            int vertex_range = 20;
            int non_zero_entry_count_diff = 0;
            real total_error = 0;
            real total_relative_error = 0;
            for(int vertex_id = vertex_start; vertex_id < vertex_start + vertex_range ; ++vertex_id){
                Eigen::Matrix<real, vertex_dim, 1> delta_xyz = Eigen::Matrix<real, vertex_dim, 1>::Ones() * eps;
                std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> hess_at_q;
                hess_at_q.resize(element_num);
                VectorXr q_pertubated_plus = q_next;
                q_pertubated_plus.segment(vertex_dim * vertex_id, vertex_dim) += delta_xyz;
                VectorXr q_pertubated_minus = q_next;
                q_pertubated_minus.segment(vertex_dim * vertex_id, vertex_dim) -= delta_xyz;
                // compute hessian at q and store to hess_at_q
                HessAtQ(q_next, act, hess_at_q, six_choice, first_two_choice, last_two_choice);
                // use default approach and store to dA
                std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> dA;
                ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act);
                SetupShapeTargetingLocalStepDifferential(q_next, act, dA);

                ShapeTargetComputeAuxiliaryDeformationGradient(q_pertubated_plus, act);
                auto force_q_plus = ShapeTargetingForce(q_pertubated_plus, act);
                ShapeTargetComputeAuxiliaryDeformationGradient(q_pertubated_minus, act);
                auto force_q_minus = ShapeTargetingForce(q_pertubated_minus, act);
                auto force_diff = force_q_plus - force_q_minus;
                auto q_delta = q_pertubated_plus - q_pertubated_minus;
                // multiply hess_at_q with 2 * delta_xyz
                auto estimated_force = ApplyHessToQ(q_delta, hess_at_q);
                auto estimated_force_2 = PdLhsMatrixOp(q_delta, augmented_dirichlet) - ApplyShapeTargetingLocalStepDifferential(q_delta, act, dA, q_delta);
                std::cout << "diff w backward() = " << (estimated_force - estimated_force_2).norm() << std::endl;

                // get the nonzero entries
                std::vector<real> non_zero_entries_of_force_diff;
                std::vector<real> non_zero_entries_of_estimated_force;
                int count_force_non_zero = 0;
                int count_estimated_force_non_zero = 0;
                for(int i = 0; i < dofs_ / 3; ++i){
                    if(force_diff(3 * i) != 0 || force_diff(3 * i + 1) != 0 || force_diff(3 * i + 2) != 0){
                        non_zero_entries_of_force_diff.push_back(force_diff(3 * i));
                        non_zero_entries_of_force_diff.push_back(force_diff(3 * i + 1));
                        non_zero_entries_of_force_diff.push_back(force_diff(3 * i + 2));
                        count_force_non_zero++;
                    } 
                    if(estimated_force(3 * i) != 0 || estimated_force(3 * i + 1) != 0 || estimated_force(3 * i + 2) != 0){
                        non_zero_entries_of_estimated_force.push_back(estimated_force(3 * i));
                        non_zero_entries_of_estimated_force.push_back(estimated_force(3 * i + 1));
                        non_zero_entries_of_estimated_force.push_back(estimated_force(3 * i + 2));
                        count_estimated_force_non_zero++;
                    }
                        
                }
                // std::cout << "count_force_non_zero = " << count_force_non_zero << ", count_estimated_force_non_zero = " << count_estimated_force_non_zero << std::endl;
                // std::cout << "force_diff = " << force_diff.norm() << ", estimated_force = " << estimated_force.norm() << std::endl;
                total_error += abs(force_diff.norm() - estimated_force.norm());
                total_relative_error += abs(force_diff.norm() - estimated_force.norm()) / force_diff.norm();
                if (count_force_non_zero != count_estimated_force_non_zero){
                    non_zero_entry_count_diff++;
                }
                // std::cout << "vertex_id = " << vertex_id << ", running avg error = " << total_error / (vertex_id + 1) << ", running avg relative error = " << total_relative_error / (vertex_id + 1) << std::endl;
            }
            // std::cout << "i = " << i << ", six_choice = " << six_choice << ", first_two_choice = " << first_two_choice << ", last_two_choice = " << last_two_choice << std::endl;
            // std::cout << "avg error = " << total_error / vertex_range << std::endl;

            // i = 0, six_choice = 0, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.45117e-07
            // i = 1, six_choice = 0, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.45117e-07
            // i = 2, six_choice = 0, first_two_choice = 1, last_two_choice = 0
            // avg error = 7.2019e-13
            // i = 3, six_choice = 0, first_two_choice = 1, last_two_choice = 1
            // avg error = 7.2019e-13
            // i = 4, six_choice = 1, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.28733e-07
            // i = 5, six_choice = 1, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.28733e-07
            // i = 6, six_choice = 1, first_two_choice = 1, last_two_choice = 0
            // avg error = 1.50742e-08
            // i = 7, six_choice = 1, first_two_choice = 1, last_two_choice = 1
            // avg error = 1.50742e-08
            // i = 8, six_choice = 2, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.62119e-07
            // i = 9, six_choice = 2, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.62119e-07
            // i = 10, six_choice = 2, first_two_choice = 1, last_two_choice = 0
            // avg error = 1.47231e-08
            // i = 11, six_choice = 2, first_two_choice = 1, last_two_choice = 1
            // avg error = 1.47231e-08
            // i = 12, six_choice = 3, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.09178e-07
            // i = 13, six_choice = 3, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.09178e-07
            // i = 14, six_choice = 3, first_two_choice = 1, last_two_choice = 0
            // avg error = 7.08529e-08
            // i = 15, six_choice = 3, first_two_choice = 1, last_two_choice = 1
            // avg error = 7.08529e-08
            // i = 16, six_choice = 4, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.23798e-07
            // i = 17, six_choice = 4, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.23798e-07
            // i = 18, six_choice = 4, first_two_choice = 1, last_two_choice = 0
            // avg error = 6.30666e-08
            // i = 19, six_choice = 4, first_two_choice = 1, last_two_choice = 1
            // avg error = 6.30666e-08
            // i = 20, six_choice = 5, first_two_choice = 0, last_two_choice = 0
            // avg error = 1.32326e-07
            // i = 21, six_choice = 5, first_two_choice = 0, last_two_choice = 1
            // avg error = 1.32326e-07
            // i = 22, six_choice = 5, first_two_choice = 1, last_two_choice = 0
            // avg error = 1.32326e-07
            // i = 23, six_choice = 5, first_two_choice = 1, last_two_choice = 1
            // avg error = 1.32326e-07
        }
    }

    bool check_secondOrder_energy_wrt_act = true;
    if(check_secondOrder_energy_wrt_act){ 

        auto act_mat_to_vec = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& act){
            VectorXr act_vec = VectorXr::Zero(6);
            act_vec[0] = act(0, 0);
            act_vec[1] = act(0, 1);
            act_vec[2] = act(0, 2);
            act_vec[3] = act(1, 1);
            act_vec[4] = act(1, 2);
            act_vec[5] = act(2, 2);
            return act_vec;
        };
        auto act_vec_to_mat = [&](const VectorXr& act_vec){
            Eigen::Matrix<real, vertex_dim, vertex_dim> act = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            act(0, 0) = act_vec[0];
            act(0, 1) = act_vec[1];
            act(0, 2) = act_vec[2];
            act(1, 0) = act_vec[1];
            act(1, 1) = act_vec[3];
            act(1, 2) = act_vec[4];
            act(2, 0) = act_vec[2];
            act(2, 1) = act_vec[4];
            act(2, 2) = act_vec[5];
            return act;
        };
        auto expand_1 = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& mat){
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> expand; expand.setZero();
            for(int i = 0; i < vertex_dim; ++i){
                for(int j = 0; j < vertex_dim; ++j){
                    expand(i, j) = mat(i, j);
                    expand(i + vertex_dim, j + vertex_dim) = mat(i, j);
                    expand(i + 2 * vertex_dim, j + 2 * vertex_dim) = mat(i, j);
                }
            }
            return expand;
        };
        auto expand_2 = [&](const Eigen::Matrix<real, vertex_dim, vertex_dim>& mat){
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> expand; expand.setZero();
            for(int i = 0; i < vertex_dim; ++i){
                for(int j = 0; j < vertex_dim; ++j){
                    expand(vertex_dim * i + 0, vertex_dim * j + 0) = mat(i, j);
                    expand(vertex_dim * i + 1, vertex_dim * j + 1) = mat(i, j);
                    expand(vertex_dim * i + 2, vertex_dim * j + 2) = mat(i, j);
                }
            }
            return expand;
        };

        // Since force = w * Gc.T * (Gc - Rst * A),
        // dF_dact = w * Gc.T * ( - dRst_da * A - Rst * dA), since Gc is independent of act
        // Similar to the way we verify dE_dq, we gradually verify
        // 1. dRst_da
        // 2. d(Rst * A)_da
        // 3. dF_dact

        bool check_dRst_da = false;
        // verify dRst_da == dR_dF * F_expand . Rst is from F * A
        if(check_dRst_da){ 
            // verify if dRst_da * delta_a = Rst_plus - Rst_minus
            int element_id = 200;
            int sample_id = 5;
            Eigen::Matrix<real, vertex_dim, vertex_dim> a_delta_mat; a_delta_mat.setZero();
            a_delta_mat(0, 0) = eps; 
            VectorXr a_delta_mat_flatten = Eigen::Map<VectorXr>(a_delta_mat.data(), vertex_dim * vertex_dim); // col wise 9x1

            ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act);
            Eigen::Matrix<real, vertex_dim, vertex_dim> F_init = F_auxiliary_[element_id][sample_id].F();
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_init = F_auxiliary_[element_id][sample_id].A();
            DeformationGradientAuxiliaryData<vertex_dim> F_auxi;
            
            F_auxi.Initialize(F_init, A_init);
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dRst_dFst_default = dRFromdF(F_auxi.Fst(), F_auxi.Rst(), F_auxi.Sst());
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> F_expand = expand_1(F_init);
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dRst_da = dRst_dFst_default * F_expand; // 9x9 * 9x9
            Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dRst_da_deltaA = dRst_da * a_delta_mat_flatten * 2; // 9x9 * 9x1

            Eigen::Matrix<real, vertex_dim, vertex_dim> A_plus = A_init + a_delta_mat;
            Eigen::Matrix<real, vertex_dim, vertex_dim> A_minus = A_init - a_delta_mat;
            F_auxi.Initialize(F_init, A_plus);
            Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_plus = F_auxi.Rst();
            F_auxi.Initialize(F_init, A_minus);
            Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_minus = F_auxi.Rst();
            VectorXr Rst_plus_flatten = Eigen::Map<VectorXr>(Rst_plus.data(), vertex_dim * vertex_dim); // col wise
            VectorXr Rst_minus_flatten = Eigen::Map<VectorXr>(Rst_minus.data(), vertex_dim * vertex_dim); // col wise
            VectorXr Rst_diff = Rst_plus_flatten - Rst_minus_flatten;

            real error = (dRst_da_deltaA - Rst_diff).norm();
            std::cout << "Error of dRst_da: " << error << ", with dRst_da_deltaA.norm() = " << dRst_da_deltaA.norm() << ", Rst_diff.norm() = " << Rst_diff.norm() << std::endl;
        }   

        bool check_dRst_A_da = true;
        // verify d(Rst * A)_da == A_expand * dR_dF * F_expand  + Rst_expand
        if(check_dRst_A_da){ 
            int controller = 4;
            for(int ctrl = 0; ctrl < controller; ++ctrl){
                int first_ctrl = ctrl % 2;
                int second_ctrl = ctrl / 2;
                real error = 0;

                int element_start = 0;
                int element_range = element_num;
                for(int element_id = element_start; element_id < element_start + element_range; ++element_id){ 
                    int sample_id = 5;
                    Eigen::Matrix<real, vertex_dim, vertex_dim> a_delta_mat; a_delta_mat.setZero();
                    a_delta_mat(0, 0) = eps; 
                    VectorXr a_delta_mat_flatten = Eigen::Map<VectorXr>(a_delta_mat.data(), vertex_dim * vertex_dim); // col wise 9x1

                    ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> F_init = F_auxiliary_[element_id][sample_id].F();
                    Eigen::Matrix<real, vertex_dim, vertex_dim> A_init = F_auxiliary_[element_id][sample_id].A();
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand, F_expand, Rst_expand;
                    F_expand = expand_1(F_init); // verified from last section
                    DeformationGradientAuxiliaryData<vertex_dim> F_auxi;

                    F_auxi.Initialize(F_init, A_init);
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dRst_dFst_default = dRFromdF(F_auxi.Fst(), F_auxi.Rst(), F_auxi.Sst());
                    
                    if(first_ctrl == 0){
                        A_expand = expand_1(A_init);
                    }else if(first_ctrl == 1){
                        A_expand = expand_2(A_init); // correct
                    }
                    if(second_ctrl == 0){
                        Rst_expand = expand_1(F_auxi.Rst());
                    }else if(second_ctrl == 1){
                        Rst_expand = expand_2(F_auxi.Rst()); // correct
                    }
                    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dRst_A_da = A_expand * dRst_dFst_default * F_expand + Rst_expand;
                    Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dRst_A_da_deltaA = dRst_A_da * a_delta_mat_flatten * 2; // 9x9 * 9x1
                    
                    Eigen::Matrix<real, vertex_dim, vertex_dim> A_plus = A_init + a_delta_mat;
                    Eigen::Matrix<real, vertex_dim, vertex_dim> A_minus = A_init - a_delta_mat;
                    F_auxi.Initialize(F_init, A_plus);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_plus_A = F_auxi.Rst() * A_plus;
                    F_auxi.Initialize(F_init, A_minus);
                    Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_minus_A = F_auxi.Rst() * A_minus;
                    VectorXr Rst_plus_A_flatten = Eigen::Map<VectorXr>(Rst_plus_A.data(), vertex_dim * vertex_dim); // col wise
                    VectorXr Rst_minus_A_flatten = Eigen::Map<VectorXr>(Rst_minus_A.data(), vertex_dim * vertex_dim); // col wise
                    VectorXr Rst_A_diff = Rst_plus_A_flatten - Rst_minus_A_flatten;

                    error += (dRst_A_da_deltaA - Rst_A_diff).norm();
                }
                std::cout << "Ctrl1 = " << first_ctrl << ", Ctrl2 = " << second_ctrl << ", avg error = " << error / element_range << std::endl;
                // Ctrl1 = 0, Ctrl2 = 0, avg error = 3.94652e-12
                // Ctrl1 = 1, Ctrl2 = 0, avg error = 9.50241e-16
                // Ctrl1 = 0, Ctrl2 = 1, avg error = 2.15736e-09
                // Ctrl1 = 1, Ctrl2 = 1, avg error = 2.15735e-09
            }
        }

        // verify dF_dact
        bool check_secondOrder = true;
        if(check_secondOrder){
            // To verify d2E_dact. First compute force at a1, and a2, which is a_delta away from act
            // ideally, d2E_dact * 2 * a_delta should be the difference of force_a1 and force_a2
            // basically same as d2E_dq, but with act
            // 2nd order will return a 9x24 d2E_dact. Mutiply it with a 9x1 flattened a_delta to get the estimated force
            // the 24x1 force can be added by to a full length vector 
            auto applyActTo2ndOrderdEdact = [&](const VectorXr& q, const VectorXr& act, const VectorXr& a_delta, 
                                                int c1, int c2, int c3){
                ShapeTargetComputeAuxiliaryDeformationGradient(q, act);
                VectorXr force_delta = VectorXr::Zero(q.size()); force_delta.setZero();
                for (int i = 0; i < element_num; i++) {
                    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
                    Eigen::Matrix<real, vertex_dim * element_dim, 1> second_order_local; second_order_local.setZero(); 
                    for (int j = 0; j < element_dim ; j++){  
                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> Gc = finite_element_samples_[i][j].dF_dxkd_flattened();
                        VectorXr a_delta_local_vec = a_delta.segment( i * element_dim * a_vec_dim + j * a_vec_dim, a_vec_dim); // 6x1
                        Eigen::Matrix<real, vertex_dim, vertex_dim> a_delta_mat = act_vec_to_mat(a_delta_local_vec); // 3x3
                        VectorXr a_delta_local_flatten = Eigen::Map<VectorXr>(a_delta_mat.data(), vertex_dim * vertex_dim); // col wise 9x1
                        
                        DeformationGradientAuxiliaryData<vertex_dim>& F = F_auxiliary_[i][j];
                        Eigen::Matrix<real, vertex_dim, vertex_dim> A_mat = F.A();
                        Eigen::Matrix<real, vertex_dim, vertex_dim> F_mat = F.F();
                        Eigen::Matrix<real, vertex_dim, vertex_dim> Rst_mat = F.Rst();
                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dR_dF = dRFromdF(F.Fst(), F.Rst(), F.Sst());

                        Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> A_expand, Rst_expand, F_expand;
                        if(c1 == 0){
                            Rst_expand = expand_1(Rst_mat); // correct
                        }else if(c1 == 1){
                            Rst_expand = expand_2(Rst_mat);
                        }
                        if(c2 == 0){
                            F_expand = expand_1(F_mat); // correct
                        }else if(c2 == 1){
                            F_expand = expand_2(F_mat);
                        }
                        if(c3 == 0){
                            A_expand = expand_1(A_mat);
                        }else if(c3 == 1){
                            A_expand = expand_2(A_mat); // correct
                        }
                        second_order_local += -1 * w_ * Gc.transpose() * (Rst_expand + A_expand * dR_dF * F_expand) * a_delta_local_flatten; // 24x9 * 9x9 * 9x1
                    }


                    for (int j = 0; j < element_dim; j++) {  
                        force_delta.segment(vertex_dim * vi(j), vertex_dim) += second_order_local.segment(j * vertex_dim, vertex_dim); 
                    }
                    
                }
                return force_delta;
            };

            int controller = 8;
            for(int ctrl = 0; ctrl < controller; ++ctrl){
            int first_ctrl = ctrl % 2;
            int remainder = ctrl / 2;
            int second_ctrl = remainder % 2;
            int third_ctrl = remainder / 2;
        
            int element_start = 0;
            int element_range = 1;
            real total_error = 0;
            for( int element_id = element_start; element_id < element_start + element_range; element_id++){
                int sample_id = 5;
                Eigen::Matrix<real, vertex_dim, vertex_dim> a_delta_mat; a_delta_mat.setZero();
                a_delta_mat(0, 0) = eps;   
                VectorXr a_delta_vec_x6 = act_mat_to_vec(a_delta_mat); // 6x1
                VectorXr act_pertubated_plus = act;
                act_pertubated_plus.segment( element_id * element_dim * a_vec_dim + sample_id * a_vec_dim, a_vec_dim) += a_delta_vec_x6;
                VectorXr act_pertubated_minus = act;
                act_pertubated_minus.segment( element_id * element_dim * a_vec_dim + sample_id * a_vec_dim, a_vec_dim) -= a_delta_vec_x6;
                VectorXr act_delta = act_pertubated_plus - act_pertubated_minus; 

                VectorXr dF_estimated = applyActTo2ndOrderdEdact(q_next, act, act_delta, first_ctrl, second_ctrl, third_ctrl);
                ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act_pertubated_plus);
                auto force_plus = ShapeTargetingForce(q_next, act_pertubated_plus);
                ShapeTargetComputeAuxiliaryDeformationGradient(q_next, act_pertubated_minus);
                auto force_minus = ShapeTargetingForce(q_next, act_pertubated_minus);

                auto force_diff = force_plus - force_minus; 
                total_error += abs(dF_estimated.norm() - force_diff.norm());
                std::cout << "element_id = " << element_id << ", error = " << abs(dF_estimated.norm() - force_diff.norm()) << ", dF_estimated = " << dF_estimated.norm() << ", force_diff = " << force_diff.norm() << std::endl;
            }
            std::cout << "first_ctrl = " << first_ctrl << ", second_ctrl = " << second_ctrl << ", third_ctrl = " << third_ctrl << std::endl;
            std::cout << "avg error of d2E_dact: " << total_error / element_range << std::endl;
        }
        }
    }
} 



template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
