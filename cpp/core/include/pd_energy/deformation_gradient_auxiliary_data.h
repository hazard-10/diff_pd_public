#ifndef PD_ENERGY_DEFORMATION_GRADIENT_AUXILIARY_DATA_H
#define PD_ENERGY_DEFORMATION_GRADIENT_AUXILIARY_DATA_H

#include "common/config.h"

template<int dim>
class DeformationGradientAuxiliaryData {
public:
    DeformationGradientAuxiliaryData() {}
    ~DeformationGradientAuxiliaryData() {}

    void Initialize(const Eigen::Matrix<real, dim, dim>& F);
    void Initialize(const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& A);
    const Eigen::Matrix<real, dim, dim>& F() const { return F_; }
    const Eigen::Matrix<real, dim, dim>& U() const { return U_; }
    const Eigen::Matrix<real, dim, dim>& V() const { return V_; }
    const Eigen::Matrix<real, dim, 1>& sig() const { return sig_; }
    const Eigen::Matrix<real, dim, dim>& R() const { return R_; }
    const Eigen::Matrix<real, dim, dim>& S() const { return S_; }

    // add shape target act
    const Eigen::Matrix<real, dim, dim>& A()   const {  assert(use_shape_target_); return A_; }
    const Eigen::Matrix<real, dim, dim>& Fst() const {  assert(use_shape_target_); return Fst_; }
    const Eigen::Matrix<real, dim, dim>& Rst() const {  assert(use_shape_target_); return Rst_; }
    const Eigen::Matrix<real, dim, dim>& Sst() const {  assert(use_shape_target_); return Sst_; }
    const Eigen::Matrix<real, dim, dim>& Ust() const {  assert(use_shape_target_); return Ust_; }
    const Eigen::Matrix<real, dim, 1>& sigst() const {  assert(use_shape_target_); return sigst_; }
    const Eigen::Matrix<real, dim, dim>& Vst() const {  assert(use_shape_target_); return Vst_; }

private:
    // F = U * sig * V.transpose()
    // F = R * S.
    Eigen::Matrix<real, dim, dim> F_;
    Eigen::Matrix<real, dim, dim> U_, V_;
    Eigen::Matrix<real, dim, 1> sig_;
    Eigen::Matrix<real, dim, dim> R_;
    Eigen::Matrix<real, dim, dim> S_;

    Eigen::Matrix<real, dim, dim> A_; // shape target act
    Eigen::Matrix<real, dim, dim> Fst_; // shape target FA
    Eigen::Matrix<real, dim, dim> Rst_; // shape target R*
    Eigen::Matrix<real, dim, dim> Sst_; // shape target S*
    Eigen::Matrix<real, dim, dim> Ust_; // shape target U*
    Eigen::Matrix<real, dim, 1> sigst_; // shape target sig*
    Eigen::Matrix<real, dim, dim> Vst_; // shape target V*

    bool use_shape_target_ = false;
};

#endif