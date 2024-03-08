#include "pd_energy/deformation_gradient_auxiliary_data.h"

template<int dim>
void DeformationGradientAuxiliaryData<dim>::Initialize(const Eigen::Matrix<real, dim, dim>& F) {
    F_ = F;
    const Eigen::JacobiSVD<Eigen::Matrix<real, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    sig_ = svd.singularValues();
    R_ = U_ * V_.transpose();
    S_ = V_ * sig_.asDiagonal() * V_.transpose();
}

// for shape target only
template<int dim>
void DeformationGradientAuxiliaryData<dim>::Initialize(const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& A) {
    F_ = F;
    A_ = A;
    const Eigen::JacobiSVD<Eigen::Matrix<real, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    sig_ = svd.singularValues();
    R_ = U_ * V_.transpose();
    S_ = V_ * sig_.asDiagonal() * V_.transpose();

    // shape target
    Fst_ = F*A;
    const Eigen::JacobiSVD<Eigen::Matrix<real, dim, dim>> svd_st(Fst_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Ust_ = svd_st.matrixU();
    Vst_ = svd_st.matrixV();
    sigst_ = svd_st.singularValues();
    Rst_ = Ust_ * Vst_.transpose();
    Sst_ = Vst_ * sigst_.asDiagonal() * Vst_.transpose();
}

template class DeformationGradientAuxiliaryData<2>;
template class DeformationGradientAuxiliaryData<3>;