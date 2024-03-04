#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "solver/deformable_preconditioner.h"
#include "Eigen/SparseCholesky"


template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetingBackward(const VectorXr& q, const VectorXr& act, const VectorXr& q_next,
        const VectorXr& dl_dq_next, const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dact, 
        VectorXr& dl_dpd_mat_w, VectorXr& dl_dact_w) const {
        // write 3 12s to dl_dact
}

// Energy specific functions

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ShapeTargetingForceDifferential(const VectorXr& q, const VectorXr& act, SparseMatrixElements& dq, SparseMatrixElements& da,
        SparseMatrixElements& dw) const {

}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
