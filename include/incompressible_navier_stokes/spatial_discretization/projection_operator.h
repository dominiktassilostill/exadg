/*
 * projection_operator.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_

#include "../user_interface/input_parameters.h"

#include "../../operators/linear_operator_base.h"
#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"

#include "operators/elementwise_operator.h"
#include "solvers_and_preconditioners/preconditioner/elementwise_preconditioners.h"
#include "solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h"

using namespace dealii;

namespace IncNS
{
/*
 *  Combined divergence and continuity penalty operator: applies the operation
 *
 *   mass matrix operator + dt * divergence penalty operator + dt * continuity penalty operator .
 *
 *  The divergence and continuity penalty operators can also be applied separately. In detail
 *
 *  Mass matrix operator: ( v_h , u_h )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *
 *  Divergence penalty operator: ( div(v_h) , tau_div * div(u_h) )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 *
 *
 *  Continuity penalty operator: ( v_h , tau_conti * jump(u_h) )_dOmega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *   jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */

/*
 *  Operator data.
 */
struct ProjectionOperatorData
{
  ProjectionOperatorData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      use_divergence_penalty(true),
      use_continuity_penalty(true),
      penalty_factor_div(1.0),
      penalty_factor_conti(1.0),
      which_components(ContinuityPenaltyComponents::Normal),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(1000, 1.e-12, 1.e-1 /*rel_tol TODO*/, 1000))
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // specify which penalty terms to be used
  bool use_divergence_penalty, use_continuity_penalty;

  // scaling factor
  double penalty_factor_div, penalty_factor_conti;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;
};

template<int dim, int degree, typename Number>
class ProjectionOperator : public LinearOperatorBase
{
private:
  typedef ProjectionOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number>     FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, dim, Number> FEEvalFace;

public:
  typedef Number value_type;

  ProjectionOperator(MatrixFree<dim, Number> const & data_in,
                     unsigned int const              dof_index_in,
                     unsigned int const              quad_index_in,
                     ProjectionOperatorData const    operator_data_in)
    : data(data_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      array_conti_penalty_parameter(0),
      array_div_penalty_parameter(0),
      time_step_size(1.0),
      operator_data(operator_data_in),
      scaling_factor_div(1.0),
      scaling_factor_conti(1.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      block_diagonal_preconditioner_is_initialized(false)
  {
    unsigned int n_cells = data.n_cell_batches() + data.n_ghost_cell_batches();

    if(operator_data.use_divergence_penalty)
      array_div_penalty_parameter.resize(n_cells);

    if(operator_data.use_continuity_penalty)
      array_conti_penalty_parameter.resize(n_cells);

    if(operator_data.use_divergence_penalty)
      fe_eval.reset(
        new FEEvalCell(this->get_data(), this->get_dof_index(), this->get_quad_index()));

    if(operator_data.use_continuity_penalty)
    {
      fe_eval_m.reset(
        new FEEvalFace(this->get_data(), true, this->get_dof_index(), this->get_quad_index()));
      fe_eval_p.reset(
        new FEEvalFace(this->get_data(), false, this->get_dof_index(), this->get_quad_index()));
    }
  }

  MatrixFree<dim, Number> const &
  get_data() const
  {
    return data;
  }

  AlignedVector<VectorizedArray<Number>> const &
  get_array_div_penalty_parameter() const
  {
    return array_div_penalty_parameter;
  }

  unsigned int
  get_dof_index() const
  {
    return dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return quad_index;
  }

  /*
   *  Set the time step size.
   */
  void
  set_time_step_size(double const & delta_t)
  {
    time_step_size = delta_t;
  }

  /*
   *  Get the time step size.
   */
  double
  get_time_step_size() const
  {
    return time_step_size;
  }

  void
  calculate_array_penalty_parameter(VectorType const & velocity)
  {
    if(operator_data.use_divergence_penalty)
      calculate_array_div_penalty_parameter(velocity);
    if(operator_data.use_continuity_penalty)
      calculate_array_conti_penalty_parameter(velocity);
  }

  void
  calculate_array_div_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEvalCell fe_eval(data, dof_index, quad_index);

    AlignedVector<scalar> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_cell_batches() + data.n_ghost_cell_batches(); ++cell)
    {
      scalar tau_convective = make_vectorized_array<Number>(0.0);
      scalar tau_viscous    = make_vectorized_array<Number>(operator_data.viscosity);

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(velocity);
        fe_eval.evaluate(true, false);

        scalar volume      = make_vectorized_array<Number>(0.0);
        scalar norm_U_mean = make_vectorized_array<Number>(0.0);
        JxW_values.resize(fe_eval.n_q_points);
        fe_eval.fill_JxW_values(JxW_values);
        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          volume += JxW_values[q];
          norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective =
          norm_U_mean * std::exp(std::log(volume) / (double)dim) / (double)(degree + 1);
      }

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_div_penalty_parameter[cell] =
          operator_data.penalty_factor_div * (tau_convective + tau_viscous);
      }
    }
  }

  void
  calculate_array_conti_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEvalCell fe_eval(data, dof_index, quad_index);

    AlignedVector<scalar> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_cell_batches() + data.n_ghost_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity);
      fe_eval.evaluate(true, false);
      scalar volume      = make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = make_vectorized_array<Number>(0.0);
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
      }
      norm_U_mean /= volume;

      scalar tau_convective = norm_U_mean;
      scalar h              = std::exp(std::log(volume) / (double)dim) / (double)(degree + 1);
      scalar tau_viscous    = make_vectorized_array<Number>(operator_data.viscosity) / h;

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_conti_penalty_parameter[cell] =
          operator_data.penalty_factor_conti * (tau_convective + tau_viscous);
      }
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    if(operator_data.use_divergence_penalty && operator_data.use_continuity_penalty)
      do_apply(dst, src, true);
    else if(operator_data.use_divergence_penalty && !operator_data.use_continuity_penalty)
      do_apply_mass_div_penalty(dst, src, true);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    if(operator_data.use_divergence_penalty && operator_data.use_continuity_penalty)
      do_apply(dst, src, false);
    else if(operator_data.use_divergence_penalty && !operator_data.use_continuity_penalty)
      do_apply_mass_div_penalty(dst, src, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    do_apply_div_penalty(dst, src, true);
  }

  void
  apply_add_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    do_apply_div_penalty(dst, src, false);
  }

  void
  apply_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    do_apply_conti_penalty(dst, src, true);
  }

  void
  apply_add_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    do_apply_conti_penalty(dst, src, false);
  }


  /*
   *  Calculate inverse diagonal which is needed for the Jacobi preconditioner.
   */
  void
  calculate_inverse_diagonal(VectorType & diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Initialize dof vector (required when using the Jacobi preconditioner).
   */
  void
  initialize_dof_vector(VectorType & vector) const
  {
    data.initialize_dof_vector(vector, dof_index);
  }

  /*
   * Block diagonal preconditioner.
   */

  // apply the inverse block diagonal operator (for matrix-based and matrix-free variants)
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    // matrix-free
    if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // Solve block Jacobi problems iteratively using an elementwise solver vectorized
      // over several elements.
      bool const variable_not_needed = false;
      elementwise_solver->solve(dst, src, variable_not_needed);
    }
    else // matrix based
    {
      // Simply apply inverse of block matrices (using the LU factorization that has been computed
      // before).
      data.cell_loop(&This::cell_loop_apply_inverse_block_diagonal, this, dst, src);
    }
  }

  /*
   * Update block diagonal preconditioner: initialize everything related to block diagonal
   * preconditioner when this function is called the first time. Recompute block matrices in case of
   * matrix-based implementation.
   */
  void
  update_block_diagonal_preconditioner() const
  {
    // initialization

    if(!block_diagonal_preconditioner_is_initialized)
    {
      if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
      {
        initialize_block_diagonal_preconditioner_matrix_free();
      }
      else // matrix-based variant
      {
        // Note that the velocity has dim components.
        unsigned int dofs_per_cell = data.get_shape_info().dofs_per_component_on_cell * dim;

        matrices.resize(data.n_macro_cells() * VectorizedArray<Number>::n_array_elements,
                        LAPACKFullMatrix<Number>(dofs_per_cell, dofs_per_cell));
      }

      block_diagonal_preconditioner_is_initialized = true;
    }

    // update

    // For the matrix-free variant there is nothing to do.
    // For the matrix-based variant we have to recompute the block matrices.
    if(!operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // clear matrices
      initialize_block_jacobi_matrices_with_zero(matrices);

      // compute block matrices and add
      this->add_block_diagonal_matrices(matrices);

      calculate_lu_factorization_block_jacobi(matrices);
    }
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1.0) const
  {
    (void)problem_size;

    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    if(operator_data.use_divergence_penalty)
    {
      fe_eval->reinit(cell);

      unsigned int dofs_per_cell = fe_eval->dofs_per_cell;

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval->begin_dof_values()[i] = src[i];

      fe_eval->evaluate(true, true, false);

      do_cell_integral(*fe_eval);

      fe_eval->integrate(true, true);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval->begin_dof_values()[i];
    }

    if(operator_data.use_continuity_penalty)
    {
      // face integrals
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        fe_eval_m->reinit(cell, face);
        fe_eval_p->reinit(cell, face);

        unsigned int dofs_per_cell = fe_eval_m->dofs_per_cell;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_m->begin_dof_values()[i] = src[i];

        // do not need to read dof values for fe_eval_p (already initialized with 0)

        fe_eval_m->evaluate(true, false);

        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        if(bid == numbers::internal_face_boundary_id) // internal face
        {
          do_face_int_integral(*fe_eval_m, *fe_eval_p);
        }
        else // boundary face
        {
          // use same fe_eval so that the result becomes zero (only jumps involved)
          do_face_int_integral(*fe_eval_m, *fe_eval_m);
        }

        fe_eval_m->integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[i] += fe_eval_m->begin_dof_values()[i];
      }
    }
  }

private:
  void
  do_apply_div_penalty(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    data.cell_loop(&This::cell_loop_div_penalty, this, dst, src, zero_dst_vector);
  }

  void
  do_apply_mass_div_penalty(VectorType &       dst,
                            VectorType const & src,
                            bool const         zero_dst_vector) const
  {
    data.cell_loop(&This::cell_loop, this, dst, src, zero_dst_vector);
  }

  void
  do_apply_conti_penalty(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    data.loop(&This::cell_loop_empty,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              zero_dst_vector,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  do_apply(VectorType & dst, VectorType const & src, bool const zero_dst_vector) const
  {
    data.loop(&This::cell_loop,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              zero_dst_vector,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  template<typename FEEval>
  void
  do_cell_integral(FEEval & fe_eval) const
  {
    scalar tau = fe_eval.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value(fe_eval.get_value(q), q);
      fe_eval.submit_divergence(tau * fe_eval.get_divergence(q), q);
    }
  }

  template<typename FEEval>
  void
  do_cell_integral_div_penalty(FEEval & fe_eval) const
  {
    scalar tau = fe_eval.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_divergence(tau * fe_eval.get_divergence(q), q);
    }
  }

  template<typename FEFaceEval>
  void
  do_face_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    scalar tau = 0.5 *
                 (fe_eval.read_cell_data(array_conti_penalty_parameter) +
                  fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vector uM         = fe_eval.get_value(q);
      vector uP         = fe_eval_neighbor.get_value(q);
      vector jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval.submit_value(tau * jump_value, q);
        fe_eval_neighbor.submit_value(-tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = fe_eval.get_normal_vector(q);

        fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
        fe_eval_neighbor.submit_value(-tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_int_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    scalar tau = 0.5 *
                 (fe_eval.read_cell_data(array_conti_penalty_parameter) +
                  fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vector uM = fe_eval.get_value(q);
      vector uP; // set uP to zero
      vector jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = fe_eval.get_normal_vector(q);
        fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_ext_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    scalar tau = 0.5 *
                 (fe_eval.read_cell_data(array_conti_penalty_parameter) +
                  fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
                 scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vector uM; // set uM to zero
      vector uP         = fe_eval_neighbor.get_value(q);
      vector jump_value = uP - uM; // interior - exterior = uP - uM (neighbor!)

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval_neighbor.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        vector normal = fe_eval_neighbor.get_normal_vector(q);
        fe_eval_neighbor.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }


  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, true, true);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_div_penalty(MatrixFree<dim, Number> const & data,
                        VectorType &                    dst,
                        VectorType const &              src,
                        Range const &                   cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, false, true);

      do_cell_integral_div_penalty(fe_eval);

      fe_eval.integrate_scatter(false, true, dst);
    }
  }

  void
  cell_loop_empty(MatrixFree<dim, Number> const & /*data*/,
                  VectorType & /*dst*/,
                  VectorType const & /*src*/,
                  Range const & /*cell_range*/) const
  {
    // do nothing
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, dof_index, quad_index);
    FEEvalFace fe_eval_neighbor(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_integral(fe_eval, fe_eval_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_empty(MatrixFree<dim, Number> const & /*data*/,
                           VectorType & /*dst*/,
                           VectorType const & /*src*/,
                           Range const & /*face_range*/) const
  {
    // do nothing
  }

  /*
   *  This function calculates the diagonal of the projection operator including the mass matrix,
   * divergence penalty and continuity penalty operators. A prerequisite to call this function is
   * that the time step size is set correctly.
   */
  void
  calculate_diagonal(VectorType & diagonal) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    VectorType src_dummy(diagonal);
    data.loop(&This::cell_loop_diagonal,
              &This::face_loop_diagonal,
              &This::boundary_face_loop_diagonal,
              this,
              diagonal,
              src_dummy,
              true /*zero dst vector = true*/,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Calculation of diagonal (cell loop).
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const & /*src*/,
                     Range const & cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (face loop).
   */
  void
  face_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const & /*src*/,
                     Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, dof_index, quad_index);
    FEEvalFace fe_eval_neighbor(data, false, dof_index, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // element-
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;
      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral(fe_eval, fe_eval_neighbor);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);

      // neighbor (element+)
      unsigned int dofs_per_cell_neighbor = fe_eval_neighbor.dofs_per_cell;
      scalar       local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral(fe_eval, fe_eval_neighbor);

        fe_eval_neighbor.integrate(true, false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (boundary face loop).
   */
  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & /*data*/,
                              VectorType & /*dst*/,
                              VectorType const & /*src*/,
                              Range const & /*face_range*/) const
  {
    // do nothing
  }

  void
  initialize_block_diagonal_preconditioner_matrix_free() const
  {
    elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

    if(this->operator_data.preconditioner_block_jacobi == PreconditionerBlockDiagonal::None)
    {
      typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
      elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
    }
    else if(this->operator_data.preconditioner_block_jacobi ==
            PreconditionerBlockDiagonal::InverseMassMatrix)
    {
      typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, degree, Number> INVERSE_MASS;

      elementwise_preconditioner.reset(
        new INVERSE_MASS(this->get_data(), this->get_dof_index(), this->get_quad_index()));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    Elementwise::IterativeSolverData iterative_solver_data;
    iterative_solver_data.solver_type = Elementwise::SolverType::CG;
    iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

    elementwise_solver.reset(new ELEMENTWISE_SOLVER(
      *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
      *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
      iterative_solver_data));
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    scaling_factor_div   = time_step_size;
    scaling_factor_conti = time_step_size;

    VectorType src;

    if(operator_data.use_cell_based_loops)
    {
      data.cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      data.loop(&This::cell_loop_calculate_block_diagonal,
                &This::face_loop_calculate_block_diagonal,
                &This::boundary_face_loop_calculate_block_diagonal,
                this,
                matrices,
                src);
    }
  }


  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, dof_index, quad_index);
    FEEvalFace fe_eval_neighbor(data, false, dof_index, quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral(fe_eval, fe_eval_neighbor);

        fe_eval.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral(fe_eval, fe_eval_neighbor);

        fe_eval_neighbor.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  boundary_face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &,
                                              std::vector<LAPACKFullMatrix<Number>> &,
                                              VectorType const &,
                                              Range const &) const
  {
    // do nothing
  }

  void
  cell_based_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                           std::vector<LAPACKFullMatrix<Number>> & matrices,
                                           VectorType const &,
                                           Range const & cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);
    FEEvalFace fe_eval_m(data, true, dof_index, quad_index);
    FEEvalFace fe_eval_p(data, false, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        fe_eval_m.reinit(cell, face);
        fe_eval_p.reinit(cell, face);
        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          fe_eval_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          fe_eval_m.evaluate(true, false);

          if(bid == numbers::internal_face_boundary_id) // internal face
          {
            do_face_int_integral(fe_eval_m, fe_eval_p);
          }
          else // boundary face
          {
            // use same fe_eval so that the result becomes zero (only jumps involved)
            do_face_int_integral(fe_eval_m, fe_eval_m);
          }

          fe_eval_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   cell_range) const
  {
    FEEvalCell fe_eval(data, dof_index, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(dofs_per_cell);
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell * VectorizedArray<Number>::n_array_elements + v].solve(src_vector, false);

        // write solution to dst-vector
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values(dst);
    }
  }

  MatrixFree<dim, Number> const & data;

  unsigned int const dof_index;
  unsigned int const quad_index;

  AlignedVector<scalar> array_conti_penalty_parameter;
  AlignedVector<scalar> array_div_penalty_parameter;

  double time_step_size;

  ProjectionOperatorData operator_data;

  // Scaling factors for divergence and continuity penalty term:
  // Normally, the scaling factor equals the time step size when applying the combined operator
  // consisting of mass, divergence penalty and continuity penalty operators. In case that the
  // divergence and continuity penalty terms are applied separately (e.g. coupled solution approach,
  // penalty terms added to monolithic system), these scaling factors have to be set to a value
  // of 1.
  mutable double scaling_factor_div, scaling_factor_conti;

  unsigned int n_mpi_processes;

  /*
   * Vector of matrices for block-diagonal preconditioners.
   */
  mutable std::vector<LAPACKFullMatrix<Number>> matrices;

  /*
   * We want to initialize the block diagonal preconditioner (block diagonal matrices or elementwise
   * iterative solvers in case of matrix-free implementation) only once, so we store the status of
   * initialization in a variable.
   */
  mutable bool block_diagonal_preconditioner_is_initialized;


  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef Elementwise::OperatorBase<dim, Number, This>             ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>> PRECONDITIONER_BASE;
  typedef Elementwise::
    IterativeSolver<dim, dim, degree, Number, ELEMENTWISE_OPERATOR, PRECONDITIONER_BASE>
      ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR> elementwise_operator;
  mutable std::shared_ptr<PRECONDITIONER_BASE>  elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>   elementwise_solver;

  /*
   * FEEvaluation objects required for elementwise block Jacobi operations
   */
  std::shared_ptr<FEEvalCell> fe_eval;
  std::shared_ptr<FEEvalFace> fe_eval_m;
  std::shared_ptr<FEEvalFace> fe_eval_p;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATOR_H_ \
        */