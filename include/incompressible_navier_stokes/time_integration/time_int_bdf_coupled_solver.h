/*
 * time_int_bdf_coupled.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "time_integration/push_back_vectors.h"

namespace IncNS
{
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFCoupled
  : public TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
{
public:
  typedef TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation> Base;

  typedef typename Base::VectorType                           VectorType;
  typedef LinearAlgebra::distributed::BlockVector<value_type> BlockVectorType;

  TimeIntBDFCoupled(std::shared_ptr<NavierStokesOperation> navier_stokes_operation_in,
                    InputParameters<dim> const &           param_in,
                    unsigned int const                     n_refine_time_in,
                    bool const                             use_adaptive_time_stepping)
    : TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>(
        navier_stokes_operation_in,
        param_in,
        n_refine_time_in,
        use_adaptive_time_stepping),
      navier_stokes_operation(navier_stokes_operation_in),
      solution(this->order),
      vec_convective_term(this->order),
      computing_times(2),
      iterations(2),
      N_iter_nonlinear(0.0),
      scaling_factor_continuity(1.0),
      characteristic_element_length(1.0)
  {
  }

  virtual ~TimeIntBDFCoupled()
  {
  }

  virtual void
  analyze_computing_times() const;

  void
  postprocessing_stability_analysis();

private:
  virtual void
  setup_derived();

  virtual void
  initialize_vectors();

  virtual void
  initialize_current_solution();

  virtual void
  initialize_former_solution();

  void
  initialize_vec_convective_term();

  virtual void
  solve_timestep();

  virtual void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  postprocess_velocity();

  virtual void
  postprocessing() const;

  virtual void
  postprocessing_steady_problem() const;

  virtual void
  prepare_vectors_for_next_timestep();

  virtual LinearAlgebra::distributed::Vector<value_type> const &
  get_velocity();

  virtual LinearAlgebra::distributed::Vector<value_type> const &
  get_velocity(unsigned int i /* t_{n-i} */);

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  std::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  std::vector<BlockVectorType> solution;
  BlockVectorType              solution_np;

  BlockVectorType rhs_vector;

  std::vector<VectorType> vec_convective_term;

  // performance analysis: average number of iterations and solver time
  std::vector<value_type>   computing_times;
  std::vector<unsigned int> iterations;
  unsigned int              N_iter_nonlinear;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;

  // temporary vectors needed for pseudo-timestepping algorithm
  VectorType velocity_tmp;
  VectorType pressure_tmp;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_vectors()
{
  // solution
  for(unsigned int i = 0; i < solution.size(); ++i)
    navier_stokes_operation->initialize_block_vector_velocity_pressure(solution[i]);
  navier_stokes_operation->initialize_block_vector_velocity_pressure(solution_np);

  // convective term
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);
  }

  // temporal derivative term: sum_i (alpha_i * u_i)
  navier_stokes_operation->initialize_vector_velocity(this->sum_alphai_ui);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    navier_stokes_operation->initialize_block_vector_velocity_pressure(rhs_vector);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_current_solution()
{
  navier_stokes_operation->prescribe_initial_conditions(solution[0].block(0),
                                                        solution[0].block(1),
                                                        this->get_time());
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    navier_stokes_operation->prescribe_initial_conditions(solution[i].block(0),
                                                          solution[i].block(1),
                                                          this->get_previous_time(i));
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::setup_derived()
{
  // scaling factor continuity equation:
  // Calculate characteristic element length h
  characteristic_element_length = calculate_minimum_vertex_distance(
    navier_stokes_operation->get_dof_handler_u().get_triangulation());

  characteristic_element_length =
    calculate_characteristic_element_length(characteristic_element_length, fe_degree_u);

  // convective term treated explicitly (additive decomposition)
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     this->param.start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],
                                                      solution[i].block(0),
                                                      this->get_previous_time(i));
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
LinearAlgebra::distributed::Vector<value_type> const &
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::get_velocity()
{
  return solution[0].block(0);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
LinearAlgebra::distributed::Vector<value_type> const &
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::get_velocity(unsigned int i)
{
  return solution[i].block(0);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::read_restart_vectors(
  boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for(unsigned int i = 0; i < solution.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), solution[i].block(0).begin());
  }
  for(unsigned int i = 0; i < solution.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), solution[i].block(1).begin());
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  VectorView<value_type> tmp(solution[0].block(0).local_size(), solution[0].block(0).begin());
#pragma GCC diagnostic pop
  oa << tmp;
  for(unsigned int i = 1; i < solution.size(); i++)
  {
    tmp.reinit(solution[i].block(0).local_size(), solution[i].block(0).begin());
    oa << tmp;
  }
  for(unsigned int i = 0; i < solution.size(); i++)
  {
    tmp.reinit(solution[i].block(1).local_size(), solution[i].block(1).begin());
    oa << tmp;
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // update scaling factor of continuity equation
  if(this->param.use_scaling_continuity == true)
  {
    scaling_factor_continuity = this->param.scaling_factor_continuity *
                                characteristic_element_length / this->get_time_step_size();
    navier_stokes_operation->set_scaling_factor_continuity(scaling_factor_continuity);
  }
  else // use_scaling_continuity == false
  {
    scaling_factor_continuity = 1.0;
  }

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(this->extra.get_beta(i), solution[i]);

  // update of turbulence model
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    navier_stokes_operation->update_turbulence_model(solution_np.block(0));

    if(this->get_time_step_number() % this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: " << std::scientific
                  << timer_turbulence.wall_time() << std::endl;
    }
  }

  // calculate auxiliary variable p^{*} = 1/scaling_factor * p
  solution_np.block(1) *= 1.0 / scaling_factor_continuity;

  // Update divegence and continuity penalty operator in case
  // that these terms are added to the monolithic system of equations
  // instead of applying these terms in a postprocessing step.
  if(this->param.add_penalty_terms_to_monolithic_system == true)
  {
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      // extrapolate velocity to time t_n+1 and use this velocity field to
      // caculate the penalty parameter for the divergence and continuity penalty term
      VectorType velocity_extrapolated(solution[0].block(0));
      velocity_extrapolated = 0;
      for(unsigned int i = 0; i < solution.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), solution[i].block(0));

      if(this->param.use_divergence_penalty == true)
      {
        navier_stokes_operation->update_divergence_penalty_operator(velocity_extrapolated);
      }
      if(this->param.use_continuity_penalty == true)
      {
        navier_stokes_operation->update_continuity_penalty_operator(velocity_extrapolated);
      }
    }
  }

  // if the problem to be solved is linear
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // calculate rhs vector for the Stokes problem, i.e., the convective term is neglected in this
    // step
    navier_stokes_operation->rhs_stokes_problem(rhs_vector, this->get_next_time());

    // Add the convective term to the right-hand side of the equations
    // if the convective term is treated explicitly (additive decomposition):
    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(this->param.equation_type == EquationType::NavierStokes &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      navier_stokes_operation->evaluate_convective_term(vec_convective_term[0],
                                                        solution[0].block(0),
                                                        this->get_time());

      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        rhs_vector.block(0).add(-this->extra.get_beta(i), vec_convective_term[i]);
    }

    // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
    // and operator-integration-factor (OIF) splitting
    if(this->param.equation_type == EquationType::NavierStokes &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      this->calculate_sum_alphai_ui_oif_substepping(this->cfl, this->cfl_oif);
    }
    // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
    else
    {
      this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(),
                              solution[0].block(0));
      for(unsigned int i = 1; i < solution.size(); ++i)
      {
        this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(),
                                solution[i].block(0));
      }
    }

    // apply mass matrix to sum_alphai_ui and add to rhs vector
    navier_stokes_operation->apply_mass_matrix_add(rhs_vector.block(0), this->sum_alphai_ui);

    // solve coupled system of equations
    unsigned int linear_iterations = navier_stokes_operation->solve_linear_stokes_problem(
      solution_np, rhs_vector, this->get_scaling_factor_time_derivative_term());

    iterations[0] += linear_iterations;
    computing_times[0] += timer.wall_time();

    // write output
    if(this->get_time_step_number() % this->param.output_solver_info_every_timesteps == 0)
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << "Solve linear Stokes problem:" << std::endl
            << "  Iterations: " << std::setw(6) << std::right << linear_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // a nonlinear system of equations has to be solved
  {
    // calculate Sum_i (alpha_i/dt * u_i)
    this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(),
                            solution[0].block(0));
    for(unsigned int i = 1; i < solution.size(); ++i)
    {
      this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(),
                              solution[i].block(0));
    }

    // Newton solver
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;
    navier_stokes_operation->solve_nonlinear_problem(
      solution_np,
      this->sum_alphai_ui,
      this->get_next_time(),
      this->get_scaling_factor_time_derivative_term(),
      newton_iterations,
      linear_iterations);

    N_iter_nonlinear += newton_iterations;
    iterations[0] += linear_iterations;
    computing_times[0] += timer.wall_time();

    // write output
    if(this->get_time_step_number() % this->param.output_solver_info_every_timesteps == 0)
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      pcout << "Solve nonlinear Navier-Stokes problem:" << std::endl
            << "  Newton iterations: " << std::setw(6) << std::right << newton_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2)
            << std::right
            << ((newton_iterations > 0) ? (double(linear_iterations) / (double)newton_iterations) :
                                          linear_iterations)
            << " (avg)" << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2)
            << std::right << linear_iterations << " (tot)" << std::endl;
    }
  }

  // reconstruct pressure solution p from auxiliary variable p^{*}: p = scaling_factor * p^{*}
  solution_np.block(1) *= scaling_factor_continuity;

  // special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      navier_stokes_operation->shift_pressure(solution_np.block(1), this->get_next_time());
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      set_zero_mean_value(solution_np.block(1));
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      navier_stokes_operation->shift_pressure_mean_value(solution_np.block(1),
                                                         this->get_next_time());
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }

  // If the penalty terms are applied in a postprocessing step
  if(this->param.add_penalty_terms_to_monolithic_system == false)
  {
    // postprocess velocity field using divergence and/or continuity penalty terms
    if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
    {
      timer.restart();

      postprocess_velocity();

      computing_times[1] += timer.wall_time();
    }
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::postprocess_velocity()
{
  Timer timer;
  timer.restart();

  VectorType temp(solution_np.block(0));
  navier_stokes_operation->apply_mass_matrix(temp, solution_np.block(0));

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // caculate the penalty parameter for the divergence and continuity penalty term
  VectorType velocity_extrapolated(solution[0].block(0));
  velocity_extrapolated = 0;
  for(unsigned int i = 0; i < solution.size(); ++i)
    velocity_extrapolated.add(this->extra.get_beta(i), solution[i].block(0));

  // update projection operator
  navier_stokes_operation->update_projection_operator(velocity_extrapolated,
                                                      this->get_time_step_size());

  // solve projection (where also the preconditioner is updated)
  unsigned int iterations_postprocessing =
    navier_stokes_operation->solve_projection(solution_np.block(0), temp);

  iterations[1] += iterations_postprocessing;

  // write output
  if(this->get_time_step_number() % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Postprocessing of velocity field:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations_postprocessing
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::postprocessing() const
{
  navier_stokes_operation->do_postprocessing(solution[0].block(0),
                                             solution[0].block(1),
                                             this->get_time(),
                                             this->get_time_step_number());
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
  postprocessing_steady_problem() const
{
  navier_stokes_operation->do_postprocessing_steady_problem(solution[0].block(0),
                                                            solution[0].block(1));
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
  postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis"));

  AssertThrow(solution[0].block(0).l2_norm() < 1.e-15 && solution[0].block(1).l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  const unsigned int size = solution[0].block(0).local_size();

  LAPACKFullMatrix<value_type> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    solution[0].block(0).local_element(j) = 1.0;

    // solve time step
    solve_timestep();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = solution_np.block(0).local_element(i);
    }

    // reset j-th element to 0
    solution[0].block(0).local_element(j) = 0.0;
  }

  // compute eigenvalues
  propagation_matrix.compute_eigenvalues();

  double norm_max = 0.0;

  std::cout << "List of all eigenvalues:" << std::endl;

  for(unsigned int i = 0; i < size; ++i)
  {
    double norm = std::abs(propagation_matrix.eigenvalue(i));
    if(norm > norm_max)
      norm_max = norm;

    // print eigenvalues
    std::cout << std::scientific << std::setprecision(5) << propagation_matrix.eigenvalue(i)
              << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
  prepare_vectors_for_next_timestep()
{
  push_back(solution);
  solution[0].swap(solution_np);

  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    push_back(vec_convective_term);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  double const initial_residual = evaluate_residual();

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    while(!converged && this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = this->solution[0].block(0);
      pressure_tmp = this->solution[0].block(1);

      // calculate normm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep();

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, this->solution[0].block(0));
      pressure_tmp.add(1.0, this->solution[0].block(1));

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->get_time_step_number() % this->param.output_solver_info_every_timesteps == 0)
      {
        this->pcout << std::endl
                    << "Norm of solution increment:" << std::endl
                    << "  ||incr_abs|| = " << std::scientific << std::setprecision(10) << incr
                    << std::endl
                    << "  ||incr_rel|| = " << std::scientific << std::setprecision(10) << incr_rel
                    << std::endl;
      }

      // check convergence
      if(incr < this->param.abs_tol_steady || incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    while(!converged && this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      this->do_timestep();

      // check convergence by evaluating the residual of
      // the steady-state incompressible Navier-Stokes equations
      double const residual = evaluate_residual();

      if(residual < this->param.abs_tol_steady ||
         residual / initial_residual < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    ExcMessage(
      "Maximum number of time steps exceeded! This might be due to the fact that "
      "(i) the maximum number of iterations is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
double
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::evaluate_residual()
{
  this->navier_stokes_operation->evaluate_nonlinear_residual_steady(this->solution_np,
                                                                    this->solution[0]);

  double residual = this->solution_np.l2_norm();

  // write output
  if((this->get_time_step_number() - 1) % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Norm of residual of steady Navier-Stokes equations:" << std::endl
                << "  ||r|| = " << std::scientific << std::setprecision(10) << residual
                << std::endl;
  }

  return residual;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::analyze_computing_times()
  const
{
  std::string  names[2]     = {"Coupled system ", "Postprocessing "};
  unsigned int N_time_steps = this->get_time_step_number() - 1;

  // iterations
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Average number of iterations:" << std::endl;

  for(unsigned int i = 0; i < iterations.size(); ++i)
  {
    if(i == 0) // coupled system
    {
      this->pcout << "  Step " << i + 1 << ": " << names[i];

      if(this->param.equation_type == EquationType::Stokes ||
         this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
         this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
      {
        this->pcout << std::scientific << std::setprecision(4) << std::setw(10)
                    << iterations[i] / (double)N_time_steps << " linear iterations" << std::endl;
      }
      else
      {
        double n_iter_nonlinear          = (double)N_iter_nonlinear / (double)N_time_steps;
        double n_iter_linear_accumulated = (double)iterations[0] / (double)N_time_steps;

        this->pcout << std::scientific << std::setprecision(4) << std::setw(10) << n_iter_nonlinear
                    << " nonlinear iterations" << std::endl;

        this->pcout << "                         " << std::scientific << std::setprecision(4)
                    << std::setw(10) << n_iter_linear_accumulated
                    << " linear iterations (accumulated)" << std::endl;

        this->pcout << "                         " << std::scientific << std::setprecision(4)
                    << std::setw(10) << n_iter_linear_accumulated / n_iter_nonlinear
                    << " linear iterations (per nonlinear iteration)" << std::endl;
      }
    }
    else if(i == 1) // postprocessing of velocity
    {
      this->pcout << "  Step " << i + 1 << ": " << names[i];
      this->pcout << std::scientific << std::setprecision(4) << std::setw(10)
                  << iterations[i] / (double)N_time_steps << std::endl;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // computing times
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Computing times:          min        avg        max        rel      p_min  p_max "
              << std::endl;

  double total_avg_time = 0.0;

  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    total_avg_time += data.avg;
  }

  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  Step " << i + 1 << ": " << names[i] << std::scientific << std::setprecision(4)
                << std::setw(10) << data.min << " " << std::setprecision(4) << std::setw(10)
                << data.avg << " " << std::setprecision(4) << std::setw(10) << data.max << " "
                << std::setprecision(4) << std::setw(10) << data.avg / total_avg_time << "  "
                << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
                << data.max_index << std::endl;
  }

  this->pcout << "  Time in steps 1-" << computing_times.size() << ":                "
              << std::setprecision(4) << std::setw(10) << total_avg_time << "            "
              << std::setprecision(4) << std::setw(10) << total_avg_time / total_avg_time
              << std::endl;

  this->pcout << std::endl
              << "Number of time steps =              " << std::left << N_time_steps << std::endl
              << "Average wall time per time step =   " << std::scientific << std::setprecision(4)
              << total_avg_time / (double)N_time_steps << std::endl
              << std::endl;

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  this->pcout << "Total wall time in [s] =            " << std::scientific << std::setprecision(4)
              << data.avg << std::endl;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  this->pcout << "Number of MPI processes =           " << N_mpi_processes << std::endl
              << "Computational costs in [CPUs] =     " << data.avg * (double)N_mpi_processes
              << std::endl
              << "Computational costs in [CPUh] =     "
              << data.avg * (double)N_mpi_processes / 3600.0 << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
