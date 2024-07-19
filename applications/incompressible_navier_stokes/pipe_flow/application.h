/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_PIPE_FLOW_MESH_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_PIPE_FLOW_MESH_H_

#include <deal.II/grid/grid_in.h>

namespace ExaDG
{
namespace IncNS
{
double
interpolate_linear_periodic(std::vector<double> x_vals, std::vector<double> y_vals, double x)
{
  // Determine the x_val within range (necessary to extrapolate periodically)
  double rx = fmod(x, x_vals.back() - x_vals[0]);

  // Determine the lower and upper element for interpolation
  auto         i = lower_bound(x_vals.begin(), x_vals.end(), rx);
  unsigned int k = i - x_vals.begin();
  if(i == x_vals.end())
    --i;
  else if(*i == rx)
  {
    return y_vals[k];
  }
  unsigned int l = k ? k - 1 : 1;

  // Perform linear interpolation
  return y_vals[l] + ((y_vals[k] - y_vals[l]) / (x_vals[k] - x_vals[l])) * (rx - x_vals[l]);
}

template<int dim>
class InflowProfileVelocity : public dealii::Function<dim>
{
public:
  InflowProfileVelocity(std::vector<double> times,
                        std::vector<double> values,
                        double const        radius_x,
                        double const        radius_z,
                        double const        x,
                        double const        z)
    : dealii::Function<dim>(dim, 0.0),
      times(times),
      values(values),
      radiusx2(radius_x * radius_x),
      radiusz2(radius_z * radius_z),
      x(x),
      z(z)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double result = 0.0;

    // Get the current time
    double const t = this->get_time();

    // Get the current flow value
    double max_vel = interpolate_linear_periodic(times, values, t);

    if(component == 1)
    {
      double dx  = p[0] - x;
      double dz  = p[2] - z;
      double fac = dx * dx / radiusx2 + dz * dz / radiusz2;
      result     = std::max(0.0, max_vel * (1.0 - fac));
    }
    return result;
  }

private:
  std::vector<double> times, values;
  double              radiusx2, radiusz2, x, z;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("InflowTimes", inflow_times, "Time values of the inlet flow profile time series.", dealii::Patterns::List(dealii::Patterns::Double(), 2));
      prm.add_parameter("InflowValues", inflow_values, "Max velocity values of the inlet flow profile time series.", dealii::Patterns::List(dealii::Patterns::Double(), 2));
      prm.add_parameter("Viscosity", input_viscosity, "Kinematic Viscosity");
      prm.add_parameter("InflowRadiusX", input_radius_x, "Ellipse x axis radius of parabolic inflow profile");
      prm.add_parameter("InflowRadiusZ", input_radius_z, "Ellipse z axis radius of parabolic inflow profile");
      prm.add_parameter("InflowPosX", input_x, "X position of parabolic inflow profile center");
      prm.add_parameter("InflowPosZ", input_z, "Z position of parabolic inflow profile center");
      prm.add_parameter("InputMesh", input_mesh_path, "Path to mesh VTU file");
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();
    double input_max_vel = *std::max_element(inflow_values.begin(), inflow_values.end());
    max_velocity         = 2.0 * input_max_vel;
    end_time             = inflow_times.back();
    end_time             = 1e-2;
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                   = ProblemType::Unsteady;
    this->param.equation_type                  = EquationType::NavierStokes;
    this->param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.use_outflow_bc_convective_term = false;
    this->param.right_hand_side                = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = input_viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                     = SolverType::Unsteady;
    this->param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Implicit;
    this->param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping          = true;
    this->param.max_velocity                    = max_velocity;
    this->param.cfl                             = 8.0;
    this->param.cfl_exponent_fe_degree_velocity = 1.5;
    this->param.time_step_size                  = 1.0e-1;
    this->param.order_time_integrator           = 2;    // 1; // 2; // 3;
    this->param.start_with_low_order            = true; // true; // false;

    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement; // ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-6;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 1000;

    // SPATIAL DISCRETIZATION
    this->param.grid.element_type                 = ElementType::Simplex;
    this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
    this->param.grid.create_coarse_triangulations = true;

    this->param.mapping_degree              = this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;

    this->param.grid.file_name = input_mesh_path;

    this->param.inverse_mass_operator.implementation_type =
      InverseMassType::ElementwiseKrylovSolver;
    this->param.inverse_mass_operator.preconditioner            = PreconditionerMass::PointJacobi;
    this->param.inverse_mass_preconditioner.implementation_type = InverseMassType::BlockMatrices;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // divergence penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.continuity_penalty_use_boundary_data       = true;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
    this->param.IP_factor_viscous      = 1.;

    // PROJECTION METHODS

    // pressure Poisson equation -> pressure
    this->param.solver_pressure_poisson              = SolverPressurePoisson::CG; // FGMRESs
    this->param.solver_data_pressure_poisson         = SolverData(1e5, 1.e-12, 1.e-5, 30);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;
    this->param.multigrid_data_pressure_poisson.smoother_data.smoother =
      MultigridSmoother::Chebyshev;
    this->param.multigrid_data_pressure_poisson.smoother_data.iterations = 5;
    this->param.multigrid_data_pressure_poisson.coarse_problem.solver =
      MultigridCoarseGridSolver::CG;
    this->param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG; // PointJacobi;

    // projection step -> penalty step
    this->param.solver_projection              = SolverProjection::CG; // FGMRES
    this->param.solver_data_projection         = SolverData(1e5, 1.e-12, 1.e-5);
    this->param.preconditioner_projection      = PreconditionerProjection::Multigrid;
    this->param.multigrid_data_projection.type = MultigridType::hMG;
    this->param.multigrid_data_projection.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    this->param.multigrid_data_projection.smoother_data.iterations = 5;
    this->param.multigrid_data_projection.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data_projection.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    this->param.update_preconditioner_projection = false;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      // -> viscous step
      this->param.solver_momentum              = SolverMomentum::CG;
      this->param.solver_data_momentum         = SolverData(1e5, 1.e-12, 1.e-5);
      this->param.preconditioner_momentum      = MomentumPreconditioner::Multigrid;
      this->param.multigrid_data_momentum.type = MultigridType::cphMG;
      this->param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionDiffusion;
      this->param.multigrid_data_momentum.smoother_data.smoother   = MultigridSmoother::Chebyshev;
      this->param.multigrid_data_momentum.smoother_data.iterations = 5;
      this->param.multigrid_data_momentum.coarse_problem.solver    = MultigridCoarseGridSolver::CG;
      this->param.multigrid_data_momentum.coarse_problem.preconditioner =
        MultigridCoarseGridPreconditioner::AMG;
      this->param.multigrid_data_momentum.coarse_problem.amg_data.ml_data.n_cycles     = 1;
      this->param.multigrid_data_momentum.coarse_problem.amg_data.boomer_data.max_iter = 1;
    }

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // Newton solver
      this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

      // linear solver
      this->param.solver_momentum                = SolverMomentum::GMRES;
      this->param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-6, 100);
      this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
      this->param.update_preconditioner_momentum = false;
    }


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES; // GMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        GridUtilities::read_external_triangulation<dim>(tria, this->param.grid);
        if(global_refinements > 0)
          tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity

    // no-slip walls
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // inflow
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1,
           new InflowProfileVelocity<dim>(
             inflow_times, inflow_values, input_radius_x, input_radius_z, input_x, input_z)));

    // outflow
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure

    // no-slip walls
    this->boundary_descriptor->pressure->neumann_bc.insert(0);

    // inflow
    this->boundary_descriptor->pressure->neumann_bc.insert(1);

    // outflow
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active                = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time               = start_time;
    pp_data.output_data.time_control_data.trigger_interval         = (end_time - start_time) / 100.0;
    //pp_data.output_data.time_control_data.trigger_every_time_steps = true;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = this->output_parameters.filename;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_q_criterion         = false;
    pp_data.output_data.write_higher_order        = true;
    pp_data.output_data.write_boundary_IDs        = true;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      // Deactivate higher-order output for simplex elements
      pp_data.output_data.write_higher_order = false;
    }
    pp_data.output_data.degree = this->param.degree_u;

    pp_data.error_data_u.write_errors_to_file               = true;
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data_u.analytical_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.name                      = "velocity_error";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string         input_mesh_path;
  double              input_radius_x  = 1.0;
  double              input_radius_z  = 1.0;
  double              input_x         = 0.0;
  double              input_z         = 0.0;
  double              input_viscosity = 1.0e-1;
  std::vector<double> inflow_times    = {0.0, 1.0};
  std::vector<double> inflow_values   = {1.0, 1.0};

  double max_velocity;

  double const start_time = 0.0;
  double       end_time   = 1.0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_POISEUILLE_H_ */
