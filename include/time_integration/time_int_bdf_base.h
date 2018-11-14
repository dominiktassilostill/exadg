/*
 * time_int_bdf_base.h
 *
 *  Created on: Nov 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_
#define INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include "time_integration/bdf_time_integration.h"
#include "time_integration/extrapolation_scheme.h"

#include "functionalities/restart_data.h"
#include "time_integration/restart.h"

class TimeIntBDFBase
{
public:
  /*
   * Constructor.
   */
  TimeIntBDFBase(double const        start_time_,
                 double const        end_time_,
                 unsigned int const  max_number_of_time_steps_,
                 double const        order_,
                 bool const          start_with_low_order_,
                 bool const          adaptive_time_stepping_,
                 RestartData const & restart_data_);

  /*
   * Destructor.
   */
  virtual ~TimeIntBDFBase()
  {
  }

  /*
   * Perform whole timeloop from start time to end time.
   */
  void
  timeloop();

  /*
   * Pseudo-timestepping for steady-state problems.
   */
  void
  timeloop_steady_problem();

  /*
   * Perform only one time step (which is used when coupling different solvers, equations, etc.).
   */
  bool
  advance_one_timestep(bool write_final_output);

  /*
   * Setters and getters.
   */
  double
  get_scaling_factor_time_derivative_term() const;

  /*
   * Reset the current time.
   */
  void
  reset_time(double const & current_time);

  /*
   * Get the current time t_{n}.
   */
  double
  get_time() const;

  /*
   * Get the time step size.
   */
  double
  get_time_step_size(int const index = 0) const;

  /*
   * Set the time step size. Note that the push-back of time step sizes in case of adaptive time
   * stepping may not be done here because calling this public function several times would falsify
   * the results. Hence, set_time_step_size() is only allowed to overwrite the current time step
   * size.
   */
  void
  set_time_step_size(double const & time_step);

  /*
   * Setup function where allocations/initializations are done. Calls another function
   * setup_derived() in which the setup of derived classes can be performed.
   */
  void
  setup(bool const do_restart);

protected:
  /*
   * Get time at the end of the current time step t_{n+1}.
   */
  double
  get_next_time() const;

  /*
   * Get time at the end of the current time step.
   */
  double
  get_previous_time(int const i /* t_{n-i} */) const;

  /*
   * Get the current time step number.
   */
  unsigned int
  get_time_step_number() const;

  /*
   * Do one time step including different updates before and after the actual solution of the
   * current time step.
   */
  void
  do_timestep();

  /*
   * Update the time integrator constants.
   */
  virtual void
  update_time_integrator_constants();

  /*
   * Get reference to vector with time step sizes
   */
  std::vector<double>
  get_time_step_vector() const;

  /*
   * Update of time step sizes in case of variable time steps.
   */
  void
  push_back_time_step_sizes();

  /*
   * This function implements the OIF sub-stepping algorithm.
   */
  void
  calculate_sum_alphai_ui_oif_substepping(double const cfl, double const cfl_oif);

  /*
   * Read all relevant data from restart files to start the time integrator.
   */
  virtual void
  read_restart();

  /*
   * Calculate time step size.
   */
  virtual void
  calculate_time_step_size() = 0;

  /*
   * Start and end times.
   */
  double const start_time, end_time;

  /*
   * Maximum number of time steps.
   */
  unsigned int const max_number_of_time_steps;

  /*
   * Order of time integration scheme.
   */
  unsigned int const order;

  /*
   * Time integration constants. The extrapolation scheme is not necessarily used for a BDF time
   * integration scheme with fully implicit time stepping, implying a violation of the Liskov
   * substitution principle (OO software design principle). However, it does not appear to be
   * reasonable to complicate the inheritance due to this fact.
   */
  BDFTimeIntegratorConstants bdf;
  ExtrapolationConstants     extra;

  /*
   * Start with low order (1st order) time integration scheme in first time step.
   */
  bool const start_with_low_order;

  /*
   * Use adaptive time stepping?
   */
  bool const adaptive_time_stepping;

  /*
   * Physical time.
   */
  double time;

  /*
   * Vector with time step sizes.
   */
  std::vector<double> time_steps;

  /*
   * Computation time (wall clock time).
   */
  Timer  global_timer;
  double total_time;

  /*
   * A small number which is much smaller than the time step size.
   */
  double const eps;

  /*
   * Output to screen.
   */
  ConditionalOStream pcout;

private:
  /*
   * Allocate solution vectors (has to be implemented by derived classes).
   */
  virtual void
  allocate_vectors() = 0;

  /*
   * Initializes everything related to OIF substepping.
   */
  virtual void
  initialize_oif() = 0;

  /*
   * Initializes the solution vectors by prescribing initial conditions are reading data from
   * restart files and calculates the time step size.
   */
  virtual void
  initialize_solution_and_calculate_timestep(bool do_restart);

  /*
   * Initializes the solution vectors at time t
   */
  virtual void
  initialize_current_solution() = 0;

  /*
   * Initializes the solution vectors at time t - dt[1], t - dt[1] - dt[2], etc.
   */
  virtual void
  initialize_former_solutions() = 0;

  /*
   * Setup of derived classes.
   */
  virtual void
  setup_derived() = 0;

  /*
   * This function prepares the solution vectors for the next time step, e.g., by switching pointers
   * to the solution vectors (called push back here).
   */
  virtual void
  prepare_vectors_for_next_timestep() = 0;

  /*
   * Solve the current time step.
   */
  virtual void
  solve_timestep() = 0;

  /*
   * Solve for a steady-state solution using pseudo-timestepping.
   */
  virtual void
  solve_steady_problem();


  /*
   * Output solver information before solving the time step
   */
  virtual void
  output_solver_info_header() const = 0;

  /*
   * Output estimated computation time until completion of the simulation.
   */
  virtual void
  output_remaining_time() const = 0;

  /*
   * Postprocessing of solution.
   */
  virtual void
  postprocessing() const = 0;

  virtual void
  postprocessing_steady_problem() const;

  /*
   * Analysis of computation times called after having performed the time loop.
   */
  virtual void
  analyze_computing_times() const = 0;

  /*
   * Restart: read solution vectors (has to be implemented in derived classes).
   */
  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) = 0;

  /*
   * Restart: write solution vectors (has to be implemented in derived classes).
   */
  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const = 0;

  /*
   * Write solution vectors to files so that the simulation can be restart from an intermediate
   * state.
   */
  virtual void
  write_restart() const;

  /*
   * Recalculate the time step size after each time step in case of adaptive time stepping.
   */
  virtual double
  recalculate_adaptive_time_step() = 0;

  /*
   * Initializes the solution for OIF sub-stepping at time t_{n-i}.
   */
  virtual void
  initialize_solution_oif_substepping(unsigned int i);

  /*
   * Adds result of OIF sub-stepping for outer loop index i to sum_alphai_ui.
   */
  virtual void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  /*
   * Perform one timestep for OIF sub-stepping and update the solution vectors (switch pointers).
   */
  virtual void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

private:
  /*
   * The number of the current time step starting with time_step_number = 1.
   */
  unsigned int time_step_number;

  /*
   * Restart.
   */
  RestartData const restart_data;
};


TimeIntBDFBase::TimeIntBDFBase(double const        start_time_,
                               double const        end_time_,
                               unsigned int const  max_number_of_time_steps_,
                               double const        order_,
                               bool const          start_with_low_order_,
                               bool const          adaptive_time_stepping_,
                               RestartData const & restart_data_)
  : start_time(start_time_),
    end_time(end_time_),
    max_number_of_time_steps(max_number_of_time_steps_),
    order(order_),
    bdf(order_, start_with_low_order_),
    extra(order_, start_with_low_order_),
    start_with_low_order(start_with_low_order_),
    adaptive_time_stepping(adaptive_time_stepping_),
    time(start_time_),
    time_steps(order_, -1.0),
    total_time(0.0),
    eps(1.e-10),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_step_number(1),
    restart_data(restart_data_)
{
}

void
TimeIntBDFBase::setup(bool const do_restart)
{
  pcout << std::endl << "Setup time integrator ..." << std::endl << std::endl;

  // operator-integration-factor splitting
  initialize_oif();

  // allocate global solution vectors
  allocate_vectors();

  // initializes the solution and calculates the time step size
  initialize_solution_and_calculate_timestep(do_restart);

  // this is where the setup of derived classes is performed
  setup_derived();

  pcout << std::endl << "... done!" << std::endl;
}

void
TimeIntBDFBase::initialize_solution_and_calculate_timestep(bool do_restart)
{
  if(do_restart)
  {
    // The solution vectors and the current time, the time step size, etc. have to be read from
    // restart files.
    read_restart();
  }
  else
  {
    // The time step size might depend on the current solution.
    initialize_current_solution();

    // The time step size has to be computed before the solution can be initialized at times t -
    // dt[1], t - dt[1] - dt[2], etc.
    calculate_time_step_size();

    // Finally, prescribe initial conditions at former instants of time. This is only necessary if
    // the time integrator starts with a high-order scheme in the first time step.
    if(start_with_low_order == false)
      initialize_former_solutions();
  }
}

void
TimeIntBDFBase::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  while(time < (end_time - eps) && time_step_number <= max_number_of_time_steps)
  {
    do_timestep();

    postprocessing();
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... finished time loop!" << std::endl;

  analyze_computing_times();
}

void
TimeIntBDFBase::timeloop_steady_problem()
{
  global_timer.restart();

  postprocessing_steady_problem();

  solve_steady_problem();

  postprocessing_steady_problem();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

bool
TimeIntBDFBase::advance_one_timestep(bool write_final_output)
{
  bool started = time > (start_time - eps);

  // If the time integrator has not yet started, simply increment physical time without solving the
  // current time step.
  if(!started)
  {
    time += time_steps[0];
  }

  if(started && time_step_number == 1)
  {
    pcout << std::endl << "Starting time loop ..." << std::endl;

    global_timer.restart();

    postprocessing();
  }

  // check if we have reached the end of the time loop
  bool finished = !(time < (end_time - eps) && time_step_number <= max_number_of_time_steps);

  // advance one time step and perform postprocessing
  if(started && !finished)
  {
    do_timestep();

    postprocessing();
  }

  // for the statistics
  if(finished && write_final_output)
  {
    total_time += global_timer.wall_time();

    pcout << std::endl << "... done!" << std::endl;

    analyze_computing_times();
  }

  return finished;
}

double
TimeIntBDFBase::get_scaling_factor_time_derivative_term() const
{
  return bdf.get_gamma0() / time_steps[0];
}

double
TimeIntBDFBase::get_time() const
{
  return this->time;
}

double
TimeIntBDFBase::get_next_time() const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}      t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  return this->time + this->time_steps[0];
}

double
TimeIntBDFBase::get_previous_time(int const i /* t_{n-i} */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0, ExcMessage("Invalid parameter."));

  double t = this->time;

  for(int k = 1; k <= i; ++k)
    t -= this->time_steps[k];

  return t;
}

unsigned int
TimeIntBDFBase::get_time_step_number() const
{
  return this->time_step_number;
}

void
TimeIntBDFBase::reset_time(double const & current_time)
{
  // Only allow overwriting the time to a value smaller than start_time (which is needed when
  // coupling different solvers, different domains, etc.).
  if(current_time <= start_time + eps)
    this->time = current_time;
  else
    AssertThrow(false, ExcMessage("The variable time may not be overwritten via public access."));
}

double
TimeIntBDFBase::get_time_step_size(int const i /* dt[i] */) const
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}      t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   */
  AssertThrow(i >= 0 && i <= int(order) - 1, ExcMessage("Invalid access."));

  if(adaptive_time_stepping == true)
  {
    if(time > start_time - eps)
    {
      AssertThrow(time_steps[i] > 0.0, ExcMessage("Invalid or uninitialized time step size."));

      return time_steps[i];
    }
    else // time integrator has not yet started
    {
      // return a large value because we take the minimum time step size when coupling this time
      // integrator to others. This way, this time integrator does not pose a restriction on the
      // time step size.
      return std::numeric_limits<double>::max();
    }
  }
  else // constant time step size
  {
    AssertThrow(time_steps[i] > 0.0, ExcMessage("Invalid or uninitialized time step size."));

    return time_steps[i];
  }
}

std::vector<double>
TimeIntBDFBase::get_time_step_vector() const
{
  return time_steps;
}

void
TimeIntBDFBase::push_back_time_step_sizes()
{
  /*
   * push back time steps
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *                    dt[1]  <- dt[0] <- new_dt
   *
   */
  for(unsigned int i = order - 1; i > 0; --i)
    time_steps[i] = time_steps[i - 1];
}

void
TimeIntBDFBase::set_time_step_size(double const & time_step)
{
  // constant time step sizes
  if(adaptive_time_stepping == false)
  {
    AssertThrow(time_step_number == 1,
                ExcMessage("For time integration with constant time step sizes this "
                           "function can only be called in the very first time step."));
  }

  time_steps[0] = time_step;

  // Fill time_steps array in the first time step
  if(time_step_number == 1)
  {
    for(unsigned int i = 1; i < order; ++i)
      time_steps[i] = time_steps[0];
  }
}

void
TimeIntBDFBase::do_timestep()
{
  update_time_integrator_constants();

  output_solver_info_header();

  solve_timestep();

  output_remaining_time();

  prepare_vectors_for_next_timestep();

  time += time_steps[0];
  ++time_step_number;

  if(adaptive_time_stepping == true)
  {
    push_back_time_step_sizes();
    double const dt = recalculate_adaptive_time_step();
    set_time_step_size(dt);
  }

  if(restart_data.write_restart == true)
  {
    write_restart();
  }
}

void
TimeIntBDFBase::update_time_integrator_constants()
{
  if(adaptive_time_stepping == false) // constant time steps
  {
    bdf.update(time_step_number);
    extra.update(time_step_number);
  }
  else // adaptive time stepping
  {
    bdf.update(time_step_number, time_steps);
    extra.update(time_step_number, time_steps);
  }

  // use this function to check the correctness of the time integrator constants
  //  std::cout << std::endl << "Time step " << time_step_number << std::endl << std::endl;
  //  std::cout << "Coefficients BDF time integration scheme:" << std::endl;
  //  bdf.print();
  //  std::cout << "Coefficients extrapolation scheme:" << std::endl;
  //  extra.print();
}

void
TimeIntBDFBase::calculate_sum_alphai_ui_oif_substepping(double const cfl, double const cfl_oif)
{
  /*
   * Loop over all previous time instants required by the BDF scheme and calculate u_tilde by
   * substepping algorithm, i.e., integrate over time interval t_{n-i} <= t <= t_{n+1} for all 0 <=
   * i <= order using explicit Runge-Kutta methods.
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *  i=0:                                   k=0
   *                                    +---------->+
   *
   *  i=1:                         k=1       k=0
   *                           +--------+---------->+
   *
   *  i=2:               k=2       k=1       k=0
   *                 +-------- +--------+---------->+
   */
  for(unsigned int i = 0; i < order; ++i)
  {
    // initialize solution: u_tilde(s=0) = u(t_{n-i})
    initialize_solution_oif_substepping(i);

    // integrate over time interval t_{n-i} <= t <= t_{n+1}
    // which are i+1 "macro" time steps
    for(int k = i; k >= 0; --k)
    {
      // integrate over interval: t_{n-k} <= t <= t_{n-k+1}

      // calculate start time t_{n-k}
      double const time_n_k = this->get_previous_time(k);

      // number of sub-steps per "macro" time step
      int M = (int)(cfl / (cfl_oif - eps));

      AssertThrow(M >= 1, ExcMessage("Invalid parameters cfl and cfl_oif."));

      // make sure that cfl_oif is not violated
      if(cfl_oif < cfl / double(M) - eps)
        M += 1;

      // calculate sub-stepping time step size delta_s
      double const delta_s = this->get_time_step_size(k) / (double)M;

      for(int m = 0; m < M; ++m)
      {
        do_timestep_oif_substepping_and_update_vectors(time_n_k + delta_s * m, delta_s);
      }
    }

    update_sum_alphai_ui_oif_substepping(i);
  }
}

void
TimeIntBDFBase::read_restart()
{
  const std::string filename = restart_filename(restart_data.filename);
  std::ifstream     in(filename.c_str());

  AssertThrow(in, ExcMessage("File " + filename + " does not exist."));

  boost::archive::binary_iarchive ia(in);

  read_restart_preamble(ia, time, time_steps, order);

  pcout << std::endl
        << "______________________________________________________________________" << std::endl
        << std::endl
        << " Reading restart file at time t = " << this->get_time() << ":" << std::endl;

  read_restart_vectors(ia);

  // In order to change the CFL number (or the time step calculation criterion in general),
  // start_with_low_order = true has to be used. Otherwise, the old solutions would not fit the
  // time step increments.
  if(start_with_low_order == true)
    calculate_time_step_size();

  pcout << std::endl
        << " ... done!" << std::endl
        << "______________________________________________________________________" << std::endl
        << std::endl;
}

void
TimeIntBDFBase::write_restart() const
{
  double const wall_time = global_timer.wall_time();

  if(restart_data.do_restart(wall_time, time - start_time, time_step_number, time_step_number == 2))
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Writing restart file at time t = " << this->get_time() << ":" << std::endl;

    std::ostringstream              oss;
    boost::archive::binary_oarchive oa(oss);

    std::string const name = restart_data.filename;

    write_restart_preamble(oa, name, time, time_steps, order);
    write_restart_vectors(oa);
    write_restart_file(oss, name);

    pcout << std::endl
          << " ... done!" << std::endl
          << "______________________________________________________________________" << std::endl;
  }
}

void
TimeIntBDFBase::postprocessing_steady_problem() const
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

void
TimeIntBDFBase::solve_steady_problem()
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

void
TimeIntBDFBase::initialize_solution_oif_substepping(unsigned int)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

void
TimeIntBDFBase::update_sum_alphai_ui_oif_substepping(unsigned int)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

void
TimeIntBDFBase::do_timestep_oif_substepping_and_update_vectors(double const, double const)
{
  AssertThrow(false, ExcMessage("This function has to be implemented by derived classes."));
}

#endif /* INCLUDE_TIME_INTEGRATION_TIME_INT_BDF_BASE_H_ */