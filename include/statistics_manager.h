
#ifndef __indexa_statistics_manager_h
#define __indexa_statistics_manager_h

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/base/std_cxx11/function.h>
#include <fluid_base_algorithm.h>
#include <FE_Parameters.h>

template <int dim>
class StatisticsManager
{
public:

  StatisticsManager(const DoFHandler<dim> &dof_handler_velocity);
  // The argument grid_transform indicates how the y-direction that is
  // initially distributed from [0,1] to the actual grid. This must match the
  // transform applied to the triangulation, otherwise the identification of
  // data will fail
  void setup(const std_cxx11::function<Point<dim>(const Point<dim> &)> &grid_tranform);

  void evaluate(const parallel::distributed::Vector<double> &velocity);

  void evaluate(const std::vector<parallel::distributed::Vector<double> > &velocity);

  void evaluate_xwall(const parallel::distributed::Vector<double> &velocity,
                      const DoFHandler<dim> &dof_handler_wdist,
                      const FEParameters<dim> & fe_param,
                      const double viscosity);

  void write_output(const std::string output_prefix,
                    const double      viscosity);

  void reset();

private:
  static const int n_points_y = 21;

  void do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> &velocity);

  void do_evaluate_xwall(const std::vector<const parallel::distributed::Vector<double> *> &velocity,
                         const DoFHandler<dim> &dof_handler_wdist,
                         const FEParameters<dim> & fe_param,
                         const double viscosity);

  const DoFHandler<dim> &dof_handler;
  MPI_Comm communicator;

  std::vector<double> y_glob;
  std::vector<double> velx_glob;
  std::vector<double> vely_glob;
  std::vector<double> velz_glob;
  std::vector<double> velxsq_glob;
  std::vector<double> velysq_glob;
  std::vector<double> velzsq_glob;
  std::vector<double> veluv_glob;
  int numchsamp;
};

#endif
