#include <iostream>

#include "Kokkos_Core.hpp"

using namespace std;

int main(int argc, char** argv)
{
  using ExeSpace = Kokkos::DefaultExecutionSpace;
  using TeamPolicy = Kokkos::TeamPolicy<ExeSpace>;
  using MemberType = typename TeamPolicy::member_type;
  using view = Kokkos::View<int**>;

  Kokkos::initialize(argc, argv);

  assert(argc == 3);

  const int num_teams = atoi(argv[1]);
  const int team_size = atoi(argv[2]);

  TeamPolicy policy(num_teams, team_size);

  view my_view("test view", num_teams, team_size);

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const MemberType& team) {
    const int i = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&] (const int& k) {
      my_view(i, k) = i*100 + k;
    });
  });

  using minmax_t       = Kokkos::MinMax<int>;
  using minmax_value_t = typename minmax_t::value_type;

  minmax_value_t one_outer, two_outer;

  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(const MemberType& team, minmax_value_t& one_arg, minmax_value_t& two_arg) {
    const int i = team.league_rank();
    Kokkos::single(Kokkos::PerTeam(team), [&] () {
      const auto tmin = my_view(i, 0);
      const auto tmax = my_view(i, team_size-1);
      if (tmin < one_arg.min_val) one_arg.min_val = tmin;
      if (tmax > one_arg.min_val) one_arg.max_val = tmax;

      const auto tmin2 = my_view(i, 0) + 1;
      const auto tmax2 = my_view(i, team_size-1) + 1;
      if (tmin2 < two_arg.min_val) two_arg.min_val = tmin;
      if (tmax2 > two_arg.min_val) two_arg.max_val = tmax;
    });
  }, minmax_t(one_outer), minmax_t(two_outer));

  cout << "Expected Min: " << my_view(0,0)      << ", expected max: " << my_view(num_teams-1, team_size-1) << std::endl;
  cout << "Min was:      " << one_outer.min_val << ", max was:      " << one_outer.max_val << std::endl;

  cout << "Expected Min2: " << my_view(0,0)+1    << ", expected max2: " << my_view(num_teams-1, team_size-1)+1 << std::endl;
  cout << "Min2 was:      " << two_outer.min_val << ", max2 was:      " << two_outer.max_val << std::endl;

  minmax_value_t se_b_mm, ke_b_mm, wv_b_mm, wl_b_mm;
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(const MemberType& team, minmax_value_t& se_b_arg, minmax_value_t& ke_b_arg) {
  }, minmax_t(se_b_mm), minmax_t(ke_b_mm));
  return 0;
}
