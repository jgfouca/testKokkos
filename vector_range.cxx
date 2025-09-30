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

  assert(argc == 4);

  const int num_teams   = atoi(argv[1]);
  const int team_size   = atoi(argv[2]);
  const int vector_size = atoi(argv[3]);

  std::cout << "num_teams= " << num_teams << ", team_size=" << team_size << ", vector_size=" << vector_size << std::endl;

  TeamPolicy policy(num_teams, Kokkos::AUTO, vector_size);

  std::cout << "size-based ranges" << std::endl;

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const MemberType& team) {
    const int i = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](const int j) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, vector_size), [&](const int k) {
        printf("%d %d %d\n", i, j, k);
      });
    });
  });

  std::cout << "explicit ranges" << std::endl;

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const MemberType& team) {
    const int i = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, team_size), [&](const int j) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, vector_size), [&](const int k) {
        printf("%d %d %d\n", i, j, k);
      });
    });
  });

  return 0;
}
