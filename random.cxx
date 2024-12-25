#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using namespace std;

template <typename LayoutT, typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
struct MDRP
{
  static constexpr Kokkos::Iterate LeftI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Left
    : Kokkos::Iterate::Right;
  static constexpr Kokkos::Iterate RightI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Right
    : Kokkos::Iterate::Left;

  template <int Rank>
  using MDRP_t = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<Rank, LeftI, RightI> >;

  template <int N, typename IntT>
  static inline
  MDRP_t<N> get(const IntT (&upper_bounds)[N])
  {
    assert(N > 1);
    const IntT lower_bounds[N] = {0};
    return MDRP_t<N>(lower_bounds, upper_bounds); //, DefaultTile<N>::value);
  }
};

template <typename ViewT>
void func(const ViewT& view)
{
  using LayoutT = typename ViewT::array_layout;
  using MDRP_t  = MDRP<LayoutT>;

  const uint64_t seed = 12345;

  Kokkos::Random_XorShift64_Pool<> rand_pool(seed);

  Kokkos::parallel_for( MDRP_t::template get<2>({view.extent(0),view.extent(1)}), KOKKOS_LAMBDA (int i, int j) {
    auto generator = rand_pool.get_state();
    view(i, j) = generator.drand(0., 1.);
    rand_pool.free_state(generator);
  });
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    const int dim0 = 8;
    const int dim1 = 4;
    Kokkos::View<double**> orig_view("testlayout", dim0, dim1);

    func(orig_view);

    auto hostm = Kokkos::create_mirror_view(orig_view);
    Kokkos::deep_copy(hostm, orig_view);

    for (size_t i = 0; i < hostm.extent(0); ++i) {
      for (size_t j = 0; j < hostm.extent(0); ++j) {
        std::cout << hostm(i, j) << std::endl;
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
