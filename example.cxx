#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "YAKL.h"

using namespace std;

using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
using HostDevice =
  Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultHostExecutionSpace::memory_space>;

#define TIMED_KERNEL(kernel)                                            \
{                                                                       \
  kernel;                                                               \
}

#define MD_KERNEL2(n1, n2, i1, i2, kernel)                      \
  {                                                                     \
    Kokkos::Array<int, 2> dims_fmk_internal = {n1, n2};                 \
    const int dims_fmk_internal_tot = (n1)*(n2);                        \
    Kokkos::parallel_for(dims_fmk_internal_tot, KOKKOS_LAMBDA (int idx_fmk_internal) { \
      int i1, i2;                                                     \
      conv::unflatten_idx<layout_t>(idx_fmk_internal, dims_fmk_internal, i1, i2); \
      kernel;                                                           \
    });                                                                 \
  }

template <class T, int rank, int myMem> using FArray = yakl::Array<T,rank,myMem,yakl::styleFortran>;

using r1d = FArray<double,1,yakl::memDevice>;
using r2d = FArray<double,2,yakl::memDevice>;
using r3d = FArray<double,3,yakl::memDevice>;

using layout_t = Kokkos::LayoutRight;
using device_t = DefaultDevice;

using r1dk = Kokkos::View<double*,   layout_t, DefaultDevice>;
using r2dk = Kokkos::View<double**,  layout_t, DefaultDevice>;
using r3dk = Kokkos::View<double***, layout_t, DefaultDevice>;

namespace conv {

KOKKOS_INLINE_FUNCTION
void unflatten_idx_left(const int idx, const Kokkos::Array<int, 2>& dims, int& i, int& j)
{
  i = idx % dims[0];
  j = idx / dims[0];
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx_right(const int idx, const Kokkos::Array<int, 2>& dims, int& i, int& j)
{
  i = idx / dims[1];
  j = idx % dims[1];
}

template <typename LayoutT>
KOKKOS_INLINE_FUNCTION
void unflatten_idx(const int idx, const Kokkos::Array<int, 2>& dims, int& i, int& j)
{
  if constexpr (std::is_same_v<LayoutT, Kokkos::LayoutLeft>) {
    unflatten_idx_left(idx, dims, i, j);
  }
  else {
    unflatten_idx_right(idx, dims, i, j);
  }
}

// Copied from EKAT
template <typename View>
struct MemoryTraitsMask {
  enum : unsigned int {
    value = ((View::traits::memory_traits::is_random_access ? Kokkos::RandomAccess : 0) |
             (View::traits::memory_traits::is_atomic ? Kokkos::Atomic : 0) |
             (View::traits::memory_traits::is_restrict ? Kokkos::Restrict : 0) |
             (View::traits::memory_traits::is_aligned ? Kokkos::Aligned : 0) |
             (View::traits::memory_traits::is_unmanaged ? Kokkos::Unmanaged : 0))
      };
};

// Copied from EKAT
template <typename View>
using Unmanaged =
  // Provide a full View type specification, augmented with Unmanaged.
  Kokkos::View<typename View::traits::scalar_array_type,
               typename View::traits::array_layout,
               typename View::traits::device_type,
               Kokkos::MemoryTraits<
                 // All the current values...
                 MemoryTraitsMask<View>::value |
                 // ... |ed with the one we want, whether or not it's
                 // already there.
                 Kokkos::Unmanaged> >;

template <typename LayoutT=layout_t, typename DeviceT=device_t>
struct MDRP
{
  // By default, follow the Layout's fast index
  static constexpr Kokkos::Iterate LeftI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Left
    : Kokkos::Iterate::Right;
  static constexpr Kokkos::Iterate RightI = std::is_same_v<LayoutT, Kokkos::LayoutRight>
    ? Kokkos::Iterate::Right
    : Kokkos::Iterate::Left;

  using exe_space_t = typename DeviceT::execution_space;

  template <int Rank>
  using MDRP_t = Kokkos::MDRangePolicy<exe_space_t, Kokkos::Rank<Rank, LeftI, RightI> >;

  template <int N, typename IntT>
  static inline
  MDRP_t<N> get(const IntT (&upper_bounds)[N])
  {
    assert(N > 1);
    const IntT lower_bounds[N] = {0};
    return MDRP_t<N>(lower_bounds, upper_bounds); //, DefaultTile<N>::value);
  }
};

template <typename RealT=double, typename LayoutT=layout_t, typename DeviceT=device_t>
struct MemPoolSingleton
{
  template <typename T>
  using view_t = Kokkos::View<T, LayoutT, DeviceT>;

  template <typename T>
  static inline
  auto alloc(const int64_t dim1, const int64_t dim2) noexcept
  {
    using uview_t = Unmanaged<view_t<T**>>;
    static_assert(uview_t::rank == 2);
    return uview_t();
  }

  template <typename View>
  static inline
  void dealloc(const View& view) noexcept
  {
  }
};

}

////////////////////////////////////////////////////////////////////////////////
// example is lw_transport_noscat

void example_orig(
  int ncol, int nlay, int ngpt,
  r3d const &t, r2d const &a, r3d const &d, r3d const &u, r2d const &s,
  r3d const &U, r3d const &D)
{
  using yakl::fortran::parallel_for;
  using yakl::fortran::SimpleBounds;
  r2d temp_dn("tmp_dn", ncol, ngpt);
  parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ngpt,ncol) , YAKL_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=2; ilev<=nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay+1,igpt) = D(icol,nlay+1,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay; ilev>=1; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  });
}

void example_final(
  int ncol, int nlay, int ngpt,
  r3d const &t, r2d const &a, r3d const &d, r3d const &u, r2d const &s,
  r3d const &U, r3d const &D)
{
  using yakl::fortran::parallel_for;
  using yakl::fortran::SimpleBounds;
  r2d temp_dn("tmp_dn", ncol, ngpt);
  TIMED_KERNEL(parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ngpt,ncol) , YAKL_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=2; ilev<=nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay+1,igpt) = D(icol,nlay+1,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay; ilev>=1; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  }));
}

void example_orig(
  int ncol, int nlay, int ngpt,
  r3dk const &t, r2dk const &a, r3dk const &d, r3dk const &u, r2dk const &s,
  r3dk const &U, r3dk const &D)
{
  using exe_space_t = Kokkos::DefaultExecutionSpace;
  using MDRPR_t = Kokkos::MDRangePolicy<exe_space_t, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Right> >;

  r2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( MDRPR_t({0, 0}, {ngpt,ncol}) , KOKKOS_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  });
}

void example_mdrp(
  int ncol, int nlay, int ngpt,
  r3dk const &t, r2dk const &a, r3dk const &d, r3dk const &u, r2dk const &s,
  r3dk const &U, r3dk const &D)
{
  using mdrp_t  = typename conv::MDRP<>;

  r2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ngpt,ncol}) , KOKKOS_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  });
}

void example_mdrp_layout(
  int ncol, int nlay, int ngpt,
  r3dk const &t, r2dk const &a, r3dk const &d, r3dk const &u, r2dk const &s,
  r3dk const &U, r3dk const &D)
{
  using mdrp_t  = typename conv::MDRP<layout_t>;

  r2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ncol,ngpt}) , KOKKOS_LAMBDA (int icol, int igpt) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  });
}

void example_pool(
  int ncol, int nlay, int ngpt,
  r3dk const &t, r2dk const &a, r3dk const &d, r3dk const &u, r2dk const &s,
  r3dk const &U, r3dk const &D)
{
  using pool_t = conv::MemPoolSingleton<>;
  using mdrp_t = typename conv::MDRP<layout_t>;

  auto temp_dn  = pool_t::template alloc<double>(ncol,ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ncol,ngpt}) , KOKKOS_LAMBDA (int icol, int igpt) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  });

  pool_t::dealloc(temp_dn);
}

void example_wrap(
  int ncol, int nlay, int ngpt,
  r3dk const &t, r2dk const &a, r3dk const &d, r3dk const &u, r2dk const &s,
  r3dk const &U, r3dk const &D)
{
  using pool_t = conv::MemPoolSingleton<>;

  auto temp_dn  = pool_t::template alloc<double>(ncol,ngpt);
  MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  );

  pool_t::dealloc(temp_dn);
}

template <typename TT, typename AT, typename DT, typename UT,
          typename ST, typename RUT, typename RDT>
void example_generic(
  int ncol, int nlay, int ngpt,
  TT const &t, AT const &a, DT const &d, UT const &u, ST const &s,
  RUT const &U, RDT const &D)
{
  using scalar_t = typename TT::non_const_value_type;
  using layout_t = typename TT::array_layout;
  using device_t = typename TT::device_type;

  using pool_t = conv::MemPoolSingleton<scalar_t, layout_t, device_t>;

  auto temp_dn  = pool_t::template alloc<scalar_t>(ncol,ngpt);

  MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  );

  pool_t::dealloc(temp_dn);
}

template <typename TT, typename AT, typename DT, typename UT,
          typename ST, typename RUT, typename RDT>
void example_timing(
  int ncol, int nlay, int ngpt,
  TT const &t, AT const &a, DT const &d, UT const &u, ST const &s,
  RUT const &U, RDT const &D)
{
  using scalar_t = typename TT::non_const_value_type;
  using layout_t = typename TT::array_layout;
  using device_t = typename TT::device_type;

  using pool_t = conv::MemPoolSingleton<scalar_t, layout_t, device_t>;

  auto temp_dn  = pool_t::template alloc<scalar_t>(ncol,ngpt);

  TIMED_KERNEL(MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      D(icol,ilev,igpt) = t(icol,ilev-1,igpt)*D(icol,ilev-1,igpt) + d(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += D(icol,ilev,igpt);
    }

    // Surface reflection and emission
    U(icol,nlay,igpt) = D(icol,nlay,igpt)*a(icol,igpt) + s(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      U(icol,ilev,igpt) = t(icol,ilev,igpt)*U(icol,ilev+1,igpt) + u(icol,ilev,igpt);
    }
  ));

  pool_t::dealloc(temp_dn);
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    r2dk a, s;
    r3dk t, d, u, U, D;
    example_orig(1, 2, 3, t, a, d, u, s, U, D);
    example_mdrp(1, 2, 3, t, a, d, u, s, U, D);
    example_mdrp_layout(1, 2, 3, t, a, d, u, s, U, D);
    example_pool(1, 2, 3, t, a, d, u, s, U, D);
    example_wrap(1, 2, 3, t, a, d, u, s, U, D);
    example_generic(1, 2, 3, t, a, d, u, s, U, D);
    example_timing(1, 2, 3, t, a, d, u, s, U, D);
  }
  Kokkos::finalize();
  return 0;
}
