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

#define MD_KERNEL2(n1, n2, i1, i2, kernel)      \
  {                                                             \
    Kokkos::Array<int, 2> dims_fmk_internal = {n1, n2};                 \
    const int dims_fmk_internal_tot = (n1)*(n2);                        \
    Kokkos::parallel_for(dims_fmk_internal_tot, KOKKOS_LAMBDA (int idx_fmk_internal) { \
      });                                                               \
  }

template <class T, int rank, int myMem> using FArray = yakl::Array<T,rank,myMem,yakl::styleFortran>;

using real1d = FArray<double,1,yakl::memDevice>;
using real2d = FArray<double,2,yakl::memDevice>;
using real3d = FArray<double,3,yakl::memDevice>;

using layout_t = Kokkos::LayoutRight;
using device_t = DefaultDevice;

using real1dk = Kokkos::View<double*,   layout_t, DefaultDevice>;
using real2dk = Kokkos::View<double**,  layout_t, DefaultDevice>;
using real3dk = Kokkos::View<double***, layout_t, DefaultDevice>;

namespace conv {

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
    return uview_t();
  }

  template <typename View>
  static inline
  void dealloc(const View& view) noexcept
  {
  }
};

}

void lw_transport_noscat_orig(
  int ncol, int nlay, int ngpt, real3d const &trans,
  real2d const &sfc_albedo, real3d const &source_dn, real3d const &source_up, real2d const &source_sfc,
  real3d const &radn_up, real3d const &radn_dn)
{
  using yakl::fortran::parallel_for;
  using yakl::fortran::SimpleBounds;
  real2d temp_dn("tmp_dn", ncol, ngpt);
  parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ngpt,ncol) , YAKL_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=2; ilev<=nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay+1,igpt) = radn_dn(icol,nlay+1,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay; ilev>=1; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });
}

void lw_transport_noscat_final(
  int ncol, int nlay, int ngpt, real3d const &trans,
  real2d const &sfc_albedo, real3d const &source_dn, real3d const &source_up, real2d const &source_sfc,
  real3d const &radn_up, real3d const &radn_dn)
{
  using yakl::fortran::parallel_for;
  using yakl::fortran::SimpleBounds;
  real2d temp_dn("tmp_dn", ncol, ngpt);
  TIMED_KERNEL(parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ngpt,ncol) , YAKL_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=2; ilev<=nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay+1,igpt) = radn_dn(icol,nlay+1,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay; ilev>=1; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  }));
}

void lw_transport_noscat_orig(
  int ncol, int nlay, int ngpt, real3dk const &trans,
  real2dk const &sfc_albedo, real3dk const &source_dn, real3dk const &source_up,
  real2dk const &source_sfc, real3dk const &radn_up, real3dk const &radn_dn)
{
  using exe_space_t = Kokkos::DefaultExecutionSpace;
  using MDRPR_t = Kokkos::MDRangePolicy<exe_space_t, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Right> >;

  real2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( MDRPR_t({0, 0}, {ngpt,ncol}) , KOKKOS_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });
}

void lw_transport_noscat_mdrp(
  int ncol, int nlay, int ngpt, real3dk const &trans,
  real2dk const &sfc_albedo, real3dk const &source_dn, real3dk const &source_up,
  real2dk const &source_sfc, real3dk const &radn_up, real3dk const &radn_dn)
{
  using mdrp_t  = typename conv::MDRP<>;

  real2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ngpt,ncol}) , KOKKOS_LAMBDA (int igpt, int icol) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });
}

void lw_transport_noscat_mdrp_layout(
  int ncol, int nlay, int ngpt, real3dk const &trans,
  real2dk const &sfc_albedo, real3dk const &source_dn, real3dk const &source_up,
  real2dk const &source_sfc, real3dk const &radn_up, real3dk const &radn_dn)
{
  using mdrp_t  = typename conv::MDRP<layout_t>;

  real2dk temp_dn("temp_dn", ncol, ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ncol,ngpt}) , KOKKOS_LAMBDA (int icol, int igpt) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });
}

void lw_transport_noscat_pool(
  int ncol, int nlay, int ngpt, real3dk const &trans,
  real2dk const &sfc_albedo, real3dk const &source_dn, real3dk const &source_up,
  real2dk const &source_sfc, real3dk const &radn_up, real3dk const &radn_dn)
{
  using pool_t = conv::MemPoolSingleton<>;
  using mdrp_t = typename conv::MDRP<layout_t>;

  auto temp_dn  = pool_t::template alloc<double>(ncol,ngpt);
  Kokkos::parallel_for( mdrp_t::template get<2>({ncol,ngpt}) , KOKKOS_LAMBDA (int icol, int igpt) {
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });

  pool_t::dealloc(temp_dn);
}

void lw_transport_noscat_wrap(
  int ncol, int nlay, int ngpt, real3dk const &trans,
  real2dk const &sfc_albedo, real3dk const &source_dn, real3dk const &source_up,
  real2dk const &source_sfc, real3dk const &radn_up, real3dk const &radn_dn)
{
  using pool_t = conv::MemPoolSingleton<>;

  auto temp_dn  = pool_t::template alloc<double>(ncol,ngpt);
  MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  });

  pool_t::dealloc(temp_dn);
}

template <typename TransT, typename SfcAlbedoT, typename SourceDnT, typename SourceUpT,
          typename SourceSfcT, typename RadnUpT, typename RadnDnT>
void lw_transport_noscat_generic(
  int ncol, int nlay, int ngpt, TransT const &trans,
  SfcAlbedoT const &sfc_albedo, SourceDnT const &source_dn, SourceUpT const &source_up,
  SourceSfcT const &source_sfc, RadnUpT const &radn_up, RadnDnT const &radn_dn)
{
  using scalar_t = typename TransT::non_const_data_type;
  using layout_t = typename TransT::array_layout;
  using device_t = typename TransT::device_type;

  using pool_t = conv::MemPoolSingleton<scalar_t, layout_t, device_t>;

  auto temp_dn  = pool_t::template alloc<scalar_t>(ncol,ngpt);

  MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  );

  pool_t::dealloc(temp_dn);
}

template <typename TransT, typename SfcAlbedoT, typename SourceDnT, typename SourceUpT,
          typename SourceSfcT, typename RadnUpT, typename RadnDnT>
void lw_transport_noscat_timing(
  int ncol, int nlay, int ngpt, TransT const &trans,
  SfcAlbedoT const &sfc_albedo, SourceDnT const &source_dn, SourceUpT const &source_up,
  SourceSfcT const &source_sfc, RadnUpT const &radn_up, RadnDnT const &radn_dn)
{
  using scalar_t = typename TransT::non_const_data_type;
  using layout_t = typename TransT::array_layout;
  using device_t = typename TransT::device_type;

  using pool_t = conv::MemPoolSingleton<scalar_t, layout_t, device_t>;

  auto temp_dn  = pool_t::template alloc<scalar_t>(ncol,ngpt);

  TIMED_KERNEL(MD_KERNEL2(ncol, ngpt, icol, igpt,
    // Downward propagation
    for (int ilev=1; ilev<nlay+1; ilev++) {
      radn_dn(icol,ilev,igpt) = trans(icol,ilev-1,igpt)*radn_dn(icol,ilev-1,igpt) + source_dn(icol,ilev-1,igpt);
      temp_dn(icol,igpt) += radn_dn(icol,ilev,igpt);
    }

    // Surface reflection and emission
    radn_up(icol,nlay,igpt) = radn_dn(icol,nlay,igpt)*sfc_albedo(icol,igpt) + source_sfc(icol,igpt);

    // Upward propagation
    for (int ilev=nlay-1; ilev>=0; ilev--) {
      radn_up(icol,ilev,igpt) = trans(icol,ilev  ,igpt)*radn_up(icol,ilev+1,igpt) + source_up(icol,ilev,igpt);
    }
  ));

  pool_t::dealloc(temp_dn);
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    real2dk sfc_albedo, source_sfc;
    real3dk trans, source_dn, source_up, radn_up, radn_dn;
    lw_transport_noscat_orig(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_mdrp(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_mdrp_layout(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_pool(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_wrap(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_generic(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
    lw_transport_noscat_timing(1, 2, 3, trans, sfc_albedo, source_dn, source_up, source_sfc, radn_up, radn_dn);
  }
  Kokkos::finalize();
  return 0;
}
