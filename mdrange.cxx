#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include "YAKL.h"

using namespace std;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP2 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Left> >;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP3RL = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Left> >;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP3LR = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Right> >;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP3D = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3> >;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP3R = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3, Kokkos::Iterate::Right> >;

template <typename ExecutionSpace=Kokkos::DefaultExecutionSpace>
using MDRangeP3L = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3, Kokkos::Iterate::Left> >;


using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
using HostDevice =
  Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultHostExecutionSpace::memory_space>;

template <typename T>
void print_iteration(T policy, const std::string& name, int dim1, int dim2, int dim3)
{
  std::cout << "Iterating " << name << std::endl;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j, int k) {
    cout << "  Iterating (" << i << ", " << j << ", " << k << ")" << endl;
  });
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx(const int idx, const Kokkos::Array<int, 2>& dims, int& i, int& j)
{
  i = idx / dims[1];
  j = idx % dims[1];
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx(const int idx, const Kokkos::Array<int, 3>& dims, int& i, int& j, int& k)
{
  i = idx / (dims[2] * dims[1]);
  j = (idx / dims[2]) % dims[1];
  k =  idx % dims[2];
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx_rev(const int idx, const Kokkos::Array<int, 3>& dims, int& i, int& j, int& k)
{
  i = idx % dims[0];
  j = (idx / dims[0]) % dims[1];
  k = idx / (dims[0] * dims[1]);
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx(const int idx, const Kokkos::Array<int, 4>& dims, int& i, int& j, int& k, int& l)
{
  i = idx / (dims[3]*dims[2]*dims[1]);
  j = (idx / (dims[3]*dims[2])) % dims[1];
  k = (idx / dims[3]) % dims[2];
  l = idx % dims[3];
}

KOKKOS_INLINE_FUNCTION
void unflatten_idx_rev(const int idx, const Kokkos::Array<int, 4>& dims, int& i, int& j, int& k, int& l)
{
  i = idx % dims[0];
  j = (idx / dims[0]) % dims[1];
  k = (idx / (dims[0]*dims[1])) % dims[2];
  l = idx / (dims[0]*dims[1]*dims[2]);
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    int dim1 = 2;
    int dim2 = 3;
    int dim3 = 4;
    int dim4 = 5;

    print_iteration(MDRangeP3RL<>({0,0,0}, {dim1, dim2, dim3}), "MDRangeP3RL", dim1, dim2, dim3);
    print_iteration(MDRangeP3LR<>({0,0,0}, {dim1, dim2, dim3}), "MDRangeP3LR", dim1, dim2, dim3);
    print_iteration(MDRangeP3L <>({0,0,0}, {dim1, dim2, dim3}), "MDRangeP3L",  dim1, dim2, dim3);
    print_iteration(MDRangeP3R <>({0,0,0}, {dim1, dim2, dim3}), "MDRangeP3R",  dim1, dim2, dim3);
    print_iteration(MDRangeP3D <>({0,0,0}, {dim1, dim2, dim3}), "MDRangeP3D",  dim1, dim2, dim3);

    // print_iteration(MDRangeP3RL<>({0,0,0}, {dim1, dim2, dim3}, {1,1,1}), "MDRangeP3RL_tile", dim1, dim2, dim3);
    // print_iteration(MDRangeP3LR<>({0,0,0}, {dim1, dim2, dim3}, {1,1,1}), "MDRangeP3LR_tile", dim1, dim2, dim3);
    // print_iteration(MDRangeP3L <>({0,0,0}, {dim1, dim2, dim3}, {1,1,1}), "MDRangeP3L_tile",  dim1, dim2, dim3);
    // print_iteration(MDRangeP3R <>({0,0,0}, {dim1, dim2, dim3}, {1,1,1}), "MDRangeP3R_tile",  dim1, dim2, dim3);
    // print_iteration(MDRangeP3D <>({0,0,0}, {dim1, dim2, dim3}, {1,1,1}), "MDRangeP3D_tile",  dim1, dim2, dim3);

    std::cout << "YAKL fortran" << std::endl;
    parallel_for( YAKL_AUTO_LABEL() , yakl::fortran::SimpleBounds<3>(dim1, dim2, dim3) , YAKL_LAMBDA (int i, int j, int k) {
      cout << "  Iterating (" << i-1 << ", " << j-1 << ", " << k-1 << ")" << endl;
    });

    std::cout << "YAKL c" << std::endl;
    parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<3>(dim1, dim2, dim3) , YAKL_LAMBDA (int i, int j, int k) {
      cout << "  Iterating (" << i << ", " << j << ", " << k << ")" << endl;
    });

    std::cout << "Validating unflattens" << std::endl;
    Kokkos::Array<int, 3> dims3 = {dim1, dim2, dim3};
    int idx = 0;
    for (int i = 0; i < dim1; ++i) {
      for (int j = 0; j < dim2; ++j) {
        for (int k = 0; k < dim3; ++k, ++idx) {
          int iv, jv, kv;
          unflatten_idx(idx, dims3, iv, jv, kv);
          assert(k == kv);
          assert(j == jv);
          assert(i == iv);
        }
      }
    }

    idx = 0;
    for (int k = 0; k < dim3; ++k) {
      for (int j = 0; j < dim2; ++j) {
        for (int i = 0; i < dim1; ++i, ++idx) {
          int iv, jv, kv;
          unflatten_idx_rev(idx, dims3, iv, jv, kv);
          assert(k == kv);
          assert(j == jv);
          assert(i == iv);
        }
      }
    }

    Kokkos::Array<int, 4> dims4 = {dim1, dim2, dim3, dim4};

    idx = 0;
    for (int i = 0; i < dim1; ++i) {
      for (int j = 0; j < dim2; ++j) {
        for (int k = 0; k < dim3; ++k) {
          for (int l = 0; l < dim4; ++l, ++idx) {
            int iv, jv, kv, lv;
            unflatten_idx(idx, dims4, iv, jv, kv, lv);
            assert(l == lv);
            assert(k == kv);
            assert(j == jv);
            assert(i == iv);
          }
        }
      }
    }

    idx = 0;
    for (int l = 0; l < dim4; ++l) {
      for (int k = 0; k < dim3; ++k) {
        for (int j = 0; j < dim2; ++j) {
          for (int i = 0; i < dim1; ++i, ++idx) {
            int iv, jv, kv, lv;
            unflatten_idx_rev(idx, dims4, iv, jv, kv, lv);
            assert(l == lv);
            assert(k == kv);
            assert(j == jv);
            assert(i == iv);
          }
        }
      }
    }

  }
  Kokkos::finalize();
  return 0;
}
