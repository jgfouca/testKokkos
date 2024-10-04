#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using namespace std;

// Works
// template<typename T>
// void func(Kokkos::View<T**> const& arg)
// {
//   for (size_t i = 0; i < arg.size(); ++i) {
//     std::cout << arg.data()[i] << " ";
//   }
//   std::cout << std::endl;
// }

// Works
// template<typename T, typename... Props>
// void func(Kokkos::View<T**, Props...> const& arg)
// {
//   for (size_t i = 0; i < arg.size(); ++i) {
//     std::cout << arg.data()[i] << " ";
//   }
//   std::cout << std::endl;
// }

// Doesn't work, cannot convert int to const T
// template<typename T, typename... Props>
// void func(Kokkos::View<const T**, Props...> const& arg)
// {
//   for (size_t i = 0; i < arg.size(); ++i) {
//     std::cout << arg.data()[i] << " ";
//   }
//   std::cout << std::endl;
// }

// Works
void func(Kokkos::View<const int**> const& arg)
{
  for (size_t i = 0; i < arg.size(); ++i) {
    std::cout << arg.data()[i] << " ";
  }
  std::cout << std::endl;
}

// Doesn't work, cannot deduce
// template<typename T, typename Layout, typename Device, typename Traits>
// void func(Kokkos::View<T**, Layout, Device, Traits> const& arg)
// {
//   for (size_t i = 0; i < arg.size(); ++i) {
//     std::cout << arg.data()[i] << " ";
//   }
//   std::cout << std::endl;
// }

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv); {
    Kokkos::View<int**> data("data", 2, 4);
    Kokkos::View<const int**> data2 = data;
    Kokkos::deep_copy(data, 1);
    func(data);
    func(data2);
  }
  Kokkos::finalize();
  return 0;
}
