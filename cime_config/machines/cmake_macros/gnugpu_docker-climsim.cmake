set(USE_CUDA "TRUE")
string(APPEND CPPDEFS " -DGPU")
if (COMP_NAME STREQUAL gptl)
  string(APPEND CPPDEFS " -DHAVE_NANOTIME -DBIT64 -DHAVE_SLASHPROC -DHAVE_GETTIMEOFDAY")
endif()
string(APPEND CPPDEFS " -DTHRUST_IGNORE_CUB_VERSION_CHECK")
string(APPEND CMAKE_CUDA_FLAGS " -ccbin mpicxx -O2 -arch sm_80 --use_fast_math")

set(Kokkos_ARCH_AMPERE80 On)
set(Kokkos_ENABLE_CUDA On)
set(Kokkos_ENABLE_CUDA_LAMBDA On)
set(Kokkos_ENABLE_SERIAL On)
set(Kokkos_ENABLE_OPENMP Off)

set(CMAKE_CUDA_ARCHITECTURES "80")
string(APPEND CMAKE_C_FLAGS_RELEASE " -O2")
string(APPEND CMAKE_Fortran_FLAGS_RELEASE " -O2")
set(MPICC "mpicc")
set(MPICXX "mpicxx")
set(MPIFC "mpif90")
set(SCC "gcc")
set(SCXX "g++")
set(SFC "gfortran")
