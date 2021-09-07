cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
#cmake_policy(SET CMP0094 NEW)

project(DriftME LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

find_package(Python COMPONENTS Interpreter Development)
#find_package(pybind11 REQUIRED)
find_package(CUDA)

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)  # the latter is a user definable variable
   enable_language(CUDA)
   add_compile_definitions(CUDA_ENABLED)

   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_50")
else()
   message(WARNING "No CUDA compiler found or disabled by user defined command line variable '-DCUDA_DISABLED=True'.
   You may need to specify its path like \t'export CUDACXX=/usr/local/cuda/bin/nvcc'")
endif()


if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)

	add_library(dme SHARED 
		dme/DME/Rendering.cpp
		dme/DME/StringUtils.cpp
		dme/DME/DriftEstimation.cu
	)
	target_compile_definitions(dme PUBLIC DME_EXPORTS)
	target_compile_features(dme PUBLIC cxx_std_11)
	set_property(TARGET dme PROPERTY POSITION_INDEPENDENT_CODE ON)

	target_include_directories(dme PUBLIC src/*.h)
	#target_compile_features(dme PUBLIC cxx_std_11)

   #add_library(dme STATIC src/DriftEstimation.cu)
   #target_include_directories(dme PUBLIC include)
   set_target_properties(dme PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
   set_property(TARGET dme PROPERTY CUDA_ARCHITECTURES 72 75)
   target_compile_options(dme PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:  # seems to be unnecssary but keep the code block for the moment
                         --expt-relaxed-constexpr
						 --expt-extended-lambda
                         >)
                        # -rdc=true
                        # --gpu-architecture=sm_50

   link_directories(/usr/local/cuda/lib64)
   include_directories("${CUDA_INCLUDE_DIRS}")

   #pybind11_add_module(spline src/pybind_spline.cpp)
   #target_link_libraries(dme PRIVATE dme_cu_impl dme_cpu_impl)

else()  # NO CUDA
	#target_include_directories(dme_cpu_impl PUBLIC include)

   #pybind11_add_module(spline src/pybind_spline.cpp)

	add_library(dme SHARED 
		dme/DME/Rendering.cpp
		dme/DME/StringUtils.cpp
		dme/DME/DriftEstimation.cu
	)
	target_compile_features(dme PUBLIC cxx_std_11)
	set_property(TARGET dme PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(dme PUBLIC DME_EXPORTS)

endif()
