cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
#cmake_policy(SET CMP0094 NEW)

option(CUDA_DISABLED "CUDA_DISABLED" OFF)

project(DME LANGUAGES CXX)

include(CheckLanguage)

if(NOT CUDA_DISABLED)
	check_language(CUDA)
endif()

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH "bin")

#find_package(Python COMPONENTS Interpreter Development)
#find_package(pybind11 REQUIRED)
#find_package(CUDA)

# This is used on linux but ignored by the visual studio builds
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/release)

#message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)
	enable_language(CUDA)
	
	#if (UNIX)
			# cuda compiler path for linux, as cmake cant find it
			#	set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
			#	endif (UNIX)

   enable_language(CUDA)
   add_compile_definitions(CUDA_ENABLED)

   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_50 --expt-extended-lambda")
   #message(WARNING "No CUDA compiler found or disabled by user defined command line variable '-DCUDA_DISABLED=True'.
   #You may need to specify its path like \t'export CUDACXX=/usr/local/cuda/bin/nvcc'")

	message("Targeting CUDA build...")

	add_library(dme_cuda SHARED 
		dme/DME/Rendering.cpp
		dme/DME/StringUtils.cpp
		dme/DME/DriftEstimation.cu
	)
	target_compile_definitions(dme_cuda PUBLIC DME_EXPORTS)
	target_compile_features(dme_cuda PUBLIC cxx_std_14)
	set_property(TARGET dme_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

	target_include_directories(dme_cuda PUBLIC src/*.h)
	#target_compile_features(dme PUBLIC cxx_std_11)

   #add_library(dme STATIC src/DriftEstimation.cu)
   #target_include_directories(dme PUBLIC include)
   set_target_properties(dme_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
	set_target_properties(dme_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin/release/")
	set_target_properties(dme_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin/debug/")
   set_property(TARGET dme_cuda PROPERTY CUDA_ARCHITECTURES 72 75)
   target_compile_options(dme_cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:  # seems to be unnecssary but keep the code block for the moment
                        --expt-relaxed-constexpr
			--expt-extended-lambda
                         >)
                        # -rdc=true
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)                     # --gpu-architecture=sm_50

   link_directories(/usr/local/cuda/lib64)
   #include_directories("${CUDA_INCLUDE_DIRS}")

   #pybind11_add_module(spline src/pybind_spline.cpp)
   #target_link_libraries(dme PRIVATE dme_cu_impl dme_cpu_impl)

else()  # NO CUDA
	#target_include_directories(dme_cpu_impl PUBLIC include)

	message("Targeting non-CUDA build...")

	message(WARNING "No CUDA compiler found or disabled by user defined command line variable '-DCUDA_DISABLED=True'.
		In case you want to use CUDA, you may need to specify its path like \t'export CUDACXX=/usr/local/cuda/bin/nvcc'")

   #pybind11_add_module(spline src/pybind_spline.cpp)

	add_library(dme_cpu SHARED 
		dme/DME/Rendering.cpp
		dme/DME/StringUtils.cpp
		dme/DME/DriftEstimation.cpp
	)
	set_target_properties(dme_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin/release/")
	set_target_properties(dme_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin/debug/")
	target_compile_features(dme_cpu PUBLIC cxx_std_14)
	set_property(TARGET dme_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(dme_cpu PUBLIC DME_EXPORTS)

endif()
