cmake_minimum_required(VERSION 3.14)
project(micrograd 
    VERSION 1.0.0
    DESCRIPTION "C++ implementation of micrograd - a tiny autograd engine"
    LANGUAGES CXX
)

# Interface library for header-only
add_library(micrograd INTERFACE)
add_library(micrograd::micrograd ALIAS micrograd)

# Include directories
target_include_directories(micrograd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(micrograd INTERFACE cxx_std_17)

# Compiler warnings for examples
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(MICROGRAD_WARNING_FLAGS -Wall -Wextra -Wpedantic)
elseif(MSVC)
    set(MICROGRAD_WARNING_FLAGS /W4)
endif()

# Examples
option(MICROGRAD_BUILD_EXAMPLES "Build examples" ON)
if(MICROGRAD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS micrograd
    EXPORT microgradTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/micrograd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT microgradTargets
    FILE microgradTargets.cmake
    NAMESPACE micrograd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/micrograd
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/microgradConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/microgradConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/microgradConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/micrograd
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/microgradConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/microgradConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/micrograd
)