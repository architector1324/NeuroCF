cmake_minimum_required(VERSION 3.10...3.13)

project(NeuroCF LANGUAGES CXX)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")

#################
# CMake Options #
#################
option(NEUROCF_BUILD_EXAMPLES OFF)
option(NEUROCF_BUILD_TESTS OFF)

###############
# Find OpenCL #
###############
find_package(OpenCL REQUIRED)

###############
# Find OpenMP #
###############
find_package(OpenMP REQUIRED)

##############
# Add EasyCL #
##############
add_library(EasyCL INTERFACE)
add_library(EasyCL::EasyCL ALIAS EasyCL)
target_include_directories(EasyCL INTERFACE vendor/MatrixCF/vendor/EasyCL/include/EasyCL)

############
# Add json #
############
add_library(json INTERFACE)
add_library(json::json ALIAS json)
target_include_directories(json INTERFACE vendor/MatrixCF/vendor/json/include)

################
# Add MatrixCF #
################
add_library(MatrixCF INTERFACE)
add_library(MatrixCF::MatrixCF ALIAS MatrixCF)
target_include_directories(MatrixCF INTERFACE vendor/MatrixCF/include/MatrixCF)

################
# Add NeuroCF  #
################
add_library(NeuroCF INTERFACE)
add_library(NeuroCF::NeuroCF ALIAS NeuroCF)
target_include_directories(NeuroCF INTERFACE include)

#######################################
#      Dependency Table               #
#                                     #
# EasyCL   <-- OpenCL                 #
# MatrixCF <-- OpenMP + EasyCL + json #
# NeuroCF <-- MatrixCF                #
#######################################

target_link_libraries(EasyCL INTERFACE OpenCL::OpenCL)
target_link_libraries(MatrixCF INTERFACE OpenMP::OpenMP_CXX)
target_link_libraries(MatrixCF INTERFACE EasyCL::EasyCL)
target_link_libraries(MatrixCF INTERFACE json::json)
target_link_libraries(NeuroCF INTERFACE MatrixCF::MatrixCF)

##################
# Build Examples #
##################
if(NEUROCF_BUILD_EXAMPLES)
    macro(neurocf_add_example EXAMPLENAME)
        add_executable(${EXAMPLENAME} ${ARGN})
        target_link_libraries(${EXAMPLENAME} PRIVATE NeuroCF::NeuroCF)
        set_target_properties(${EXAMPLENAME} PROPERTIES FOLDER examples)
    endmacro()
    add_subdirectory(examples)
endif()