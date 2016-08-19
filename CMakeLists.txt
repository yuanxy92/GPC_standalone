# CAMKE file of Multi view 3D reconstrcution
#
# Author: Shane Yuan
# Date: May 30, 2016
#
cmake_minimum_required(VERSION 2.8)
project( GPC )

if ( WIN32 )
	# add OpenCV path
	set(OpenCV_DIR "D:/lib/opencv/opencv/build/x64/vc14/lib")
	set(OpenCV_CONFIG_PATH "D:/lib/opencv/opencv/build")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB HEADERS  *.h *.hpp)
set(SRC_LIST "gpc_evaluate.cpp" "gpc_train.cpp" "sparse_matching_gpc.cpp" ${HEADERS})

add_executable( GPC ${SRC_LIST} )
target_link_libraries( GPC ${OpenCV_LIBS} )