cmake_minimum_required(VERSION 3.14 )

project( stereoPflasterl )

option (WITH_PCL "Enable PCL (if found)" ON)

find_package( OpenCV REQUIRED )

find_package(Boost 1.74.0 REQUIRED COMPONENTS filesystem system thread)
include_directories(${Boost_INCLUDE_DIRS}) 

# for exporting point clouds:
if(WITH_PCL)
     find_package(PCL 1.7 COMPONENTS visualization)
     if(PCL_FOUND)
       add_definitions(-DPCL_VIZ=1)
     else()
       add_definitions(-DPCL_VIZ=0)
     endif()

     #look for the rest
     find_package(PCL 1.7)
     if(PCL_FOUND)
       include_directories(${PCL_INCLUDE_DIRS} )
         link_directories(${PCL_LIBRARY_DIRS}) 
         add_definitions(-DPCL_FOUND=1 ${PCL_DEFINITIONS})
       else()
         add_definitions(-DPCL_FOUND=0)
     endif()
else()
    add_definitions(-DPCL_FOUND=0 -DPCL_VIZ=0)
endif()

add_executable( stereoPflasterl stereoPflasterl.cpp )
target_link_libraries( stereoPflasterl ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${Boost_THREAD_LIBRARY} )
