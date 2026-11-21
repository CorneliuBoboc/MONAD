cmake_minimum_required(VERSION 3.16)
project(morphix_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(MHD REQUIRED libmicrohttpd)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
pkg_check_modules(CURL REQUIRED libcurl)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)
pkg_check_modules(ZIP REQUIRED libzip)
pkg_check_modules(PUGIXML REQUIRED pugixml)

add_executable(morphix_cpp app.cpp)
target_include_directories(morphix_cpp PRIVATE
    ${MHD_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${POPPLER_INCLUDE_DIRS}
    ${ZIP_INCLUDE_DIRS}
    ${PUGIXML_INCLUDE_DIRS}
)
target_link_libraries(morphix_cpp PRIVATE
    ${MHD_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    ${CURL_LIBRARIES}
    ${POPPLER_LIBRARIES}
    ${ZIP_LIBRARIES}
    ${PUGIXML_LIBRARIES}
    Threads::Threads
)
