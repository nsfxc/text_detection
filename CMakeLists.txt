CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

PROJECT(TextDetection)

ADD_EXECUTABLE(TextDetection main.cpp Image.cpp Image.h Pixel.cpp Pixel.h kmeans.cpp kmeans.h image_process.h image_process.cpp)
TARGET_LINK_LIBRARIES(TextDetection ${OpenCV_LIBS})
