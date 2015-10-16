CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(TextDetection)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(main main.cpp Image.cpp Image.h Pixel.cpp Pixel.h kmeans.cpp kmeans.h image_process.cpp image_process.h)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})
