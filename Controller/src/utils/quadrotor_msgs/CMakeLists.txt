cmake_minimum_required(VERSION 2.8.3)
project(yopo_quadrotor_msgs)

# 1. 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  message_generation
  message_runtime
)

find_package(Eigen3 REQUIRED)

# 2. 设置头文件路径 (包含 Eigen 和 Catkin 依赖)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# 3. 添加消息文件 (.msg)
add_message_files(
  FILES
  AuxCommand.msg
  Corrections.msg
  Gains.msg
  OutputData.msg
  PositionCommand.msg
  PPROutputData.msg
  Serial.msg
  SO3Command.msg
  StatusData.msg
  TRPYCommand.msg
  Odometry.msg
  PolynomialTrajectory.msg
  LQRTrajectory.msg
)

# 4. 添加服务文件 (.srv)
add_service_files(
  DIRECTORY srv
  FILES
  SetTakeoffLand.srv
)

# 5. 生成消息头文件
generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
)

# 6. 导出包信息
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES encode_msgs decode_msgs
  CATKIN_DEPENDS roscpp nav_msgs geometry_msgs message_runtime
)

# 7. 编译库文件
add_library(decode_msgs src/decode_msgs.cpp)
add_library(encode_msgs src/encode_msgs.cpp)

# 8. 【关键】添加对消息生成的依赖
# 必须使用 ${PROJECT_NAME}，这会解析为 yopo_quadrotor_msgs_generate_messages_cpp
# 确保在编译 cpp 之前，消息头文件已经生成完毕
add_dependencies(encode_msgs ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(decode_msgs ${PROJECT_NAME}_generate_messages_cpp)

# 9. 链接库 (通常需要链接 catkin 库)
target_link_libraries(encode_msgs ${catkin_LIBRARIES})
target_link_libraries(decode_msgs ${catkin_LIBRARIES})
