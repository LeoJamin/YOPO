#ifndef __QUADROTOR_MSGS_QUADROTOR_MSGS_H__
#define __QUADROTOR_MSGS_QUADROTOR_MSGS_H__

#include <stdint.h>
#include <vector>
#include <yopo_quadrotor_msgs/OutputData.h>
#include <yopo_quadrotor_msgs/StatusData.h>
#include <yopo_quadrotor_msgs/PPROutputData.h>

namespace yopo_quadrotor_msgs
{

bool decodeOutputData(const std::vector<uint8_t> &data,
                      yopo_quadrotor_msgs::OutputData &output);

bool decodeStatusData(const std::vector<uint8_t> &data,
                      yopo_quadrotor_msgs::StatusData &status);

bool decodePPROutputData(const std::vector<uint8_t> &data,
                         yopo_quadrotor_msgs::PPROutputData &output);
}

#endif
