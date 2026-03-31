#ifndef __QUADROTOR_MSGS_QUADROTOR_MSGS_H__
#define __QUADROTOR_MSGS_QUADROTOR_MSGS_H__

#include <stdint.h>
#include <vector>
#include <yopo_quadrotor_msgs/SO3Command.h>
#include <yopo_quadrotor_msgs/TRPYCommand.h>
#include <yopo_quadrotor_msgs/Gains.h>

namespace yopo_quadrotor_msgs
{

void encodeSO3Command(const yopo_quadrotor_msgs::SO3Command &so3_command,
                      std::vector<uint8_t> &output);
void encodeTRPYCommand(const yopo_quadrotor_msgs::TRPYCommand &trpy_command,
                       std::vector<uint8_t> &output);

void encodePPRGains(const yopo_quadrotor_msgs::Gains &gains,
                    std::vector<uint8_t> &output);
}

#endif
