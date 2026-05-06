#ifndef NETWORK_CONTROL_H_
#define NETWORK_CONTROL_H_

#include <Eigen/Eigen>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <yopo_quadrotor_msgs/PositionCommand.h>
#include <yopo_quadrotor_msgs/SO3Command.h>
#include <tf/transform_datatypes.h>
#include <ros/ros.h>
#include <so3_control/SO3Control.h>
#include <so3_control/HGDO.h>
#include <so3_control/mavros_interface.h>
#include <yopo_quadrotor_msgs/SetTakeoffLand.h>
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <regex>
#include <fstream>
#include <thread>
#include <mutex>
#include <algorithm> 

static constexpr double ONE_G = 9.81;

class NetworkControl
{
public:
    NetworkControl(ros::NodeHandle &node){
        nh_ = node;

        so3_controller_.setMass(mass_);
        disturbance_observer_ = HGDO(control_dt_);

        nh_.param("is_simulation", is_simulation_, false);
        nh_.param("use_disturbance_observer", use_disturbance_observer_, false);
        nh_.param("hover_thrust", hover_thrust_, 0.4);
        nh_.param("kx_xy", kx_xy, 5.7);
        nh_.param("kx_z", kx_z, 6.2);
        nh_.param("kv_xy", kv_xy, 3.4);
        nh_.param("kv_z", kv_z, 4.0);
        nh_.param("record_log", record_log_, false);
        nh_.param("logger_file_name", logger_file_name, std::string("/home/lu/"));
        printf("kx: (%f, %f, %f), kv: (%f, %f, %f) \n", kx_xy, kx_xy, kx_z, kv_xy, kv_xy, kv_z);

        so3_command_pub_ = nh_.advertise<yopo_quadrotor_msgs::SO3Command>("so3_cmd", 10);
        position_cmd_sub_ = nh_.subscribe("position_cmd", 1, &NetworkControl::network_cmd_callback, this, ros::TransportHints().tcpNoDelay());
        odom_sub_ = nh_.subscribe("odom", 1, &NetworkControl::odom_callback, this, ros::TransportHints().tcpNoDelay());
        imu_sub_ = nh_.subscribe("imu", 1, &NetworkControl::imu_callback, this, ros::TransportHints().tcpNoDelay());

        takeoff_land_control_timer = nh_.createTimer(ros::Duration(control_dt_), &NetworkControl::timerCallback, this);

        takeoff_land_srv = nh_.advertiseService("takeoff_land", &NetworkControl::takeoff_land_srv_handle, this);

        if (is_simulation_) {
            // Wait for first odom (state_init_) before takeoff. Bypass the
            // takeoff_land service round-trip and call takeoff_land_thread
            // directly on a detached thread — avoids self-service-call hangs.
            sim_takeoff_timer_ = nh_.createTimer(ros::Duration(0.5), [this](const ros::TimerEvent&) {
                ROS_INFO_ONCE("sim_takeoff_timer fired (state_init=%d)", (int)this->state_init_);
                if (this->state_init_) {
                    ROS_INFO("Triggering simulated takeoff (direct thread, bypassing service)");
                    std::thread t([this]() {
                        yopo_quadrotor_msgs::SetTakeoffLand::Request req;
                        req.takeoff = true;
                        req.takeoff_altitude = 2.0;
                        this->takeoff_land_thread(req);
                    });
                    t.detach();
                    this->sim_takeoff_timer_.stop();
                }
            }, false);
        }
        
    };

    ~NetworkControl(){};

private:
    ros::NodeHandle nh_;
    ros::Publisher so3_command_pub_;
    ros::Subscriber position_cmd_sub_, odom_sub_, imu_sub_;
    ros::ServiceServer takeoff_land_srv;
    ros::Timer takeoff_land_control_timer;
    ros::Timer sim_takeoff_timer_;
    std::mutex mutex_;

    double mass_ = 0.98;
    double control_dt_ = 0.02;
    double hover_thrust_ = 0.4;
    double kx_xy, kx_z, kv_xy, kv_z;
    
    double cur_yaw_ = 0;
    Eigen::Vector3d cur_pos_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d cur_vel_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d cur_acc_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Quaterniond cur_att_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d dis_acc_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d last_des_acc_ = Eigen::Vector3d(0, 0, 0);
    double last_thrust_ = 0;

    Eigen::Vector3d des_pos_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d des_vel_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d des_acc_ = Eigen::Vector3d(0, 0, 0);
    double des_yaw_ = 0;
    double des_yaw_dot_ = 0;

    bool is_simulation_ = false;
    bool state_init_ = false;
    bool ref_valid_ = false;
    bool ctrl_valid_ = false;
    bool position_cmd_init_ = false;
    bool takeoff_cmd_init_ = false;
    bool use_disturbance_observer_ = false;
    bool record_log_ = false;
    
    SO3Control so3_controller_;
    HGDO disturbance_observer_;
    Mavros_Interface mavros_interface_;
    
    std::ofstream logger;
    std::string logger_file_name;

    void initLogRecorder();

    void recordLog(Eigen::Vector3d &cur_v, Eigen::Vector3d &cur_a, Eigen::Vector3d &des_a, Eigen::Vector3d &dis_a, double cur_yaw, double des_yaw);

    Eigen::Vector3d publishHoverSO3Command(Eigen::Vector3d des_pos, Eigen::Vector3d des_vel, Eigen::Vector3d des_acc, double des_yaw, double des_yaw_dot);

    Eigen::Vector3d get_Q_from_ACC(const Eigen::Vector3d &ref_acc, double ref_yaw, Eigen::Quaterniond &quat_des, Eigen::Vector3d &force_des);

    Eigen::Vector3d pub_SO3_command(Eigen::Vector3d ref_acc, double ref_yaw, double cur_yaw);

    void limite_acc(Eigen::Vector3d &acc);

    void network_cmd_callback(const yopo_quadrotor_msgs::PositionCommand::ConstPtr &cmd);

    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom);

    void imu_callback(const sensor_msgs::Imu &imu);

    void timerCallback(const ros::TimerEvent&);

    // mavros interface
    bool takeoff_land_srv_handle(yopo_quadrotor_msgs::SetTakeoffLand::Request &req,
                                 yopo_quadrotor_msgs::SetTakeoffLand::Response &res){
        // Copy request data to avoid dangling reference after service handler returns
        auto req_copy = std::make_shared<yopo_quadrotor_msgs::SetTakeoffLand::Request>(req);
        std::thread t([this, req_copy]() { this->takeoff_land_thread(*req_copy); });
        t.detach();
        res.res = true;
        return true;
    }

    bool arm_disarm_vehicle(bool arm);

    void takeoff_land_thread(yopo_quadrotor_msgs::SetTakeoffLand::Request &req);

    void simulateTakeoff() {
        ros::ServiceClient client = nh_.serviceClient<yopo_quadrotor_msgs::SetTakeoffLand>("takeoff_land");
        yopo_quadrotor_msgs::SetTakeoffLand srv;
        srv.request.takeoff = true;
        srv.request.takeoff_altitude = 2.0;
    
        if (client.call(srv)) {
            ROS_INFO("Takeoff called successfully");
        } else {
            ROS_ERROR("Failed to call takeoff service");
        }
    }
};

#endif