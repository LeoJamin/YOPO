#ifndef __QUADROTOR_SIMULATOR_QUADROTOR_H__
#define __QUADROTOR_SIMULATOR_QUADROTOR_H__

#include <Eigen/Core>
#include <boost/array.hpp>

namespace QuadrotorSimulator
{

class Quadrotor
{
public:
  struct State
  {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
    Eigen::Matrix3d R;
    Eigen::Vector3d omega;
    Eigen::Array4d  motor_rpm;
    Eigen::Vector2d swing_angle; // [theta, phi], only used when cable_length_ > 0
    Eigen::Vector2d swing_velocity; // [dot_theta, dot_phi], only used when cable_length_ > 0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  Quadrotor();

  const Quadrotor::State& getState(void) const;

  void setState(const Quadrotor::State& state);

  void setStatePos(const Eigen::Vector3d& Pos);

  double getMass(void) const;
  void setMass(double mass);

  double getGravity(void) const;
  void setGravity(double g);

  const Eigen::Matrix3d& getInertia(void) const;
  void setInertia(const Eigen::Matrix3d& inertia);

  double getArmLength(void) const;
  void setArmLength(double d);

  double getPropRadius(void) const;
  void setPropRadius(double r);

  double getPropellerThrustCoefficient(void) const;
  void setPropellerThrustCoefficient(double kf);

  double getPropellerMomentCoefficient(void) const;
  void setPropellerMomentCoefficient(double km);

  double getMotorTimeConstant(void) const;
  void setMotorTimeConstant(double k);

  const Eigen::Vector3d& getExternalForce(void) const;
  void setExternalForce(const Eigen::Vector3d& force);

  const Eigen::Vector3d& getExternalMoment(void) const;
  void setExternalMoment(const Eigen::Vector3d& moment);

  double getMaxRPM(void) const;
  void setMaxRPM(double max_rpm);

  double getMinRPM(void) const;
  void setMinRPM(double min_rpm);

  // Inputs are desired RPM for the motors
  // Rotor numbering is:
  //   *1*    Front
  // 3     4
  //    2
  // with 1 and 2 clockwise and 3 and 4 counter-clockwise (looking from top)
  void setInput(double u1, double u2, double u3, double u4);

  // Runs the actual dynamics simulation with a time step of dt
  void step(double dt);

  // For internal use, but needs to be public for odeint
  typedef boost::array<double, 26> InternalState;
  void operator()(const Quadrotor::InternalState& x,
                  Quadrotor::InternalState&       dxdt, const double /* t */);

  Eigen::Vector3d getAcc() const;
  const InternalState& getInternalState() const { return internal_state_; }

private:
  void updateInternalState(void);
  double          payload_mass_; // mass of payload, not included in mass_
  double          cable_length_; // length of cable, 0 for no cable
  double          alpha0; // AOA
  double          g_;     // gravity
  double          mass_;
  Eigen::Matrix3d J_; // Inertia
  double          kf_;
  double          km_;
  double          prop_radius_;
  double          arm_length_;
  double          motor_time_constant_; // unit: sec
  double          max_rpm_;
  double          min_rpm_;

  Quadrotor::State state_;

  Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();

  Eigen::Array4d  input_;
  Eigen::Vector3d external_force_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d external_moment_ = Eigen::Vector3d::Zero();

  InternalState internal_state_;
};
}
#endif
