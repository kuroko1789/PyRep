from pyrep.backend import vrep, utils
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.robots.robot_component import RobotComponent
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from pyrep.const import PYREP_SCRIPT_TYPE
from typing import List
import numpy as np
from math import pi, sqrt
from pyrep.misc.laser import Laser

class OmniRob(RobotComponent):
    """Base class representing a robot mobile base with path planning support.
    """

    def __init__(self, count: int, num_wheels: int, num_lasers: int, name: str, laser_name: str):
        """Count is used for when we have multiple copies of mobile bases.

        :param count: used for multiple copies of robots
        :param num_wheels: number of actuated wheels
        :param num_laser: number of lasers
        :param name: string with robot name (same as base in vrep model).
        
        """

        joint_names = ['%s_m_joint%s' % (name, str(i + 1)) for i in
                       range(num_wheels)]

        super().__init__(count, name, joint_names)

        self.num_wheels = num_wheels
        suffix = '' if count == 0 else '#%d' % (count - 1)

        wheel_names = ['%s_wheel%s%s' % (name, str(i + 1), suffix) for i in
                       range(self.num_wheels)]
        self.wheels = [Shape(name) for name in wheel_names]

        joint_slipping_names = [
            '%s_slipping_m_joint%s%s' % (name, str(i + 1), suffix) for i in
            range(self.num_wheels)]
        self.joints_slipping = [Joint(jsname)
                                for jsname in joint_slipping_names]

        laser_suffixs = ['', '#0']
        self.lasers = [Laser(laser_name, suffix) for suffix in laser_suffixs]
        '''
        # Motion planning handles
        self.intermediate_target_base = Dummy(
            '%s_intermediate_target_base%s' % (name, suffix))
        self.target_base = Dummy('%s_target_base%s' % (name, suffix))

        self._collision_collection = vrep.simGetCollectionHandle(
            '%s_base%s' % (name, suffix))
        '''
        # Robot parameters and handle
        self.z_pos = self.get_position()[2]
        #self.target_z = self.target_base.get_position()[-1]
        #self.wheel_radius = self.wheels[0].get_bounding_box()[5]  # Z
        self.w = np.linalg.norm(
            np.array(self.wheels[0].get_position()) -
            np.array(self.wheels[3].get_position())) / 2.
        
        self.l = np.linalg.norm(
            np.array(self.wheels[0].get_position()) -
            np.array(self.wheels[1].get_position())) / 2.
        # Make sure dummies are orphan if loaded with ttm
        #self.intermediate_target_base.set_parent(None)
        #self.target_base.set_parent(None)
       
    def get_velocity(self) -> List[float]:
        joint_velocities = self.get_joint_velocities()
        fb_vel = 0.25 * (joint_velocities[0] + joint_velocities[1] - joint_velocities[2] - joint_velocities[3])
        lr_vel = 0.25 * (-joint_velocities[0] + joint_velocities[1] + joint_velocities[2] - joint_velocities[3])
        r_vel  = 0.25 * (-joint_velocities[0] - joint_velocities[1] - joint_velocities[2] - joint_velocities[3])
        return [fb_vel, lr_vel, r_vel]

    def get_sensor_data(self) -> List[float]:
        return self.lasers[0].get_laser_data() + self.lasers[1].get_laser_data()


    def get_2d_pose(self) -> List[float]:
        """Gets the 2D (top-down) pose of the robot [x, y, yaw].

        :return: A List containing the x, y, yaw (in radians).
        """
        return (self.get_position()[:2] +
                self.get_orientation()[-1:])

    def set_2d_pose(self, pose: List[float]) -> None:
        """Sets the 2D (top-down) pose of the robot [x, y, yaw]

        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, yaw = pose
        self.set_position([x, y, self.z_pos])
        self.set_orientation([0, 0, yaw])

    def assess_collision(self):
        """Silent detection of the robot base with all other entities present in the scene.

        :return: True if collision is detected
        """
        return vrep.simCheckCollision(self._collision_collection,
                                      vrep.sim_handle_all) == 1

    def set_base_angular_velocites(self, velocity: List[float]):
        """Calls required functions to achieve desired omnidirectional effect.

        :param velocity: A List with forwardBackward, leftRight and rotation
            velocity (in radian/s)
        """
        self._reset_wheel()
        fBVel = velocity[0]
        lRVel = velocity[1]
        rVel = velocity[2]
        self.set_joint_target_velocities(
            [fBVel - lRVel -  rVel, fBVel + lRVel - rVel,
             -fBVel + lRVel - rVel, -fBVel - lRVel - rVel])

        #self.set_joint_target_velocities(
        #[1,1,1,1])
   

    def _reset_wheel(self):
        """Required to achieve desired omnidirectional wheel effect.
        """
        [j.reset_dynamic_object() for j in self.wheels]

        p = [[-pi / 4, 0, 0], [pi / 4, 0, pi], [-pi / 4, 0, 0], [pi / 4, 0, pi]]

        for i in range(self.num_wheels):
            self.joints_slipping[i].set_position([0, 0, 0],
                                                 relative_to=self.joints[i],
                                                 reset_dynamics=False)
            self.joints_slipping[i].set_orientation(p[i],
                                                    relative_to=self.joints[i],
                                                    reset_dynamics=False)
            self.wheels[i].set_position([0, 0, 0], relative_to=self.joints[i],
                                        reset_dynamics=False)
            self.wheels[i].set_orientation([0, 0, 0],
                                           relative_to=self.joints[i],
                                           reset_dynamics=False)



