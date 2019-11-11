from pyrep.backend import vrep, utils
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.robots.robot_component import RobotComponent
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from pyrep.const import PYREP_SCRIPT_TYPE
from pyrep.misc.laser import Laser
from typing import List
import numpy as np
from math import pi, sqrt
from math import sqrt, atan2, sin, cos

class PioneerP3dx(RobotComponent):
	"""Base class representing a robot mobile base with path planning support.
	"""

	def __init__(self, count: int, num_wheels: int, num_sensors: int, name: str, sensor_name: str, laser_name: str):
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

		#joint_slipping_names = [
		#    '%s_slipping_m_joint%s%s' % (name, str(i + 1), suffix) for i in
		#    range(self.num_wheels)]
		#self.joints_slipping = [Joint(jsname)
		#                        for jsname in joint_slipping_names]
		self.num_sensors = num_sensors
		sensor_names = ['%s_%s%s' % (name, sensor_name, str(i + 1)) for i in range(self.num_sensors)]
		self.sensors = [ProximitySensor(name) for name in sensor_names]
		
		self.lasers = Laser(laser_name,'')

		# Motion planning handles
		self.intermediate_target_base = Dummy(
			'%s_intermediate_target_base%s' % (name, suffix))
		
		self.target_base = Dummy('%s_target_base%s' % (name, suffix))
		
		self._collision_collection = vrep.simGetCollectionHandle(
			'Robot')
		self.target_z = self.target_base.get_position()[-1]
		# Robot parameters and handle
		self.z_pos = self.get_position()[2]
		#self.target_z = self.target_base.get_position()[-1]
		self.wheel_radius = self.wheels[0].get_bounding_box()[1] 
		self.wheel_distance = np.linalg.norm(
			np.array(self.wheels[0].get_position()) -
			np.array(self.wheels[1].get_position()))

		# Make sure dummies are orphan if loaded with ttm
		self.intermediate_target_base.set_parent(None)
		self.target_base.set_parent(None)

		self.cummulative_error = 0
		self.prev_error = 0

		# PID controller values.
		# TODO: expose to user through constructor.
		self.Kp = 0.1
		self.Ki = 0.0
		self.Kd = 0.0
		self.desired_velocity = 0.1

		self._path_done = False
		self.i_path = -1
		self.inter_done = True

		self.drawing_handle = vrep.simAddDrawingObject(
			objectType=vrep.sim_drawing_points, size=10, duplicateTolerance=0,
			parentObjectHandle=-1, maxItemCount=99999,
			ambient_diffuse=[1, 0, 0])


	#def get_wheel_radius(self):
	#    min_x = vrep.simGetObjectFloatParameter(self.wheels[0]._handle, 15)
	#    max_x = vrep.simGetObjectFloatParameter(self.wheels[0]._handle, 18)
	#    return (max_x - min_x)/2.

	def get_velocity(self) -> List[float]:
		#print(self.wheel_radius)
		joint_velocities = self.get_joint_velocities()
		
		v_l = joint_velocities[0]
		v_r = joint_velocities[1]
		#print('get_velocity')
		#print(v_l)
		#print(v_r)
		v = self.wheel_radius/2. * (v_l + v_r)
		omega = self.wheel_radius/self.wheel_distance * (v_r - v_l)
		return [v, omega]

	def set_base_angular_velocites(self, velocity: List[float]):
		v = velocity[0]
		omega = velocity[1]   
   
		vr = ((2. * v + omega * self.wheel_distance) /
			  (2. * self.wheel_radius))
		
		vl = ((2. * v - omega * self.wheel_distance) /
			  (2. * self.wheel_radius))

		
		self.set_joint_target_velocities([vl, vr])
	   
	def get_sensor_data(self):
		data = []
		for i in range(self.num_sensors):
			ret, detected_point = self.sensors[i].get_sensor_data()
			dist = np.linalg.norm(detected_point)
			data.append(ret)
			data.append(dist)
		return data

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



	def get_nonlinear_path_points(self, position: List[float],
						   angle=0,
						   boundaries=8,
						   path_pts=600,
						   ignore_collisions=False,
						   algorithm=Algos.RRTConnect) -> List[List[float]]:
		"""Gets a non-linear (planned) configuration path given a target pose.

		:param position: The x, y, z position of the target.
		:param angle: The z orientation of the target (in radians).
		:param boundaries: A float defining the path search in x and y direction
		[[-boundaries,boundaries],[-boundaries,boundaries]].
		:param path_pts: number of sampled points returned from the computed path
		:param ignore_collisions: If collision checking should be disabled.
		:param algorithm: Algorithm used to compute path
		:raises: ConfigurationPathError if no path could be created.

		:return: A non-linear path (x,y,angle) in the xy configuration space.
		"""

		# Base dummy required to be parent of the robot tree
		# self.base_ref.set_parent(None)
		# self.set_parent(self.base_ref)

		# Missing the dist1 for intermediate target

		self.target_base.set_position([position[0], position[1], self.target_z])
		self.target_base.set_orientation([0, 0, angle])

		handle_base = self.get_handle()
		handle_target_base = self.target_base.get_handle()

		# Despite verbosity being set to 0, OMPL spits out a lot of text
		with utils.suppress_std_out_and_err():
			_, ret_floats, _, _ = utils.script_call(
				'getNonlinearPathMobile@PyRep', PYREP_SCRIPT_TYPE,
				ints=[handle_base, handle_target_base,
					  self._collision_collection,
					  int(ignore_collisions), path_pts], floats=[boundaries],
					  strings=[algorithm.value])

		# self.set_parent(None)
		# self.base_ref.set_parent(self)

		if len(ret_floats) == 0:
			raise ConfigurationPathError('Could not create path.')

		path = []
		for i in range(0, len(ret_floats) // 3):
			inst = ret_floats[3 * i:3 * i + 3]
			if i > 0:
				dist_change = sqrt((inst[0] - prev_inst[0]) ** 2 + (
				inst[1] - prev_inst[1]) ** 2)
			else:
				dist_change = 0
			inst.append(dist_change)

			path.append(inst)

			prev_inst = inst

		self._path_points = path
		self._set_inter_target(0)
		return path



	def visualize(self, path_points) -> None:
		"""Draws a visualization of the path in the scene.

		The visualization can be removed
		with :py:meth:`ConfigurationPath.clear_visualization`.
		"""
		if len(self._path_points) <= 0:
			raise RuntimeError("Can't visualise a path with no points.")

		
	  
		self._drawing_handle = vrep.simAddDrawingObject(
			objectType=vrep.sim_drawing_lines, size=3, duplicateTolerance=0,
			parentObjectHandle=-1, maxItemCount=99999,
			ambient_diffuse=[1, 0, 1])
		vrep.simAddDrawingObjectItem(self._drawing_handle, None)
		init_pose = self.get_2d_pose()
		self.set_2d_pose(self._path_points[0][:3])
		prev_point = self.get_position()

		for i in range(len(self._path_points)):
			points = self._path_points[i]
			self.set_2d_pose(points[:3])
			p = self.get_position()
			vrep.simAddDrawingObjectItem(self._drawing_handle, prev_point + p)
			prev_point = p

		# Set the arm back to the initial config
		self.set_2d_pose(init_pose[:3])


	def get_base_actuation(self):
		"""A controller using PID.

		:return: A list with left and right joint velocity, and bool if target is reached.
		"""

		d_x, d_y, _ = self.intermediate_target_base.get_position(
			relative_to=self)
		#print('dx and dy')
		print(d_x)
		print(d_y)
		d_x_final, d_y_final, _ = self.target_base.get_position(
			relative_to=self)

		if sqrt((d_x_final) ** 2 + (d_y_final) ** 2) < 1.0:
			return [0., 0.]

		alpha = atan2(d_y, d_x)
		e = atan2(sin(alpha), cos(alpha))
		e_P = e
		e_I = self.cummulative_error + e
		e_D = e - self.prev_error
		w = self.Kp * e_P + self.Ki * e_I + self.Kd * e_D
		w = atan2(sin(w), cos(w))

		self.cummulative_error = self.cummulative_error + e
		self.prev_error = e

		return [0.1, 0]




	def step(self) -> bool:
		#print(self._path_points[0][0], self._path_points[0][1])
		pos_inter = self.intermediate_target_base.get_position(
			relative_to=None)
		vrep.simAddDrawingObjectItem(self.drawing_handle, pos_inter[:2] + [0.0])
		pos_inter = self.intermediate_target_base.get_position(
			relative_to=self)
		#print(pos_inter)
		

		if self.inter_done:
			self._next_i_path()
			self._set_inter_target(self.i_path)
			self.inter_done = False

		if sqrt((pos_inter[0]) ** 2 + (pos_inter[1]) ** 2) < 0.1:
			self.inter_done = True
			#actuation, _ = self.get_base_actuation()
		#else:
			#actuation, _ = self.get_base_actuation()

			#self._mobile.set_joint_target_velocities(actuation)

		#if self.i_path == len(self._path_points) - 1:
		#	self._path_done = True
		
		
		#actuation, self._path_done = self._mobile.get_base_actuation()
		#self._mobile.set_joint_target_velocities(actuation)

		return self.get_base_actuation(), False


	def _set_inter_target(self, i):
		self.intermediate_target_base.set_position(
			[self._path_points[i][0], self._path_points[i][1],
			 self.target_z])
		self.intermediate_target_base.set_orientation(
			[0, 0, self._path_points[i][2]])


	def _next_i_path(self):
		incr = 0.01
		dist_to_next = 0
		while dist_to_next < incr:
			self.i_path += 1
			if self.i_path == len(self._path_points) - 1:
				self.i_path = len(self._path_points) - 1
				break
			dist_to_next += self._path_points[self.i_path][-1]