from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.backend import vrep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.misc.distance import Distance
from pyrep.const import JointMode
import numpy as np
import tensorflow as tf
from math import sqrt, atan2, sin, cos, exp
import gym
from gym import spaces

from PIL import Image

MIN_DISTANCE = 0.1
R_COLLISION = -50
REWARD_CONST = 50
TARGET_POS_MIN, TARGET_POS_MAX = [1.0, 1.0], [4.0, 4.0]
#TARGET_POS_MIN, TARGET_POS_MAX = [-1.5, 2.5], [1.5, 4.5]

class NavigationEnv(gym.Env):
	
	def __init__(self, scene_file, robot_class, *args, **kwargs):
		self.pr = PyRep()
		self.pr.launch(scene_file, headless=False)
		self.pr.start()
		#self.agent = PioneerP3dx(0, 2, 16, 'Pioneer_p3dx', 'ultrasonicSensor', 'fastHokuyo')
		self.agent = robot_class(*args, **kwargs)
		self.agent.set_motor_locked_at_zero_velocity(True)
		self.agent.set_joint_mode(JointMode.FORCE)
		self.distance = Distance('Distance')
		self.starting_pose = self.agent.get_2d_pose()
		self.target = Shape.create(type=PrimitiveShape.SPHERE,
					  size=[0.1, 0.1, 0.1],
					  color=[1.0, 0.1, 0.1],
					  static=True, respondable=False)
		#high = np.array([np.inf] * 1084)
		high = np.array([np.inf] * 34)
		self.action_space = spaces.Box(np.array([0, -0.7]), np.array([0.7, 0.7]), dtype=np.float32)
		#self.observation_space = spaces.Box(shape=(37,), dtype=np.float32)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

		self.obstacle_ranges = [[(-1.8, -2.5),(1.8, -1.5)],
								[(-1.8, 1.5), (1.8, 2.5)],
								[(2.4, -1.8),(3.6, 1.8)],
								[(-3.6, -1.8),(2.4, 1.8)]]

		#self.map_sensor = VisionSensor("mapSensor")
	def shutdown(self):
		self.pr.stop()
		self.pr.shutdown()

	def step(self, action):
		self.agent.set_base_angular_velocites(action)  # Execute action 
		self.pr.step()  # Step the physics simulation

		target_pos = self.target.get_position(relative_to=self.agent)
		target_dist = np.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
	
		robot_position = self.agent.get_2d_pose()
		d = self.distance.read()
		if d < MIN_DISTANCE:
			print("collision")
			return self._get_state(), R_COLLISION, True, {'robot_position': robot_position}

		if target_dist < 0.5:
			return self._get_state(), 150, True, {'robot_position': robot_position}
	
		reward =  REWARD_CONST * (self.old_distance - target_dist)
		#reward -= 3.0*np.exp(-4*d)
		
		self.old_distance = target_dist
		#print(reward)
		return self._get_state(), reward, False, {'robot_position': robot_position}


	def check_collision(self, target_pos):
		for obstacle_range in self.obstacle_ranges:
			x_in_obstacle_range = target_pos[0] > obstacle_range[0][0] and target_pos[0] < obstacle_range[1][0]
			y_in_obstacle_range = target_pos[1] > obstacle_range[0][1] and target_pos[1] < obstacle_range[1][1]
			if (x_in_obstacle_range and y_in_obstacle_range):
				return True
		return False

	def get_target_pos(self):
		target_pos = list(np.random.uniform(TARGET_POS_MIN, TARGET_POS_MAX))
		if self.check_collision(target_pos):
			return [2.0, 0.0]		
		else:
			return target_pos


	def reset(self):
		target_pos = list(np.random.uniform(TARGET_POS_MIN, TARGET_POS_MAX))
		angle = np.random.uniform(-np.pi, np.pi)
		#print(angle)
		target_pos = self.get_target_pos()
		if np.random.random() < 0.5:
			target_pos[0] = -target_pos[0]
		
		if np.random.random() < 0.5:
			target_pos[1] = -target_pos[1]
		
		target_pos = target_pos + [0.025]
		#target_pos = [0.65, 3.3] + [0.025]

		self.agent.set_2d_pose([0, 0, angle])
		self.target.set_position(target_pos)
		
		
		self.agent.set_joint_target_velocities([0,0])
		self.pr.step()
		#self.pr.step()
		target_pos = self.target.get_position(relative_to=self.agent)
		target_dist = np.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
		self.old_distance = target_dist
		return self._get_state()


	def _get_state(self):
		#sensor_datas = self.agent.get_sensor_data()
		sensor_datas = self.agent.get_laser_data() # len(sensor_data) = 1080
		n = 36
		sensor_datas = sensor_datas = [min(sensor_datas[i:i + n]) for i in range(0, len(sensor_datas), n)]
		target_pos = self.target.get_position(relative_to=self.agent)[:2]
		#dist = sqrt(target_pos[0]**2 + target_pos[1]**2)
		#angle = atan2(target_pos[1], target_pos[0])
		#print(target_pos)
		velocity = self.agent.get_velocity()
		return np.array(sensor_datas + target_pos + velocity) # shape (1084, )


