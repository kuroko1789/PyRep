import numpy as np
import tensorflow as tf
from model import policy_net, q_net
class ReplayBuffer:
	"""
	A simple FIFO experience replay buffer for TD3 agents.
	"""

	def __init__(self, obs_dim, act_dim, size):
		self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
		self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
		self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
		self.rews_buf = np.zeros(size, dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.ptr, self.size, self.max_size = 0, 0, size

	def store(self, obs, act, rew, next_obs, done):
		self.obs1_buf[self.ptr] = obs
		self.obs2_buf[self.ptr] = next_obs
		self.acts_buf[self.ptr] = act
		self.rews_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.ptr = (self.ptr+1) % self.max_size
		self.size = min(self.size+1, self.max_size)

	def sample_batch(self, batch_size=32):
		idxs = np.random.randint(0, self.size, size=batch_size)
		return dict(obs1=self.obs1_buf[idxs],
					obs2=self.obs2_buf[idxs],
					acts=self.acts_buf[idxs],
					rews=self.rews_buf[idxs],
					done=self.done_buf[idxs])

obs_dim = 37
act_dim = 2
act_limit = 1.0

def get_vars(scope):
	return [x for x in tf.global_variables() if scope in x.name]

def td3(env, steps_per_epoch=5000, epochs=1000, replay_size=int(1e6), gamma=0.95, 
		polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=50000, 
		act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, max_ep_len=300):
	
	episode = 0
	
	replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
	
	x_ph = tf.placeholder(tf.float32, [None, obs_dim])
	x2_ph = tf.placeholder(tf.float32, [None, obs_dim])
	a_ph = tf.placeholder(tf.float32, [None, act_dim])
	r_ph = tf.placeholder(tf.float32, [None])
	d_ph = tf.placeholder(tf.float32, [None])
	
	with tf.variable_scope('main'):
		with tf.variable_scope('pi'):
			pi = policy_net(x_ph, act_dim)
		with tf.variable_scope('q1'):
			q1 = q_net(x_ph, a_ph)
		with tf.variable_scope('q2'):
			q2 = q_net(x_ph, a_ph)
		with tf.variable_scope('q1', reuse=True):
			q1_pi = q_net(x_ph, pi)
	
	# Target policy network
	with tf.variable_scope('target'):
		with tf.variable_scope('pi'):
			pi_targ = policy_net(x2_ph, act_dim)
		
	
	# Target Q networks
	with tf.variable_scope('target'):

		# Target policy smoothing, by adding clipped noise to target actions
		noise = tf.random.normal(tf.shape(pi_targ), stddev=target_noise)
		a2 = pi_targ + tf.clip_by_value(noise, -noise_clip, noise_clip)
		#a2 = tf.clip_by_value(a2, -act_limit, act_limit)
		a2 = tf.clip_by_value(a2, [0,-1.0], [1.0,1.0])
		# Target Q-values, using action from smoothed target policy
		with tf.variable_scope('q1'):
			q1_targ = q_net(x2_ph, a2)
		with tf.variable_scope('q2'):
			q2_targ = q_net(x2_ph, a2)

	# Experience buffer
	replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

	# Bellman backup for Q functions, using Clipped Double-Q targets
	min_q_target = tf.math.minimum(q1_targ, q2_targ)
	backup = r_ph + gamma * (1 - d_ph) * min_q_target
	backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_target) # do I need tf.stop_gradient?
	# TD3 losses
   
	pi_loss = -tf.reduce_mean(q1_pi)
	q1_loss = tf.losses.mean_squared_error(q1, backup) 
	q2_loss = tf.losses.mean_squared_error(q2, backup)
	q_loss = q1_loss + q2_loss 
	
	pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
	q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
	train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
	train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

	target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
							  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

	# Initializing targets to match main variables
	target_init = tf.group([tf.assign(v_targ, v_main)
							  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(target_init)


	def get_action(o, noise_scale): # o.shape (1, obs_dim)
		a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0] # a.shape (act_dim,)
		#print(a)
		a += noise_scale * np.random.randn(act_dim)
		return np.clip(a, [0, -1.0], [1.0, 1.0])
	
	ep_len = 0 
	ep_ret = 0
	o = env.reset()     # reset the environment    
	 
	
	total_steps = steps_per_epoch * epochs
	rewards = []
	for t in range(total_steps):
		if t > start_steps:
			a = get_action(o, act_noise)
		else:
			a = np.random.uniform([0.0, -1.0],[1.0, 1.0])
			#a = env.action_space.sample()
 
		#print(a)
		o2, reward, done = env.step(a)
		rewards.append(reward)
		#print(o2)
		#print(reward)
		#print(done)
		ep_len += 1
		ep_ret += reward
		
		done = False if ep_len==max_ep_len else done
		replay_buffer.store(o, a, reward, o2, done)
		
		o = o2

		if done or (ep_len == max_ep_len):
			
			for k in range(len(rewards)):
				discounts = [gamma**i for i in range(len(rewards[k:])+1)]
				#print(sum([a*b for a,b in zip(discounts, rewards[k:])]))
			for j in range(ep_len):
				#print(j)
				batch = replay_buffer.sample_batch(batch_size)
				# x_ph (batch_size, obs_dim)
				# x2_ph (batch_size, obs_dim)
				# a_ph (batch_size, act_dim)
				# r_ph (batch_size,)
				#print(batch['obs1'])
				#print(batch['obs2'])
				#print(batch['acts'])
				#print(batch['rews'])
				#print(batch['done'])
				feed_dict = {x_ph: batch['obs1'],
							 x2_ph: batch['obs2'],
							 a_ph: batch['acts'],
							 r_ph: batch['rews'],
							 d_ph: batch['done']}
				q_ops = [q_loss, train_q_op]
				outputs = sess.run(q_ops, feed_dict)
			
				if j % policy_delay == 0:
					outputs = sess.run([train_pi_op, target_update], feed_dict)
			episode += 1
			rewards = []
			print("episode: ", episode, ep_ret)
			print("===================================")
			ep_len = 0
			ep_ret = 0
			o = env.reset()    # reset the environment    
			