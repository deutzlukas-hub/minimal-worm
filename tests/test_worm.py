import numpy as np
from fenics import *
from minimal_worm.util import f2n
from minimal_worm import Frame
from ufl import atan_2
from minimal_worm import Worm
from minimal_worm.model_parameters import DMP

def test_finite_backwards_difference():
	'''	
	Tests coefficients for finite backwards difference of nth 
	derivative of order k. Solutions for weighting coefficents 
	are taken from

	from https://web.media.mit.edu/~crtaylor/calculator.html
	'''	

	# Dictionary keys represent derivative and order (n,k)
	solution = {}
	solution[(1, 1)] = np.array([-1, 1])
	solution[(1, 2)] = np.array([1, -4, 3]) / 2
	solution[(1, 3)] = np.array([-2, 9, -18, 11]) / 6
	solution[(1, 4)] = np.array([3, -16, 36, -48, 25]) / 12

	solution[(2, 1)] = np.array([1, -2, 1]) 
	solution[(2, 2)] = np.array([-1, 4, -5, 2])
	solution[(2, 3)] = np.array([11, -56, 114, -104, 35]) / 12
	solution[(2, 4)] = np.array([-10, 61, -156, 214, -154, 45]) / 12

	worm = Worm(100, 0.01)

	# Calculate coefficients twice to check if caching works
	for _ in range(2):	
		# Approximate nth derivative	
		for n in [1,2]:
			# kth order derivative
			for k in np.arange(1, 4.01, dtype = int):								
				c_arr, s_arr = worm._finite_difference_coefficients(n, k)

				assert np.allclose(c_arr, solution[n,k]), f"n={n}, k={k}" 

	print('Passed test: Finite backwards difference coefficients \n' 
		'are calculate correctly!')

def test_assign_initial_configuration():
	'''
	Test if initial centreline coordinates and Euler angles
	are assigned correctly
	'''

	N = 100
	worm = Worm(N, 0.01)
	worm._assign_initial_values()
	s_arr = np.linspace(0,1,N)

	# Default configuration
	r0_arr = np.zeros((3, N))
	r0_arr[2, :] = s_arr
	theta0_arr = np.zeros((3, N))

	for u in worm.u_old_arr:

		r, theta = split(u)
		r, theta = project(r, worm.V3), project(theta, worm.V3) 

		assert np.allclose(r0_arr, f2n(r))
		assert np.allclose(theta0_arr, f2n(theta))

	print('Passed test: Default initial configuration \n' 
		'is assigned correctly!')

	return

def test_zero_control():

	worm = Worm(100, 0.01)
	
	CS = {} 
	CS['k'] = Constant((0, 0, 0))
	CS['sig'] = Constant((0, 0, 0))
	
	
	MP = DMP.dimless_from_physical_parameters(mu = 1e0)
		
	FK = ['r', 'd1', 'd2', 'd3', 'theta']
	
	FS, _ = worm.solve(2.0, MP, CS, FK=FK) 
				
	e1 =  np.tile(np.array([1, 0, 0])[:, None], (1, worm.N))
	e2 =  np.tile(np.array([0, 1, 0])[:, None], (1, worm.N))
	e3 =  np.tile(np.array([0, 0, 1])[:, None], (1, worm.N))
	
	r0 = np.zeros((3, worm.N))
	r0[2, :] = np.linspace(0, 1, worm.N) 
	theta0 = 0
	
	atol = 1e-4
	
	for i in range(len(FS)):

		assert np.allclose(FS.r[i, :, :], r0, atol = atol) 
		assert np.allclose(FS.theta[i, :, :], theta0, atol = atol)
		assert np.allclose(FS.d1[i, :, :], e1, atol = atol)
		assert np.allclose(FS.d2[i, :, :], e2, atol = atol)
		assert np.allclose(FS.d3[i, :, :], e3, atol = atol)
	
	print('Passed test: Zero control')
				
	return

def test_constant_control():

	worm = Worm(250, 0.01)
		
	#Semicircle
	CS = {} 
		
	k1_pref = np.pi
	CS['k'] = Constant((k1_pref, 0, 0))
	CS['sig'] = Constant((0, 0, 0))

	MP = DMP.dimless_from_physical_parameters(mu = 1e0)
	
	FK = ['r', 'theta', 'k', 'sig']
	
	FS, CS, _ = worm.solve(1.0, MP, CS, FK=FK) 
			
	k_pref = np.zeros((3, worm.N))
	k_pref[0, :] = k1_pref
	
	atol = 1e-3
	
	import matplotlib.pyplot as plt
	from minimal_worm.plot import plot_scalar_field
	
	gs = plt.GridSpec(2, 1)
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])

	plot_scalar_field(ax0, CS['k'][:, 0, :])
	plot_scalar_field(ax1, FS.k[:, 0, :])
	
	plt.show()
		
	assert np.allclose(FS.k[-1, :], k1_pref, atol = 1e-3) 
	
	return

if __name__ == '__main__':

	#test_finite_backwards_difference()
	#test_assign_initial_configuration()
	#test_zero_control()	
	test_constant_control()

	
	#test_Euler_angle_numpy()


# def test_Q():
# 	'''
# 	Test rotation matrix	
#
# 	Let's define Z is yaw y as pitch and x as roll.
#
# 	'''
# 	pass
#
# def test_assign_initial_configuration():
# 	'''
# 	Test if initial centreline coordinates and Euler angles
# 	are assigned correctly
# 	'''
#
# 	N = 100
# 	worm = Worm(N, 0.01)
# 	worm._assign_initial_values()
# 	s_arr = np.linspace(0,1,N)
#
# 	# Default configuration
# 	r0_arr = np.zeros((3, N))
# 	r0_arr[2, :] = s_arr
# 	theta0_arr = np.zeros((3, N))
#
# 	for u in worm.u_old_arr:
#
# 		r, theta = split(u)
# 		r, theta = project(r, worm.V3), project(theta, worm.V3) 
#
# 		assert np.allclose(r0_arr, f2n(r))
# 		assert np.allclose(theta0_arr, f2n(theta))
#
# 	return
#
#
#  # R = 0.5*np.pi
#  # # Circular centreline in xz-plane
#  # r0_arr[0, :] = R*np.cos(2*np.pi*s_arr)
#  # r0_arr[2, :] = R*np.sin(2*np.pi*s_arr)
#  #
#  # r = Expression(('R*cos(2*pi*x[0])', '0', 'R*sin(2*pi*x[0])'), 
#  # 	R = R, degree=worm.fe['degree'])	
#  #
#  # # Circular centreline in yz-plane
#  # r = Expression(('0', 'R*cos(2*pi*x[0])', 'R*sin(2*pi*x[0])'), 
#  # 	R = R, degree=worm.fe['degree'])	
#  #
#  # r0_arr = np.zeros((3, N))
#  # r0_arr[1, :] = R * np.cos(2*np.pi*s_arr) 
#  # r0_arr[2, :] = R * np.sin(2*np.pi*s_arr) 
#  #
#  # # Body frame vectors
#  # d3 = Expression(('0', '-sin(2*pi*x[0])', 'cos(2*pi*x[0])'), degree=worm.fe['degree'])
#  # d1 = Expression(('1', '0', '0'), degree=worm.fe['degree'])
#  # d2 = cross(d3, d1)
#  #
#  #
#  # # Lab frame 
#  # e1 = Constant((1, 0, 0))
#  # e2 = Constant((0, 1, 0))
#  # e3 = Constant((0, 0, 1))
#  #
#  # Q = outer(e1, d1) +  outer(e2, d2) + outer(e3, d3)
#  #
#  # Q11, Q12, Q13 = Q[0,0], Q[0,1], Q[0,2]
#  # Q21 = Q[1,0]
#  # Q31, Q32, Q33 = Q[2, 0], Q[2, 1], Q[2,2]
#  #
#  # #atan2 is not defined as a standalone function in fenics
#  # gamma = atan_2(Q21, Q11)
#  # beta = atan_2(-Q31, sqrt(Q32**2  + Q33**2))
#  # alpha = atan_2(Q32, Q33)	
#  #
#  # theta = alpha * e1 + beta * e2 + gamma * e3
#  #
#  # theta0_arr = f2n(project(theta, worm.V3))
#  #
#  # F0 = Frame(r=f2n(project(r, worm.V3)), theta=theta0_arr)
#  #
#  # worm._assign_initial_values(F0)
#  #
#  # for u in worm.u_old_arr:
#  #
#  # 	r, theta = split(u)
#  # 	r, theta = project(r, worm.V3), project(theta, worm.V3) 
#  #
#  # 	assert np.allclose(r0_arr, f2n(r))
#  # 	assert np.allclose(theta0_arr, f2n(theta))
#  #
#  # print('Passed test: Initial centreline coordinates and Euler angles ' 
#  # 	'are assigned correctly!')
#
# if __name__ == '__main__':
#
# 	#test_finite_backwards_difference()
# 	#test_assign_initial_configuration()
# 	#test_Euler_angle_numpy()
#
# # def test_Euler_angles():
# #
# # 	N = 100
# # 	s = np.linspace(0, 1, N)
# #
# # 	d3 = np.zeros((N, 3))
# # 	d3[:, 0] = -np.sin(2*np.pi*s)
# # 	d3[:, 2] =  np.cos(2*np.pi*s)	
# # 	d2 = np.zeros_like(d3)
# # 	d2[:, 1] = 1	
# # 	d1 = np.cross(d2, d3)
# #
# # 	Q = np.dstack((d1, d2, d3))
# #
# # 	alpha = np.arctan2(Q[:,1,0], Q[:,0,0])
# # 	beta =  np.arcsin(-Q[:,2,0])
# # 	gamma = np.arctan2(Q[:,2,1], Q[:,2,2])
# #
# # 	plt.plot(s, beta)
# # 	plt.plot(s, alpha)
# # 	plt.plot(s, gamma)
# #
# # 	plt.show()
# #
# # 	N = 100
# # 	worm = Worm(N, 0.01)	
# # 	s_arr = np.linspace(0,1,N)	
# # 	R = 2*np.pi
# #
# # 	# Lab frame 
# # 	e1 = Constant((1, 0, 0))
# # 	e2 = Constant((0, 1, 0))
# # 	e3 = Constant((0, 0, 1))
# #
# # 	# Body frame vectors for circle in xz-plane
# # 	d3 = Expression(('-sin(2*pi*x[0])', '0', 'cos(2*pi*x[0])'), degree=worm.fe['degree'])
# # 	d2 = Expression(('0', '1', '0'), degree=worm.fe['degree'])
# # 	d1 = cross(d2, d3)
# #
# # 	Q_xz = outer(d1, e1) +  outer(d2, e2) + outer(d3, e3)
# #
# # 	# Body frame vectors for circle in xy-plane
# # 	d3 = Expression(('0', '-sin(2*pi*x[0])', 'cos(2*pi*x[0])'), degree=worm.fe['degree'])
# # 	d1 = Expression(('1', '0', '0'), degree=worm.fe['degree'])
# # 	d2 = cross(d3, d1)
# #
# # 	Q_yz = outer(d1, e1) +  outer(d2, e2) + outer(d3, e3)
# #
# # 	for Q in [Q_xz, Q_yz]:
# #
# # 		Q11 = Q[0,0]
# # 		Q21 = Q[1,0]
# # 		Q31, Q32, Q33 = Q[2, 0], Q[2, 1], Q[2,2]
# #
# # 		#atan2 is not defined as a standalone function in fenics
# # 		gamma = atan_2(Q21, Q11)
# # 		beta =  asin(-Q31)
# # 		alpha = atan_2(Q32, Q33)	
# #
# # 		alpha_arr = f2n(project(alpha, worm.V))
# # 		beta_arr = f2n(project(beta, worm.V))
# # 		gamma_arr = f2n(project(gamma, worm.V))
# #
# # 		pass
# #
# #
# # 	return
#
#
