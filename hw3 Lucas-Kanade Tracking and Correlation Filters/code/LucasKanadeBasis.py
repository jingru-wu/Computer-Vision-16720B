import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt
import skimage.io
import scipy
import math

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    # initialize parameters
    threshold=0.01
    tre_p=10000
    p0=np.array([0,0])

    range_y=np.arange(It.shape[0])
    range_x=np.arange(It.shape[1])
    linspace_x=np.linspace(rect[0],rect[2],round(rect[2]-rect[0]+1))
    linspace_y=np.linspace(rect[1],rect[3],round(rect[3]-rect[1]+1))
    mesh_x,mesh_y=np.meshgrid(linspace_x,linspace_y)
    interpolate=RectBivariateSpline(range_y,range_x,It)
    template=interpolate.ev(mesh_y,mesh_x)

    gradient= np.asarray(np.gradient(template))
    X_gradient=gradient[1].flatten()
    Y_gradient=gradient[0].flatten()
    T_gradient=np.transpose(np.array([X_gradient,Y_gradient]))

    SD=np.zeros([len(X_gradient),2])
    sd=np.array(T_gradient)   # SD:N*2
    for i in range(bases.shape[2]):
        A=np.array([bases[:,:,i].flatten()])
        SD=SD+sd-np.dot(A.T,A).dot(sd)
    H=SD.T.dot(SD)
    #  Hessian matrix: 2*2
    i=0
    while tre_p>threshold:
        rect1=rect+np.array([p0[0],p0[1],p0[0],p0[1]])
        range_y_w=np.arange(It1.shape[0])
        range_x_w=np.arange(It1.shape[1])
        linspace_x_w=np.linspace(rect1[0],rect1[2],round(rect1[2]-rect1[0]+1))
        linspace_y_w=np.linspace(rect1[1],rect1[3],round(rect1[3]-rect1[1]+1))

        mesh_x_w,mesh_y_w=np.meshgrid(linspace_x_w,linspace_y_w)
        interpolate=RectBivariateSpline(range_y_w,range_x_w,It1)
        template_w=interpolate.ev(mesh_y_w,mesh_x_w)

        Error=template-template_w
        Error=Error.flatten()   #N*1
        B= np.dot(np.transpose(SD),Error)    # 2*1 = 2*N * N*1
        delta_p =np.dot(np.linalg.inv(H),B)  # delta p: 2*1[ px ; py]
        p0 = p0+delta_p
        tre_p = np.linalg.norm(delta_p)
        i=i+1
    p = p0
    return p
    return np.zeros(2)
