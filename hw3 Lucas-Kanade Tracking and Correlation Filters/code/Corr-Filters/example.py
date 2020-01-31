import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.ndimage import correlate,convolve


img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])


# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5


def init():
    return [cropax, patch, all_patchax]


def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    if i < N:  # If the animation is still running
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]
    else:  # Stuff to do after the animation ends
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                          Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

        # Place your solution code for question 4.3 here
        lamb=1
        S=X.dot(X.T)
        g1=S+lamb*np.eye(N)
        g1=np.linalg.inv(g1).dot(X)
        g1=g1.dot(Y)
        g1=g1.reshape([dsize[0],dsize[1]])
        corr1=correlate(img,g1)
        conv1=convolve(img,g1)
        lamb=0
        g2=S+lamb*np.eye(N)
        g2=np.linalg.inv(g2).dot(X)
        g2=g2.dot(Y)
        g2=g2.reshape([dsize[0],dsize[1]])
        corr2=correlate(img,g2)
        conv2=convolve(img,g2)

        plt.subplot(1,2,1),plt.imshow(g1,'gray'),plt.title('Lambda=1')
        plt.subplot(1,2,2),plt.imshow(g2,'gray'),plt.title('Lambda=0')
        plt.show()
        plt.subplot(1,2,1),plt.imshow(corr1,'gray'),plt.title('Lambda=1,Correlation result')
        plt.subplot(1,2,2),plt.imshow(corr2,'gray'),plt.title('Lambda=0,Correlation result')
        plt.show()

        plt.subplot(1,2,1),plt.imshow(conv1,'gray'),plt.title('Lambda=1,Convolve result')
        plt.subplot(1,2,2),plt.imshow(conv2,'gray'),plt.title('Lambda=0,Convolve result')
        plt.show()

        conv1f=convolve(np.flip(img),g1)
        conv2f=convolve(np.flip(img),g2)
        plt.subplot(1,2,1),plt.imshow(np.flip(conv1f),'gray'),plt.title('Lambda=1,Convolve result,flipped')
        plt.subplot(1,2,2),plt.imshow(np.flip(conv2f),'gray'),plt.title('Lambda=0,Convolve result,flipped')
        plt.show()

        return []

# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=1)
plt.show()
