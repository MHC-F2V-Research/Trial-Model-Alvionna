from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# blank plots
fig = plt.figure()
ax = plt.axes(projection="3d")

# creating the x,y,z functions/points
def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

#linspace -> return an evenly spaced numbers over an interval
x = np.linspace(-6, 6, 30) #30 numbers over the interval -6 to 6 (inclusive)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

'''
using the angle:
x-direction vector:
    - x-coordinate: magnitude * cos(angle)
    - y-coordinate: magnitude * sin(angle)
y-direction vector:
    - x-coordinate: magnitude * cos(angle)
    - y-coordinate: magnitude * sin(angle)
performs a vector addition to find the net force
convert the vector (from addition) to angle -> angle = tan^(-1)(y/x) (arctan y/x)
apply the equation v = sqrt(x^2 + y^2) (x,y from the vector addition)
'''

# the wireframe green plots
fig = plt.figure()
ax = plt.axes(projection="3d")
# ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.show()

# the surface plots
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
ax.set_title('surface');

plt.show()
