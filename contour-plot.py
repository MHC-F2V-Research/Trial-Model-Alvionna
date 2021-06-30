import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

#for the depth -- there's a function, if not we can just assign it manually like x and y
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40) #return an evenly spaced interval from 0 - 5 with 50 numbers

X, Y = np.meshgrid(x, y) #return coordinate matrices from coordinate vectors.
Z = f(X, Y)

#default is below - negative values represented with dashed lines
# plt.contour(X, Y, Z, colors='black');
#
# #color coded plots with cmap and we want more lines to be drawn - 20 equally spaced intervals within the data range
# #RdGy = red gray
# plt.contour(X, Y, Z, 20, cmap='RdGy');
#
# #filled contour plot with contourf and add a color bar with labeled color information
# plt.contourf(X, Y, Z, 20, cmap='RdGy')
# plt.colorbar();
#
# #the color above could look "splotchy" (like discrete more than continuous color )
# plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy') #origin -> by default is top left (unlike most contour plot)
# plt.colorbar()
# plt.axis(aspect='image'); #bcs imshow automatically adjust the axis aspect ratio to match input data, we can change it using this function

#final code (combining contour, contourf and imshow)
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();
plt.show();

# ---- OR -----

# fig,ax=plt.subplots(1,1)
# cp = ax.contourf(X, Y, Z) # the contour plot
# fig.colorbar(cp) # Add a colorbar to a plot
# ax.set_title('Filled Contours Plot')
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('y (cm)')
# plt.show()
