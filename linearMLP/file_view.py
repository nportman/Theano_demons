# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:03:37 2015

@author: sentry22
"""

import numpy as np
data_file='/Users/sentry22/UOIT_research/energy_9.06748664'
class CHGCAR:
    def __init__(self,filename,file_type='CHGCAR'):
        """ file_type=<CHGCAR|CHG>  """
        f = open(filename)
        for n, line in enumerate(f):
            if n==0:        self.comment = line.replace("\n","").strip()
            elif n==1:      self.scaling = float(line)
            elif n==2:      self.a1 = [float(i) for i in line.split() ]
            elif n==3:      self.a2 = [float(i) for i in line.split() ]
            elif n==4:      self.a3 = [float(i) for i in line.split() ]
            elif n==5:      self.symbols = line.split()
            elif n==6:      self.counts = np.array([ int(i) for i in line.split()])
            elif n==7:      self.coordinate_format = line[0]
            elif n==(7+np.sum(self.counts)+2): 
                self.dimensions = [int(i) for i in line.split()  ]          
        self.totallines = n
        f.close()
        self.name =""
        headlen = 8 + np.sum(self.counts) + 2
        if file_type=="CHGCAR":
            bodylen = int(np.ceil(np.product(self.dimensions) / 5)) #chgcar has 5 columns
        elif file_type=="CHG":
            bodylen = int(np.ceil(np.product(self.dimensions) / 10)) #chg has 10 columns
        footlen = self.totallines - headlen - bodylen + 1
        self.data = np.genfromtxt(filename,skip_header=headlen, skip_footer=footlen)
        self.data = np.resize(self.data,self.dimensions)
        #the .data attribute now contains the 3D data in an NxNxN array.
dat1=CHGCAR(data_file)            
#_______________________________________________________    
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.animation as animation
from matplotlib.widgets import Slider
      
charge_d=dat1.data
charge_d_norm=mcolors.Normalize(vmin=np.min(charge_d),vmax=np.max(charge_d),clip=False)
norm_data=charge_d_norm(charge_d)
new_map=cm.get_cmap(name='hot',lut=2048)
#new_map=mcolors.Colormap('hot', N=1024)
#norm_data_mapped=colors_quant(norm_data)
m=cm.ScalarMappable(norm=charge_d_norm,cmap=new_map)

#color_coded=m.to_rgba(charge_d)

#do slice by slice in z direction
frames=np.zeros((300,300,300))
for z in range(300):
    for y in range(300):
        for x in range(300):
            frames[x,y,z]=norm_data[x][y][z]
            

img=frames[299,:,:]
plt.imshow(img,interpolation='none',cmap=new_map)
plt.colorbar()
plt.title("E=9.06748664, Y-Z planar view")
plt.savefig('sliceyz300_9.06748664.png',bbox_inches='tight')

fig=plt.figure()

def initz():
    Matrix=frames[:,:,0]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()
    l.set_data(Matrix)
    return l
    
def updatefigz(i):
    fig.clf()
    Matrix=frames[:,:,i]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()  
    l.set_data(Matrix)
    return l
    
anim = animation.FuncAnimation(fig, updatefigz, np.arange(1,300), init_func=initz,interval=30, blit=True)
plt.show()

def inity():
    Matrix=frames[:,0,:]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()
    l.set_data(Matrix)
    return l
    
def updatefigy(i):
    fig.clf()
    Matrix=frames[:,i,:]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()  
    l.set_data(Matrix)
    return l
    
anim = animation.FuncAnimation(fig, updatefigy, np.arange(1,300), init_func=inity,interval=30, blit=True)
plt.show()

def initx():
    Matrix=frames[0,:,:]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()
    l.set_data(Matrix)
    return l
    
def updatefigx(i):
    fig.clf()
    Matrix=frames[i,:,:]
    l=plt.imshow(Matrix,interpolation='none',cmap=new_map)
    plt.colorbar()  
    l.set_data(Matrix)
    return l
        
anim = animation.FuncAnimation(fig, updatefigx, np.arange(1,300), init_func=initx, interval=30, blit=True)
plt.show()
#anim.save("testx.mp4", fps=10)         
            