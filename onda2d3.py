import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Dominio
"""

Lx = 10;
Ly = 10;
dx = 0.1;
dy = dx;
nx = np.int(Lx/dx);
ny = np.int(Ly/dy);
x = np.arange(0,Lx,dx);
y = np.arange(0,Ly,dy);

x, y=np.meshgrid(x,y);

T=40;

"""
Condiciones Iniciales
"""
m = np.multiply(1,np.ones(ny));
#m[30:60] = np.linspace(.5,1.2,ny-30);
m[40:60] = .5;
c = np.transpose(np.tile(m,(nx,1)));

dt = .05;

C = np.multiply(dt/dy,c);
#C=np.multiply(0.5,np.ones((nx,ny)))

t = 0.;
cnt = 1;

wn = np.zeros((ny,nx,np.int(T/dt)+1));
wnp1 = np.copy(wn[:,:,0]);

"""
"""
#N=50;
#h = np.linspace(-5,5,N)
#h = np.multiply(0.2,np.sinc(h));
#h = np.hanning(N);
"""
"""

while(t<T-dt):

#   PAREDES ABSORBENTES    
    wnp1[0,:]=np.sum([wn[1,:,cnt-1],
                     np.multiply(
                             np.divide((C[1,:]-1),(C[1,:]+1)),
                                 np.subtract(wnp1[1,:],
                                             wn[0,:,cnt-1]))],
                                 axis=0);    
    wnp1[-1,:]=np.sum([wn[-2,:,cnt-1],
                      np.multiply(
                              np.divide((C[-2,:]-1),(C[-2,:]+1)),
                                  np.subtract(wnp1[-2,:],
                                              wn[-1,:,cnt-1]))],
                                  axis=0);
    wnp1[:,0]=np.sum([wn[:,1,cnt-1],
                     np.multiply(
                             np.divide((C[:,1]-1),(C[:,1]+1)),
                                 np.subtract(wnp1[:,1],
                                             wn[:,0,cnt-1]))],
                                 axis=0);
    wnp1[:,-1]=np.sum([wn[:,-2,cnt-1],
                      np.multiply(
                              np.divide((C[:,-2]-1),(C[:,-2]+1)),
                                  np.subtract(wnp1[:,-2],
                                              wn[:,-1,cnt-1]))],
                                  axis=0);
    
    wn[:,:,cnt]=np.copy(wnp1);    
    
    wn[10,10,cnt]=1.5*np.sin(3*np.pi*t);

#    DOBLE SLITS
#    wn[45:55,3,cnt]=dt**2*300*np.sin(3*np.pi*t)
#    wn[:45,20,cnt]=0;
#    wn[46:54,20,cnt]=0;
#    wn[55:,20,cnt]=0;

    for i in range(1,ny-1):
        for j in range(1,nx-1):
            wnp1[i,j]=2*wn[i,j,cnt]-wn[i,j,cnt-1]+\
            C[i,j]**2*(wn[i+1,j,cnt]+wn[i,j+1,cnt]-\
            4*wn[i,j,cnt]+wn[i-1,j,cnt]+ wn[i,j-1,cnt]);
                
    cnt+=1;
    t+=dt;

fig = plt.figure()
ax = plt.axes(xlim=(1, nx-1), ylim=(1, ny-1))
im = ax.imshow(wn[:,:,0], animated=True, vmin=-.5,vmax=.5, cmap='viridis')

'''
Doble Slits
wn[:,21:,:]=np.multiply(10,wn[:,21:,:]);
wn[:45,20,:]=-100;
wn[46:54,20,:]=-100;
wn[55:,20,:]=-100;
'''


z=np.multiply(.4,np.repeat(c[:,:,np.newaxis],cnt,axis=2))-0.2;
wn+=z;

def init():
    im.set_array(wn[:,:,1])
    return im,

def animate(i):
    global wn
    im.set_array(wn[:,:,i+2])
    return im,

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=cnt-2, interval=1, blit=True)
#ani.save('mode1.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show()   