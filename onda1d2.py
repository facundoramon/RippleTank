

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Dominio
"""

Lx = 10;
dx = 0.1;
nx = np.int(Lx/dx);
x = np.arange(0,Lx,dx);

T=50;

"""
Condiciones Iniciales
"""

C=1;
c = 1;
dt = C*dx/c;

t = 0;
cnt = 1;

wn = np.zeros((nx,np.int(T/dt)+1));
wnp1 = np.copy(wn[:,cnt]);

"""
"""
N=50;
h = np.linspace(-5,5,N)
h = np.multiply(0.2,np.sinc(h));
#h = np.hanning(N);
"""
"""

#f, = plt.plot(x,wn[:,0],marker='o',lw=0,markersize=5);
#fig = plt.gca();
#fig.set_xlim([0,Lx])
#fig.set_ylim([-1,1])
#plt.show();

while(t<T):
 
    wn[:,cnt]=np.copy(wnp1);
    wn[-1,cnt]=0;
    
    if(t<=T*.1*4):
        wn[0,cnt]=.2*np.sin(.5*np.pi*t);
    
#    if (cnt<=N):
#        wn[0,cnt]=h[cnt-1];
#    else:
#        wnp1[0]=wn[1,cnt]+((C-1)/(C+1))*(wnp1[1]-wn[0,cnt]);
##        wn[0,cnt]=0;
    
#    if (t<=3):
#        wnp1[0]=pow(dt,2)*20*np.sin(np.pi*t);
#    elif (t<3.2):
#        wnp1[0]=0;
#    else:        
#        wnp1[0]=wn[1,cnt]+((C-1)/(C+1))*(wnp1[1]-wn[0,cnt]);
        
    for i in range(1,nx-1):
        wnp1[i]=2*wn[i,cnt]-wn[i,cnt-1]+pow(C,2)*(wn[i+1,cnt]-2*wn[i,cnt]+wn[i-1,cnt])
    
#    f.set_data(x,wn[:,cnt])
#    plt.pause(.001)
    
    cnt+=1;
    t+=dt;
    
fig = plt.figure()
ax = plt.axes(xlim=(0, Lx), ylim=(-.5, .5))
line, = ax.plot([], [], lw=0, marker='o')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    global x, wn
    x_ = x;
    y_ = wn[:,i]
    line.set_data(x_, y_)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=cnt, interval=200/cnt, blit=True)

anim.save('rope_armonico4.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()