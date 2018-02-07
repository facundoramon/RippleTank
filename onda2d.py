import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Dominio
"""

Lx = 10; #Dimension X
Ly = 10; #Dimension Y
dx = 0.1; #Paso espacial en X
dy = dx; #Paso espacial en Y
nx = np.int(Lx/dx); #Cantidad de pasos en X
ny = np.int(Ly/dy); #Cantidad de pasos en Y
x = np.arange(0,Lx,dx); #Vector X
y = np.arange(0,Ly,dy); #Vector Y

x, y=np.meshgrid(x,y); #Grilla XY

T=20; #Tiempo total de simulacion

"""
Condiciones Iniciales
"""
m = np.multiply(1,np.ones(ny)); #Velocidad de medio en Y
#m[30:] = np.linspace(.5,1.2,ny-30); #Gradiente de Velocidad de medio en Y
#m[40:60] = .5; #Cambio brusco de medio
c = np.transpose(np.tile(m,(nx,1))); #Matriz de velocidad en Grilla XY

dt = .05; #Paso temporal

C = np.multiply(dt/dy,c); #C=dt*c/dy

t = 0.; #Tiempo inicial
cnt = 1; #Contador Auxiliar

wn = np.zeros((ny,nx,np.int(T/dt)+1)); #Matriz 3D para resultados
wnp1 = np.copy(wn[:,:,0]); #Matriz 2D para resultados en t+1


while(t<T):

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
    
    wn[:,:,cnt]=np.copy(wnp1); #Guardo resultado anterior   
    
    """
    Fuente
    """
    
    wn[20,20,cnt]=1.5*np.sin(3*np.pi*t); #Fuente senoidal en (20,20)
    
    """
    Paredes
    """
    wn[0:40,49,cnt]=0; #Barrera Acústica

    """
    Resolucion FDTD
    """
    for i in range(1,ny-1):
        for j in range(1,nx-1):
            wnp1[i,j]=2*wn[i,j,cnt]-wn[i,j,cnt-1]+\
            C[i,j]**2*(wn[i+1,j,cnt]+wn[i,j+1,cnt]-\
            4*wn[i,j,cnt]+wn[i-1,j,cnt]+ wn[i,j-1,cnt]);
                
    cnt+=1;
    t+=dt;

"""
Calculo de RMS
"""
wRMS = np.sqrt(np.mean(wn[:,:,cnt-150:cnt]**2,axis=-1));

"""
Parámetros visuales
"""
fig = plt.figure()
ax = plt.axes(xlim=(1, nx-1), ylim=(1, ny-1))
im = ax.imshow(wn[:,:,0], animated=True, vmin=-.5,vmax=.5, cmap='viridis')

"""
Pintura de obstaculos
"""
wn[0:40,49,:]=-100;

"""
Cambio de color del medio segun c
"""
z=np.multiply(.4,np.repeat(c[:,:,np.newaxis],cnt,axis=2))-.4;
wn+=z;

"""
Animacion
"""

def init():
    im.set_array(wn[:,:,1])
    return im,

def animate(i):
    global wn
    im.set_array(wn[:,:,i+2])
    return im,

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=cnt-2, interval=1, blit=True)

#ani.save('barrera.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
  
