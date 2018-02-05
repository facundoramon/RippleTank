

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Dominio
"""

Lx = 10; #Largo de la cuerda
dx = 0.1; #Pasos espaciales
nx = np.int(Lx/dx); #cantidad de pasos espaciales
x = np.arange(0,Lx,dx); #vector de posicion en x

T=50; #Tiempo de simulacion

"""
Condiciones Iniciales
"""

C=1; #C=c*dt/dx
c = 1; #velocidad de propagacion
dt = C*dx/c; #Paso temporal

t = 0; #Tiempo inicial
cnt = 1; #Contador auxiliar

wn = np.zeros((nx,np.int(T/dt)+1)); #Matriz de resultados 
wnp1 = np.copy(wn[:,cnt]); #Resultados en t+1

"""
Estimulo
"""
N=50; #Tamaño de funcion sinc
h = np.linspace(-5,5,N)
h = np.multiply(0.2,np.sinc(h));

"""
Resolucion FDTD
"""

while(t<T):
 
    wn[:,cnt]=np.copy(wnp1); #Guarda posición anterior
    wn[-1,cnt]=0; #Condicion de borde reflejante
    
    """
    Generacion de estímulo
    """
    
    if (cnt<=N):
        wn[0,cnt]=h[cnt-1];
    else:
        wnp1[0]=wn[1,cnt]+((C-1)/(C+1))*(wnp1[1]-wn[0,cnt]);

    """
    Resolucion de ecuacion de onda
    """
    for i in range(1,nx-1):
        wnp1[i]=2*wn[i,cnt]-wn[i,cnt-1]+pow(C,2)*(wn[i+1,cnt]-2*wn[i,cnt]+wn[i-1,cnt])
    
    cnt+=1;
    t+=dt;
    
"""
Generacion de animación
"""
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

#anim.save('sinc.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
