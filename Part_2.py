import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

# Sample command: python3 Part_2.py --solver rk4 --obj planet --show_plot --q q1

parser = argparse.ArgumentParser(description='Simulate a planet orbiting the Star using different numerical methods.')
parser.add_argument('--solver', dest='solver', type=str, default='rk4', help='Choose a solver: euler, euler_cromer, midpoint, or rk4. (default: rk4)')
parser.add_argument('--tol', dest='tol', type=float, default=1e-8, help='tolerance. (default: 1e-8)')
parser.add_argument('--obj', default="planet",help='Choose either the planet or the comet for this Problem')
parser.add_argument('--q', default="q1",help='Choose which subproblem to solve(q1, q2, q3)')
parser.add_argument('--show_plot', dest='show_plot', action='store_true', help='Show plot of the simulation. (default: False)')
parser.add_argument('--save_plot', dest='save_plot', type=str, default=None, help='Save plot of the simulation to a file. (default: None)')


args = parser.parse_args()

G = 6.67e-11 # Gravitational Constant 
M= 1.98892e30 # Mass of the Sun in kg
au = 149597870700   # 1 Au in meters

def newton_second_law(r):
    """
    Return the acceleration of given position.

    Receive: current r (1D array with two elements, position vector rx and ry)

    Return: the acceleration ax and ay in [ax, ay]
    """
    array = np.zeros(2)
    x_val = r[0]
    y_val = r[1]
    r_value = np.sqrt(x_val**2+y_val**2)
    array[0] = -G*M*x_val/(r_value**3)
    array[1]=-G*M*y_val/(r_value**3)
    return array

def euler(x, v, a, dt):
    """
    Return the next step for x(t), x'(t) (v), x''(t) (a) using Euler's method.

    x: current position
    v: current velocity
    a: current acceleration
    dt: timestep


    Return: the next step x_step, v_step, a_step
    """
    x_step = x + v * dt
    v_step = v + a * dt
    a_step = newton_second_law(x_step)
    return x_step, v_step, a_step

def euler_cromer(x, v, a, dt):
    """
    Return the next step for x(t), x'(t) (v), x''(t) (a) using Euler's method.

    x: current position
    v: current velocity
    a: current acceleration
    dt: timestep

    Return: the next step x_step, v_step, a_step
    """
    v_step = v + a * dt
    x_step = x + v_step * dt
    a_step = newton_second_law(x_step)
    return x_step, v_step, a_step

def midpoint(x, v, a, dt):
    """
    Return the next step for x(t), x'(t) (v), x''(t) (a) using Midpoint method.

    x: current position
    v: current velocity
    a: current acceleration
    dt: timestep
    :return: the next step x_step, v_step, a_step
    """
    x_step = x + v * dt + 0.5 * a * dt**2
    a_step = newton_second_law(x_step)
    v_step = v + 0.5 * (a + a_step) * dt
    return x_step, v_step, a_step    

def runge_kutta(x, v, dt):
    """
    Return the next step for x(t), x'(t) (v), using 4th Runge-Kutta's method.

    x: current position
    v: current velocity
    a: current acceleration
    dt: timestep

    Return: the next step x_step, v_step
    """
    x1 = x
    v1 = v
    a1 = newton_second_law(x1)

    x2 = x1 + v1 * dt / 2
    v2 = v1 + a1 * dt / 2
    a2 = newton_second_law(x2)

    x3 = x1 + v2 * dt / 2
    v3 = v1 + a2 * dt / 2
    a3 = newton_second_law(x3)

    x4 = x1 + v3 * dt
    v4 = v1 + a3 * dt
    a4 = newton_second_law(x4)

    x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
    v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return x, v

# Part 1 


# Planet:
G = 6.67e-11
M= 1.98892e30
au = 149597870700
a_p = 1.0*au # Semimajor axis in m
e_p = 0.02 # Eccentricity
x_p0 = 0.0 # Initial x position in m
y_p0 = a_p * (1 - e_p) # Initial y position in m
r_0 = [x_p0, y_p0]
vx_p0 = np.sqrt((G*M/a_p) * (1 + e_p)/(1 - e_p)) # Initial x velocity in m/sec
vy_p0 = 0.0*au # Initial y velocity in m/s
v_0 = [vx_p0, vy_p0]
# Time parameters
t0 = 0.0 # Initial time in seconds
tend = 26.0*3600*24*365 # Final time in seconds
dt = 0.01*3600*24*365 # Timestep in seconds

def solve(x0=r_0, v0=v_0, t0=t0, tf=tend, delta_t=dt, acceleration=newton_second_law, type=1,):
    """
         return the np array for solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)

         Receive: dt(delta_t), x initial (x0), x' initial (v0), type(1,2,3,4), 1 for euler, 2 for euler cromer, 3 for midpoint, and 4 for 4th rongue-kutta), starting time(t0),
             ending time (tf), and tolerance(tol)

         Return: solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)
         """
    t = np.arange(t0,tf,delta_t)
    num_step = len(t)

    
    x_value = np.zeros((num_step, 2))
    v_value = np.zeros((num_step, 2))
    a_value = np.zeros((num_step, 2))
    x_value[0] = x0
    v_value[0] = v0
    a_value[0] = newton_second_law(x0)
    
    max_iteration = 100000
    i = 1
    t_value = 0
    t_list = [0]


    if type == 1:
        print("you are using euler")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            x_value[i], v_value[i], a_value[i] = euler(x_value[i - 1], v_value[i - 1],
                                                          newton_second_law(x_value[i - 1]), delta_t)
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value, t_list

    if type == 2:
        print("you are using euler cromer")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            
            x_value[i], v_value[i], a_value[i] = euler_cromer(x_value[i - 1], v_value[i - 1],
                                                          newton_second_law(x_value[i - 1]), delta_t)
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value, t_list

    if type == 3:
        print("you are using midpoint")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
           
            x_value[i], v_value[i], a_value[i] = midpoint(x_value[i - 1], v_value[i - 1],
                                                          newton_second_law(x_value[i - 1]), delta_t)
            
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value,  t_list
        
    if type==4:
        print("you are using Rk4")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            
            x_value[i], v_value[i] = runge_kutta(x_value[i - 1], v_value[i - 1], delta_t, )
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, t_list


# Make a plot that shows the position of the planet and comet from t = 0 to t = 26 years for a timestep of ∆t = 0.01 years.


# Comet initial conditions
G = 6.67e-11
M= 1.98892e30
au = 149597870700
a_c = 3.0*au # Semimajor axis in m
e_c = 0.95 # Eccentricity
x_c0 = 0.0*au # Initial x position in m
y_c0 = a_c * (1 - e_c) # Initial y position in m
vx_c0 = np.sqrt((G*M/a_c) * (1 + e_c)/(1 - e_c)) # Initial x velocity in m/sec
vy_c0 = 0.0*au # Initial y velocity in m/sec
r_c0 = [x_c0, y_c0]
v_c0 = [vx_c0, vy_c0]
t0 = 0.0 # Initial time in seconds
tend = 26.0*3600*24*365 # Final time in seconds
dt = 0.01*3600*24*365 # Timestep in seconds



def solve2(x0=r_c0, v0=v_c0, t0=t0, tf=tend, delta_t=dt, acceleration=newton_second_law, type=1,):
    """
         return the np array for solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)

         Receive: dt(delta_t), x initial (x0), x' initial (v0), type(1,2,3,4), 1 for euler, 2 for euler cromer, 3 for midpoint, and 4 for 4th rongue-kutta), starting time(t0),
             ending time (tf), and tolerance(tol)

         Return: solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)
         """
    t = np.arange(t0,tf,delta_t)
    num_step = len(t)

    
    x_value = np.zeros((num_step, 2))
    v_value = np.zeros((num_step, 2))
    a_value = np.zeros((num_step, 2))
    x_value[0] = x0
    v_value[0] = v0
    a_value[0] = newton_second_law(x0)
    
    max_iteration = 100000
    i = 1
    t_value = 0
    t_list = [0]


    if type == 1:
        print("you are using euler")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            x_value[i], v_value[i], a_value[i] = euler(x_value[i - 1], v_value[i - 1],
                                                          acceleration(x_value[i - 1]), delta_t)
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value, t_list

    if type == 2:
        print("you are using euler cromer")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            
            x_value[i], v_value[i], a_value[i] = euler_cromer(x_value[i - 1], v_value[i - 1],
                                                          acceleration(x_value[i - 1]), delta_t)
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value, t_list

    if type == 3:
        print("you are using midpoint")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
           
            x_value[i], v_value[i], a_value[i] = midpoint(x_value[i - 1], v_value[i - 1],
                                                          acceleration(x_value[i - 1]), delta_t)
            
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, a_value,  t_list
        
    if type==4:
        print("you are using Rk4")
        while tf-t_value>delta_t and i< max_iteration and i<num_step:
            
            x_value[i], v_value[i] = runge_kutta(x_value[i - 1], v_value[i - 1], delta_t, )
            t_value +=delta_t
            i = i+1
            t_list.append(t_value)
        return x_value, v_value, t_list

# +
#Q1 of Part1
#Make a plot that shows the position of the planet
#and comet from t = 0 to t = 26 years for a timestep of ∆t = 0.01 years.

if args.obj =="planet":
    # Run simulation using chosen solver
    if args.solver == 'euler':
        x, v,a,t = solve(r_0, v_0, t0, tend, dt, type=1,)
        
    elif args.solver == 'euler_cromer':
        x, v,a,t =solve(r_0, v_0, t0, tend, dt, type= 2,)
        
    elif args.solver == 'midpoint':
        x, v,a,t =solve(r_0, v_0, t0, tend, dt, type=3,)
        
    else:
        x, v,t =solve(r_0, v_0, t0, tend, dt, type=4,)
        # Show plot of simulation if requested
    if args.show_plot:
        x_pos = [x_val[0] for x_val in x]
        y_pos = [y_val[1] for y_val in x]
        
        
        
        plt.plot(x_pos, y_pos, label="planet")
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12, 1e12)
        plt.legend()
        plt.show()

    # Save plot of simulation to a file if requested
    if args.save_plot is not None:
        x_pos = [x_val[0] for x_val in x]
        y_pos = [y_val[1] for y_val in x]
            
        plt.plot(x_pos, y_pos,label="planet")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12, 1e12)
        plt.legend()
        plt.show()
        plt.savefig(args.save_plot)
            
    elif args.obj =="comet":
        if args.solver == 'euler':
        
            x_c, v_c, a_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type=1)
        elif args.solver == 'euler_cromer':
        
            x_c, v_c, a_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type=2)
        elif args.solver == 'midpoint':
        
            x_c, v_c, a_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type= 3)
        else:
            
            x_c, v_c, a_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type=4)
            # Show plot of simulation if requested
        if args.show_plot:
            c_x_pos = [x_val[0] for x_val in x_c]
            c_y_pos = [y_val[1] for y_val in x_c]
        
            plt.plot(c_x_pos, c_y_pos, label="comet", linestyle='dashed')
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.ylim(-1e12, 1e12)
            plt.xlim(-1e12, 1e12)
            plt.legend()
            plt.show()

        # Save plot of simulation to a file if requested
        if args.save_plot is not None:
            c_x_pos = [x_val[0] for x_val in x_c]
            c_y_pos = [y_val[1] for y_val in x_c]
            plt.plot(c_x_pos, c_y_pos, label="comet", linestyle='dashed')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim(-1e12, 1e12)
            plt.xlim(-1e12, 1e12)
            plt.legend()
            plt.show()
            plt.savefig(args.save_plot)
    else:
        AssertionError("Invalid object argument. Please choose either comet or planet")
    # -
if args.q=='q1':
    methods = [1,2,3,4]  #1 for euler, 2 for euler cromer, 3 for midpoint, and 4 for 4th rongue-kutta
    for method in methods:
        if method !=4:
            x, v,a,t = solve(r_0, v_0, t0, tend, dt, type=method,)
            x_c, v_c, a_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type=method)
            x_pos = [x_val[0] for x_val in x]
            y_pos = [y_val[1] for y_val in x]
            c_x_pos = [x_val[0] for x_val in x_c]
            c_y_pos = [y_val[1] for y_val in x_c]
            plt.plot(x_pos, y_pos)
            plt.plot(c_x_pos, c_y_pos)
            plt.plot(x_pos, y_pos, label="planet")
            plt.plot(c_x_pos, c_y_pos, label="comet", linestyle='dashed')
            plt.title(f"method used is {method}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.ylim(-1e12, 1e12)
            plt.xlim(-1e12, 1e12)
            plt.legend()
            plt.show()
        else:
            x, v,t = solve(r_0, v_0, t0, tend, dt, type=method,)
            x_c, v_c, t_c = solve2(r_c0, v_c0, t0, tend, dt, type= method)
            x_pos = [x_val[0] for x_val in x]
            y_pos = [y_val[1] for y_val in x]
            c_x_pos = [x_val[0] for x_val in x_c]
            c_y_pos = [y_val[1] for y_val in x_c]
            plt.plot(x_pos, y_pos)
            plt.plot(c_x_pos, c_y_pos)
            plt.plot(x_pos, y_pos, label="planet")
            plt.plot(c_x_pos, c_y_pos, label="comet", linestyle='dashed')
            plt.title(f"method used is {method}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.ylim(-1e12, 1e12)
            plt.xlim(-1e12, 1e12)
            plt.legend()
            plt.show()

if args.q=="q2":
    # +

    #Codes Below are the codes used for q2 of Part_2. 

    # For Type 1 (Euler's method)
    energy_difference_type1 = []
    angular_momentum_type1 = []
    c_energy_difference_type1 = []
    c_angular_momentum_type1 = []
    for dt in [1e-4, 1e-3, 1e-2, 1e-1]:
        x_value, v_value, a_value, t_value = solve(delta_t=dt*3600*24*365,type=1)
        x_c_value, v_c_value, a_v_value, t_c_value = solve2(delta_t=dt*3600*24*365,type=1)
        x_pos = [x_val[0] for x_val in x_value]
        y_pos = [y_val[1] for y_val in x_value]
        ef =  1/2*G*M/np.sqrt(x_value[-1][0]**2+x_value[-1][1]**2) 
        c_ef = 1/2*G*M/np.sqrt(x_c_value[-1][0]**2+x_c_value[-1][1]**2) 
        c_l = np.cross(x_c_value[-1],v_c_value[-1])
        l = np.cross(x_value[-1],v_value[-1])
        energy_difference_type1.append(ef)
        angular_momentum_type1.append(l)
        c_energy_difference_type1.append(c_ef)
        c_angular_momentum_type1.append(c_l)
        x_c_pos = [x_val[0] for x_val in x_c_value]
        y_c_pos = [y_val[1] for y_val in x_c_value]
        plt.plot(x_pos, y_pos,label = f"planet {dt}")
        plt.plot(x_c_pos, y_c_pos, label =f"comet {dt}",linestyle='dashed')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12,1e12)
    plt.legend()
    plt.title('Trajectory Using Euler')
    plt.show()

    # For type 2 (Euler Cromer's Method)
    energy_difference_type2 = []
    angular_momentum_type2 = []
    c_energy_difference_type2 = []
    c_angular_momentum_type2 = []

    for dt in [1e-4, 1e-3, 1e-2, 1e-1]:
        x_value, v_value, a_value, t_value = solve(delta_t=dt*3600*24*365,type=2)
        x_c_value, v_c_value, a_v_value, t_c_value = solve2(delta_t=dt*3600*24*365,type=2)
        x_pos = [x_val[0] for x_val in x_value]
        y_pos = [y_val[1] for y_val in x_value]
        ef =  1/2*G*M/np.sqrt(x_value[-1][0]**2+x_value[-1][1]**2) 
        l = np.cross(x_value[-1],v_value[-1])
        c_ef = 1/2*G*M/np.sqrt(x_c_value[-1][0]**2+x_c_value[-1][1]**2) 
        c_l = np.cross(x_c_value[-1],v_c_value[-1])
        energy_difference_type2.append(ef)
        angular_momentum_type2.append(l)
        c_energy_difference_type2.append(c_ef)
        c_angular_momentum_type2.append(c_l)
        x_c_pos = [x_val[0] for x_val in x_c_value]
        y_c_pos = [y_val[1] for y_val in x_c_value]
        plt.plot(x_pos, y_pos,label = f"planet {dt}")
        plt.plot(x_c_pos, y_c_pos, label =f"comet {dt}",linestyle='dashed')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12,1e12)
    plt.legend()
    plt.title('Trajectory Using Euler cromer')
    plt.show()

    # For type 3 (Midpoint method)
    energy_difference_type3 = []
    angular_momentum_type3 = []
    c_energy_difference_type3 = []
    c_angular_momentum_type3 = []
    for dt in [1e-4, 1e-3, 1e-2, 1e-1]:
        x_value, v_value, a_value, t_value = solve(delta_t=dt*3600*24*365,type=3)
        x_c_value, v_c_value, a_v_value, t_c_value = solve2(delta_t=dt*3600*24*365,type=3)
        x_pos = [x_val[0] for x_val in x_value]
        y_pos = [y_val[1] for y_val in x_value]
        ef =  1/2*G*M/np.sqrt(x_value[-1][0]**2+x_value[-1][1]**2) 
        l = np.cross(x_value[-1],v_value[-1])
        c_ef = 1/2*G*M/np.sqrt(x_c_value[-1][0]**2+x_c_value[-1][1]**2) 
        c_l = np.cross(x_c_value[-1],v_c_value[-1])
        c_energy_difference_type3.append(c_ef)
        c_angular_momentum_type3.append(c_l)
        energy_difference_type3.append(ef)
        angular_momentum_type3.append(l)
        x_c_pos = [x_val[0] for x_val in x_c_value]
        y_c_pos = [y_val[1] for y_val in x_c_value]
        plt.plot(x_pos, y_pos,label = f"planet {dt}")
        plt.plot(x_c_pos, y_c_pos, label =f"comet {dt}",linestyle='dashed')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12,1e12)
    plt.legend()
    plt.title('Trajectory Using midpoint')
    plt.show()


    # For Type 4 (4th Ronge-Kutta method)
    energy_difference_type4 = []
    angular_momentum_type4 = []
    c_energy_difference_type4 = []
    c_angular_momentum_type4 = []
    for dt in [1e-4, 1e-3, 1e-2, 1e-1]:
        x_value, v_value, t_value = solve(delta_t=dt*3600*24*365,type=4)
        x_c_value, v_c_value,t_c_value = solve2(delta_t=dt*3600*24*365,type=4)
        x_pos = [x_val[0] for x_val in x_value]
        y_pos = [y_val[1] for y_val in x_value]
        ef =  1/2*G*M/np.sqrt(x_value[-1][0]**2+x_value[-1][1]**2) 
        l = np.cross(x_value[-1],v_value[-1])
        c_ef = 1/2*G*M/np.sqrt(x_c_value[-1][0]**2+x_c_value[-1][1]**2) 
        c_l = np.cross(x_c_value[-1],v_c_value[-1])
        energy_difference_type4.append(ef)
        angular_momentum_type4.append(l)
        c_energy_difference_type4.append(c_ef)
        c_angular_momentum_type4.append(c_l)
        x_c_pos = [x_val[0] for x_val in x_c_value]
        y_c_pos = [y_val[1] for y_val in x_c_value]
        plt.plot(x_pos, y_pos,label = f"planet {dt}")
        plt.plot(x_c_pos, y_c_pos, label =f"comet {dt}",linestyle='dashed')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.ylim(-1e12, 1e12)
        plt.xlim(-1e12,1e12)
    plt.legend()
    plt.title('Trajectory Using 4th Rk')
    plt.show()



    # +

    # Q2 For Planet


    #energy quantity:
    e0 = G*M/(2*np.sqrt(r_0[0]**2+r_0[1]**2))
    t_list = [1e-4, 1e-3, 1e-2, 1e-1]
    error_1 = [np.abs(energy_difference_type1[i]-e0)/np.abs(e0) for i in range(4)]
    error_2 = [np.abs(energy_difference_type2[i]-e0)/np.abs(e0) for i in range(4)]
    error_3 = [np.abs(energy_difference_type3[i]-e0)/np.abs(e0) for i in range(4)]
    error_4 = [np.abs(energy_difference_type4[i]-e0)/np.abs(e0) for i in range(4)]

    fig, bx = plt.subplots(2)
    bx[0].loglog(t_list,error_1,label = 'euler')
    bx[0].loglog(t_list,error_2,label = 'euler_cromer')
    bx[0].loglog(t_list,error_3,label = 'midpoint')
    bx[0].loglog(t_list,error_4, label = 'rongue_kutta')
    bx[0].title.set_text("Planet Energy error")
    #Angular momentum 
    l0 = np.cross(r_0, v_0)
    l_error_1 = [np.abs(angular_momentum_type1[i]-l0)/np.abs(l0) for i in range(4)]
    l_error_2 = [np.abs(angular_momentum_type2[i]-l0)/np.abs(l0) for i in range(4)]
    l_error_3 = [np.abs(angular_momentum_type3[i]-l0)/np.abs(l0) for i in range(4)]
    l_error_4 = [np.abs(angular_momentum_type4[i]-l0)/np.abs(l0) for i in range(4)]
    bx[1].loglog(t_list,l_error_1,label = 'euler')
    bx[1].loglog(t_list,l_error_2,label = 'euler_cromer')
    bx[1].loglog(t_list,l_error_3,label = 'midpoint')
    bx[1].loglog(t_list, l_error_4, label = 'rongue_kutta')
    bx[1].title.set_text("Planet Momentum error")
    plt.legend()
    plt.show()




    # Q2 for comet


    #energy quantity:
    c_e0 = G*M/(2*np.sqrt(r_c0[0]**2+r_c0[1]**2))
    t_list = [1e-4, 1e-3, 1e-2, 1e-1]
    c_error_1 = [np.abs(c_energy_difference_type1[i]-c_e0)/np.abs(c_e0) for i in range(4)]
    c_error_2 = [np.abs(c_energy_difference_type2[i]-c_e0)/np.abs(c_e0) for i in range(4)]
    c_error_3 = [np.abs(c_energy_difference_type3[i]-c_e0)/np.abs(c_e0) for i in range(4)]
    c_error_4 = [np.abs(c_energy_difference_type4[i]-c_e0)/np.abs(c_e0) for i in range(4)]

    fig, bx = plt.subplots(2)
    bx[0].loglog(t_list,c_error_1,label = 'euler')
    bx[0].loglog(t_list,c_error_2,label = 'euler_cromer')
    bx[0].loglog(t_list,c_error_3,label = 'midpoint')
    bx[0].loglog(t_list,c_error_4, label = 'rongue_kutta')
    bx[0].title.set_text("Comet energy error")


    #Angular Momentum
    c_l0 = np.cross(r_c0, v_c0)
    c_l_error_1 = [np.abs(c_angular_momentum_type1[i]-c_l0)/np.abs(c_l0) for i in range(4)]
    c_l_error_2 = [np.abs(c_angular_momentum_type2[i]-c_l0)/np.abs(c_l0) for i in range(4)]
    c_l_error_3 = [np.abs(c_angular_momentum_type3[i]-c_l0)/np.abs(c_l0) for i in range(4)]
    c_l_error_4 = [np.abs(c_angular_momentum_type4[i]-c_l0)/np.abs(c_l0) for i in range(4)]
    bx[1].loglog(t_list,c_l_error_1,label = 'euler')
    bx[1].loglog(t_list,c_l_error_2,label = 'euler_cromer')
    bx[1].loglog(t_list,c_l_error_3,label = 'midpoint')
    bx[1].loglog(t_list, c_l_error_4, label = 'rongue_kutta')
    bx[1].title.set_text("Comet Momentum error")
    plt.legend()
    plt.show()

# -

# Planet:
G = 6.67e-11
M= 1.98892e30
au = 149597870700
a_p = 1.0*au # Semimajor axis in m
e_p = 0.02 # Eccentricity
x_p0 = 0.0 # Initial x position in m
y_p0 = a_p * (1 - e_p) # Initial y position in m
r_0 = [x_p0, y_p0]
vx_p0 = np.sqrt(G*M/a_p) * (1 + e_p)/(1 - e_p) # Initial x velocity in m/second
vy_p0 = 0.0*au # Initial y velocity in m/s
v_0 = [vx_p0, vy_p0]
# Time parameters
t0 = 0.0 # Initial time in seconds
tend = 26.0*3600*24*365 # Final time in seconds
dt = 0.01*3600*24*365 # Timestep in seconds
e0 = G*M/(2*np.sqrt(r_0[0]**2+r_0[1]**2))
l0 = np.cross(r_0, v_0)

def adaptive_method(a1, a2, dt, lim_frac_err, order):
    """
    Determines the next timestep to be used in the associated
    iterate method

    args:

        a1 (array) - resulting r element after taking dt/2 timesteps
             for a given position

        a2 (array) - resulting r element after taking dt timesteps
             for a given position

        dt (float) - current step size

        lim_frac_err (float) - limiting fractional error that determines
                       how accurate we want our timestep to be 

        order (int) - order of method we are using
    
    return:

        ndt (float) - new adjusted stepsize
    """
    err = 1/(2**(order+1) - 2) * np.abs(a1 - a2)

    euc_err = np.sqrt(err[0]**2 + err[1]**2)

    rho = dt*lim_frac_err / euc_err

    ndt = rho**(1/order) * dt

    return ndt


def iterate_adp(r0, v0, tf, dt0, method, order, tol = 1e-6):
    """
    Solves the given ODE using an adaptive timestep

    args:

        tf (float/int) - final time to iterate towards. Should be in seconds

        dt0 (float) - initial timestep by which we move our objects through time

        r0 (array) - initial position of object. Can be in any number of 
                     dimensions

        v0 (array) - initial velocity of object. Can be in any number of 
                     dimensions

        method (Function) - method by which we use to solve the equation

        f (Function) - function that we use, along with our method to step
                       our analysis forward in time
        
        order (int) - order associated with the method we are using

        tol (float) - fractional error for our adaptive timestep

    returns:

        r (array) - position array

        v (array) - velocity array

        t (array) - time array
    """  

    t = np.arange(0,tf,dt0)
    n = len(t)
    

    r = np.zeros([n,2])
    v = np.zeros([n,2])
    a = np.zeros([n,2])
    
    r[0] = r0
    v[0] = v0 
    a[0] = newton_second_law(r0)

    dt = dt0

    max_iter = 100000
    i = 0
    ti = 0
    while ti < tf and i <= max_iter:

        if i == len(r)-1:
            
            r = np.resize(r,[len(r)*2, 2])
            v = np.resize(v,[len(v)*2, 2])
            a = np.resize(a,[len(a)*2, 2])
            t = np.resize(t,[len(t)*2, 2])

        if i != 0 and i%5 == 0: #Check every 5 steps

            r_temp = np.zeros([3,2])
            v_temp = np.zeros([3,2])
            a_temp = np.zeros([3,2])

            r_temp[0] = r[i-1]
            v_temp[0] = v[i-1]
            a_temp[0] = a[i-1]

            for j in range(2):
                
                if method==1:

                    r_temp[j+1], v_temp[j+1], a_temp[j+1] = euler(r_temp[j], v_temp[j], newton_second_law(r_temp[j]),
                                                    dt/2,)
                if method==2:

                    r_temp[j+1], v_temp[j+1],a_temp[j+1] = euler_cromer(r_temp[j], v_temp[j], newton_second_law(r_temp[j]),
                                                    dt/2,)
                if method==3:

                    r_temp[j+1], v_temp[j+1], a_temp[j+1] = midpoint(r_temp[j], v_temp[j], newton_second_law(r_temp[j]),
                                                    dt/2,)
                if method==4:

                    r_temp[j+1], v_temp[j+1] = runge_kutta(r_temp[j], v_temp[j],
                                                    dt/2,)

            dt = adaptive_method(r_temp[-1], r[i], dt, tol, order)
            
        if method==1:
            
            r[i+1],v[i+1], a[i+1] = euler(r[i],v[i],newton_second_law(r[i]),dt)
        
        if method==2:
            
            r[i+1],v[i+1], a[i+1] = euler_cromer(r[i],v[i],newton_second_law(r[i]),dt)
        if method==3:
            
            r[i+1],v[i+1], a[i+1] = midpoint(r[i],v[i],newton_second_law(r[i]),dt)
        if method==4:
            
            r[i+1],v[i+1] = runge_kutta(r[i],v[i],dt)
        

        i+=1
        ti+=dt

    return r[0:i],v[0:i],t[0:i]


if args.q=='q3':
    # +
    #Planet
    start = time.time()

    r,v,t = iterate_adp(r_0, v_0, tend, dt, 1, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="Euler")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Planet")
    print("time spent to run euler adpt is", running)

    start = time.time()

    r,v,t = iterate_adp(r_0, v_0, tend, dt, 2, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="Euler-cromer")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Planet")
    print("time spent to run euler-cromer adpt is", running)

    start = time.time()

    r,v,t = iterate_adp(r_0, v_0, tend, dt, 3, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="midpoint")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Planet")
    print("time spent to run midpoint adpt is", running)

    start = time.time()
    r,v,t = iterate_adp(r_0, v_0, tend, dt, 4, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="4th Rk",c='b')
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.show()
    print("time spent to run rk4 adpt is", running)


    #Comet
    # Comet initial conditions
    G = 6.67e-11
    M= 1.98892e30
    au = 149597870700
    a_c = 3.0*au # Semimajor axis in m
    e_c = 0.95 # Eccentricity
    x_c0 = 0 # Initial x position in m
    y_c0 = a_c * (1 - e_c) # Initial y position in m
    vx_c0 = np.sqrt((G*M/a_c) * (1 + e_c)/(1 - e_c)) # Initial x velocity in m/sec
    vy_c0 = 0.0*au # Initial y velocity in m/sec
    r_c0 = [x_c0, y_c0]
    v_c0 = [vx_c0, vy_c0]
    t0 = 0.0 # Initial time in seconds
    tend = 26.0*3600*24*365 # Final time in seconds
    dt = 0.01*3600*24*365 # Timestep in seconds

    # Planet:
    G = 6.67e-11
    M= 1.98892e30
    au = 149597870700
    a_p = 1.0*au # Semimajor axis in m
    e_p = 0.02 # Eccentricity
    x_p0 = 0 # Initial x position in m
    y_p0 = a_p * (1 - e_p) # Initial y position in m
    r_0 = [x_p0, y_p0]
    vx_p0 = np.sqrt((G*M/a_p) * (1 + e_p)/(1 - e_p)) # Initial x velocity in m/s
    vy_p0 = 0.0*au # Initial y velocity in m/s
    v_0 = [vx_p0, vy_p0]
    # Time parameters
    t0 = 0.0 # Initial time in seconds
    tend = 26.0*3600*24*365 # Final time in seconds
    dt = 0.01*3600*24*365 # Timestep in seconds

    start = time.time()
    r,v,t = iterate_adp(r_c0, v_c0, tend, dt, 1, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="Euler")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Comet")
    print("time spent to run comet euler adpt is", running)

    start = time.time()
    r,v,t = iterate_adp(r_c0, v_c0, tend, dt, 1, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="Euler")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Comet")
    print("time spent to run comet euler-cromer  adpt is", running)

    start = time.time()
    r,v,t = iterate_adp(r_c0, v_c0, tend, dt, 1, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="Euler")
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.title("Comet")
    print("time spent to run comet midpoint adpt is", running)

    start = time.time()
    r,v,t = iterate_adp(r_c0, v_c0, tend, dt, 4, 4,1e-6)
    end = time.time()
    running = end-start
    plt.plot(r[:,0],r[:,1],label="4th Rk",c='b')
    plt.ylim(-1e12, 1e12)
    plt.xlim(-1e12,1e12)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.show()
    print("time spent to run comet rk4 adpt is", running)

    

    # -




