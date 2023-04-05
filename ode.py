import matplotlib.pyplot as plt
import numpy as np

# Accerleration Equation:
newton_second_law = lambda x: -1 * x


def leapfrog(x, v, a, t):
    """
         return the next step for x(t), x'(t) (v), x''(t) (a) using Leapfrog method

         Receive: current x, v, a, t

         Return: the next step x_step, v_step, a_step
         """
    v_half_step = v + 0.5 * a * t
    x_step = x + v_half_step * t
    a_step = newton_second_law(x_step)
    v_step = v_half_step + 0.5 * a_step * t
    return x_step, v_step, a_step


def euler(x, v, a, t):
    """
         return the next step for x(t), x'(t) (v), x''(t) (a) using Euler's method

         Receive: current x, v, a, t

         Return: the next step x_step, v_step, a_step
         """
    x_step = x + v * t
    v_step = v + a * t
    a_step = newton_second_law(x_step)
    return x_step, v_step, a_step


def euler_cromer(x, v, a, t):
    """
         return the next step for x(t), x'(t) (v), x''(t) (a) using Euler's method

         Receive: current x, v, a, t

         Return: the next step x_step, v_step, a_step
         """
    v_step = v + a * t
    x_step = x + v_step * t
    a_step = newton_second_law(x_step)
    return x_step, v_step, a_step


def runge_kutta(x, v, dt, k=1, m=1):
    x1 = x
    v1 = v
    a1 = -k * x1 / m

    x2 = x1 + v1 * dt / 2
    v2 = v1 + a1 * dt / 2
    a2 = -k * x2 / m

    x3 = x1 + v2 * dt / 2
    v3 = v1 + a2 * dt / 2
    a3 = -k * x3 / m

    x4 = x1 + v3 * dt
    v4 = v1 + a3 * dt
    a4 = -k * x4 / m

    x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
    v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return x, v


def solve(delta_t, x0, v0, type, t0=0.0, tf=4 * np.pi):
    """
         return the np array for solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)

         Receive: dt(delta_t), x initial (x0), x' initial (v0), type(either 1 or 2, 1 for LeapFrog and 2 for Euler's), starting time(t0),
                    and ending time (tf)

         Return: solution (x_value), solution'(v_value), solution''(a_value), and time(t_value)
         """
    num_step = int((tf - t0) / delta_t)
    t_value = np.linspace(t0, tf, int(num_step))
    x_value = np.zeros(num_step)
    v_value = np.zeros(num_step)
    a_value = np.zeros(num_step)
    x_value[0] = x0
    v_value[0] = v0
    a_value[0] = newton_second_law(x0)

    if type == 1:
        for i in range(1, num_step):
            x_value[i], v_value[i], a_value[i] = leapfrog(x_value[i - 1], v_value[i - 1],
                                                          newton_second_law(x_value[i - 1]), delta_t)
        return x_value, v_value, a_value, t_value
    if type == 2:
        for i in range(1, num_step):
            x_value[i], v_value[i], a_value[i] = euler(x_value[i - 1], v_value[i - 1],
                                                       newton_second_law(x_value[i - 1]), delta_t)
        return x_value, v_value, a_value, t_value
    if type == 3:
        for i in range(1, num_step):
            x_value[i], v_value[i], a_value[i] = euler_cromer(x_value[i - 1], v_value[i - 1],
                                                              newton_second_law(x_value[i - 1]), delta_t)
        return x_value, v_value, a_value, t_value
    if type ==4:
        for i in range(1, num_step):
            x_value[i], v_value[i] = runge_kutta(x_value[i - 1], v_value[i - 1], delta_t, )
        return x_value, v_value, t_value

"""
dt1 = 0.1 * np.pi
dt2 = 0.01 * np.pi
dt3 = 0.001 * np.pi

fig, ax = plt.subplots(2)
x_value, v_value, a_value, t_value = solve(dt1, 0, 1, 1)
x_value2, v_value2, a_value2, t_value2 = solve(dt2, 0, 1, 1)
x_value3, v_value3, a_value3, t_value3 = solve(dt3, 0, 1, 1)

anal_x = list(np.sin(x) for x in t_value3)
ax[0].plot(t_value, x_value, label='Leapfrog for 0.1pi')
ax[0].plot(t_value2, x_value2, label='Leapfrog for 0.01pi')
ax[0].plot(t_value3, x_value3, label='Leapfrog for 0.001pi')
ax[0].plot(t_value3, anal_x, label='Analytical Solution')
ax[0].legend(loc="lower left", prop={'size': 5})

x_value, v_value, a_value, t_value = solve(dt1, 0, 1, 2)
x_value2, v_value2, a_value2, t_value2 = solve(dt2, 0, 1, 2)
x_value3, v_value3, a_value3, t_value3 = solve(dt3, 0, 1, 2)
anal_x = list(np.sin(x) for x in t_value)
ax[1].plot(t_value, x_value, label='Euler for 0.1pi')
ax[1].plot(t_value2, x_value2, label='Euler for 0.01pi')
ax[1].plot(t_value3, x_value3, label='Euler for 0.001pi')
ax[1].plot(t_value, anal_x, label='Analytical Solution')
ax[1].legend(loc="lower left", prop={'size': 5})

# For energy Difference part only
fig, bx = plt.subplots(2)

x_value, v_value, a_value, t_value = solve(dt1, 0, 1, 1)
x_value2, v_value2, a_value2, t_value2 = solve(dt2, 0, 1, 1)
x_value3, v_value3, a_value3, t_value3 = solve(dt3, 0, 1, 1)

ef1 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value, v_value))
ef2 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value2, v_value2))
ef3 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value3, v_value3))
e0 = 0.5
de1 = list(abs(ef1 - e0) / e0 for ef1 in ef1)
de2 = list(abs(ef2 - e0) / e0 for ef2 in ef2)
de3 = list(abs(ef3 - e0) / e0 for ef3 in ef3)

bx[0].plot(t_value, de1, label='energy difference using Leapfrog for 0.1pi')
bx[0].plot(t_value2, de2, label='energy difference using Leapfrog or 0.01pi')
bx[0].plot(t_value3, de3, label='energy difference Using Leapfrog for 0.001pi')
bx[0].legend(loc="lower left", prop={'size': 5})

x1_value, v1_value, a1_value, _ = solve(dt1, 0, 1, 2)
x1_value2, v1_value2, a1_value2, _ = solve(dt2, 0, 1, 2)
x1_value3, v1_value3, a1_value3, _ = solve(dt3, 0, 1, 2)

ef1 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x1_value, v1_value))
ef2 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x1_value2, v1_value2))
ef3 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x1_value3, v1_value3))
e0 = 0.5
de1 = list(abs(ef1 - e0) / e0 for ef1 in ef1)
de2 = list(abs(ef2 - e0) / e0 for ef2 in ef2)
de3 = list(abs(ef3 - e0) / e0 for ef3 in ef3)

bx[1].plot(t_value, de1, label='energy difference using Euler for 0.1pi')
bx[1].plot(t_value2, de2, label='energy difference using Euler or 0.01pi')
bx[1].plot(t_value3, de3, label='energy difference Using Euler for 0.001pi')
bx[1].legend(loc="lower left", prop={'size': 5})
plt.show()

# for Euler-Cromer method:
fig, cx = plt.subplots(2)
x_value, v_value, a_value, t_value = solve(dt1, 0, 1, 3)
x_value2, v_value2, a_value2, t_value2 = solve(dt2, 0, 1, 3)
x_value3, v_value3, a_value3, t_value3 = solve(dt3, 0, 1, 3)

ef1 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value, v_value))
ef2 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value2, v_value2))
ef3 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value3, v_value3))
e0 = 0.5
de1 = list(abs(ef1 - e0) / e0 for ef1 in ef1)
de2 = list(abs(ef2 - e0) / e0 for ef2 in ef2)
de3 = list(abs(ef3 - e0) / e0 for ef3 in ef3)

cx[0].plot(t_value, de1, label='energy difference using Euler-cromer for 0.1pi')
cx[0].plot(t_value2, de2, label='energy difference using Euler-cromer or 0.01pi')
cx[0].plot(t_value3, de3, label='energy difference Using Euler-cromer for 0.001pi')
cx[0].legend(loc="lower left", prop={'size': 5})

# for 4th Runge-Kutta method:
x_value, v_value, t_value = solve(dt1, 0, 1, 4)
x_value2, v_value2, t_value2 = solve(dt2, 0, 1, 4)
x_value3, v_value3, t_value3 = solve(dt3, 0, 1, 4)

ef1 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value, v_value))
ef2 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value2, v_value2))
ef3 = list(0.5 * 1 * x ** 2 + 0.5 * 1 * v ** 2 for x, v in zip(x_value3, v_value3))
e0 = 0.5
de1 = list(abs(ef1 - e0) / e0 for ef1 in ef1)
de2 = list(abs(ef2 - e0) / e0 for ef2 in ef2)
de3 = list(abs(ef3 - e0) / e0 for ef3 in ef3)
fig, cx = plt.subplots(2)
cx[1].logplot(t_value, de1, label='energy difference using 4th Runge-Kutta for 0.1pi')
cx[1].plot(t_value2, de2, label='energy difference using 4th Runge-Kutta or 0.01pi')
cx[1].plot(t_value3, de3, label='energy difference Using 4th Runge-Kutta for 0.001pi')
cx[1].legend(loc="lower left", prop={'size': 5})
plt.show()"""