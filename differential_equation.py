import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation dy/dt = -2y + sin(t)
def diff_eq(t, y):
    return -2 * y + np.sin(t)

# Initial condition
y0 = [0]

# Time span for the solution
t_span = (0, 10)

# Time points at which the solution is computed
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the differential equation using solve_ivp
solution = solve_ivp(diff_eq, t_span, y0, t_eval=t_eval)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='y(t)')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.title('Solution of the Differential Equation dy/dt = -2y + sin(t)')
plt.legend()
plt.grid(True)
plt.show()
