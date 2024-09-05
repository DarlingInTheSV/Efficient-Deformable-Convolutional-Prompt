import numpy as np
import matplotlib.pyplot as plt

# Define the function
def func(sample_num):
    return 1 / ((np.sqrt(sample_num) / 200) + 1)

# Parameters
# sample_num = 100  # Adjust sample_num as needed
# warm_n = 10       # Adjust warm_n as needed

# Generate x values
x = np.linspace(0, 1000, 1000)  # Range of x values (0 to 100)

# Compute y values
y = func(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x,y)
plt.title('Plot of 1 / ((sqrt(sample_num) / warm_n) + 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig("/home/lsy/Desktop/yy2.png")