from matplotlib import pyplot as plt
from matplotlib.pyplot import figure as fig
import numpy as np
# Exercise 1: Defining functions
def f(x):
    return np.cos(x)*np.exp(x)

# Exercise 2: Plotting data
def plot_f(lower_interval, upper_interval):
    x = np.linspace(lower_interval, upper_interval, 1000)
    #y = list(map(f, x))
    y = f(x)
    plt.plot(x, y)
    plt.savefig("f_function.png")
    #plt.show()

# Exercise 3: Generating random numbers
def random():
    # Same numbers every time:
    np.random.seed(124)
    a = np.random.normal(5, 2, 100000)
    fig()
    plt.hist(a,100, density=True)
    b = np.random.uniform(0, 10, 100000)
    fig()
    plt.hist(b,100, density=True)
    #plt.show()
    print("a mean:", np.mean(a))
    print("a variance:", np.std(a))
    print("b mean:", np.mean(b))
    print("b variance:", np.std(b))


if __name__ == '__main__':
    print(f(0))
    plot_f(-2*np.pi, 2*np.pi)
    random()