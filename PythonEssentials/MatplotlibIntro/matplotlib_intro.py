# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Ben Simmons
Self Study
9/25/2017
"""
import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))
    row_means = np.mean(A, axis=1)
    return np.var(row_means)

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    xa = np.arange(100, 1001, 100)
    # ya = np.array(list(map(lambda x: var_of_means(x), xa)))
    ya = np.vectorize(lambda x: var_of_means(x))(xa)
    plt.plot(xa, ya)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))
    plt.plot(x, np.arctan(x))
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    y = np.vectorize(lambda x: 1/(x-1))
    x1 = np.linspace(-2, 1, 100, endpoint=False)
    x2 = np.linspace(1.01, 6, 100)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.plot(x1, y(x1), 'm--', linewidth=4)
    plt.plot(x2, y(x2), 'm--', linewidth=4)
    plt.show()


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2*np.pi, 100)

    plt.suptitle("Sine functions")

    ax1 = plt.subplot(221)
    ax1.set_title("sin(x)")
    ax1.axis([0, 2*np.pi, -2, 2])
    ax1.plot(x, np.sin(x), 'g-')

    ax2 = plt.subplot(222)
    ax2.set_title("sin(2x)")
    ax2.axis([0, 2*np.pi, -2, 2])
    ax2.plot(x, np.sin(2*x), 'r--')

    ax3 = plt.subplot(223)
    ax3.set_title("2sin(x)")
    ax3.axis([0, 2*np.pi, -2, 2])
    ax3.plot(x, 2*np.sin(x), 'b--')

    ax4 = plt.subplot(224)
    ax4.set_title("2sin(2x)")
    ax4.axis([0, 2*np.pi, -2, 2])
    ax4.plot(x, 2*np.sin(2*x), 'm:')

    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    # each row is [hour, long, lat]
    fars = np.load("FARS.npy")

    # scatter plot looks like the usa
    ax1 = plt.subplot(121)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal")
    ax1.plot(fars[:,1], fars[:,2], 'ko')

    # deadliest hour is 18 (6 p.m.) with ~8400 fatalities
    ax2 = plt.subplot(122)
    ax2.set_xlabel("Hour of the day")
    ax2.hist(fars[:,:1], bins=np.arange(-0.5, 24.5))

    plt.show()


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y) / (X*Y)

    plt.suptitle("g(x) = sin(x)sin(y)/xy")

    # heatmap
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="viridis")
    plt.axis([-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    plt.axis("equal")
    plt.colorbar()

    # colored contour map with 100 level curves
    plt.subplot(122)
    plt.contourf(X, Y, Z, 100, cmap="magma")
    plt.axis([-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    plt.axis("equal")
    plt.colorbar()

    plt.show()
