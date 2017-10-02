# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Ben Simmons
Self Study
10/2/2017
"""
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt


# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x, y = sy.symbols('x y')
    expr = sy.Rational(2, 5) * sy.exp(x**2 - y) * sy.cosh(x + y) + sy.Rational(3, 7) * sy.log(x*y + 1)
    return expr


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x, i, j = sy.symbols('x i j')
    expr = sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5))
    return sy.simplify(expr)


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    x, y, n = sy.symbols('x y n')
    expr1 = sy.summation(x**n / sy.factorial(n), (n, 0, N))
    expr2 = expr1.subs({x: -y**2})
    f = sy.lambdify(y, expr2, 'numpy')

    x_vals = np.linspace(-3, 3, 100)
    plot1, = plt.plot(x_vals, np.exp(-x_vals**2), alpha=.75, label="e^(-x**2)")
    plot2, = plt.plot(x_vals, f(x_vals), alpha=.75, label="Mac e^(-x**2)")
    plt.legend(loc='upper right', handles=[plot1, plot2])
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, theta = sy.symbols('x y r theta')
    expr = 1 - ((x**2 + y**2)**sy.Rational(7/2) + 18*x**5 * y - 60*x**3 * y**3 + 18*x * y**5) / (x**2 + y**2)**3
    simplified_expr = sy.simplify(expr)
    polar_expr = simplified_expr.subs({x: r * sy.cos(theta), y: r * sy.sin(theta)})
    r_solutions = sy.solve(polar_expr, r)
    r_func = sy.lambdify(theta, r_solutions[0], 'numpy')

    theta_vals = np.linspace(0, 2*np.pi, 1000)
    x_func = lambda t: r_func(t) * np.cos(t)
    y_func = lambda t: r_func(t) * np.sin(t)
    plt.plot(x_func(theta_vals), y_func(theta_vals))
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y = sy.symbols('x y')
    A = sy.Matrix([[x-y,   x,   0],
                   [  x, x-y,   x],
                   [  0,   x, x-y]])
    return dict([(e[0], e[2]) for e in A.eigenvects()])


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols('x')
    expr = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    d1 = sy.diff(expr, x)
    d2 = sy.diff(expr, x, x)
    critical_points = sy.solve(d1, x)
    minima = [x0 for x0 in critical_points if d2.evalf(subs={x: x0}) > 0]
    maxima = [x0 for x0 in critical_points if d2.evalf(subs={x: x0}) < 0]

    p = sy.lambdify(x, expr, 'numpy')
    x_vals = np.linspace(-5, 5, 100)
    plt.plot(x_vals, p(x_vals))
    plt.plot(minima, p(np.array(minima)), 'r.', markersize=10)
    plt.plot(maxima, p(np.array(maxima)), 'b.', markersize=10)
    plt.show()

    return set(minima), set(maxima)


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x, y, z, rho, phi, theta, r = sy.symbols('x y z rho phi theta r')
    f = (x**2 + y**2 + z**2)**2
    h1 = rho * sy.sin(phi) * sy.cos(theta)
    h2 = rho * sy.sin(phi) * sy.sin(theta)
    h3 = rho * sy.cos(phi)
    h = sy.Matrix([h1, h2, h3])
    J = h.jacobian([rho, theta, phi])
    integrand = sy.simplify(f.subs({x: h1, y: h2, z: h3}) * -J.det())
    v = sy.integrate(integrand, (rho, 0, r), (theta, 0, 2*sy.pi), (phi, 0, sy.pi))

    v_func = sy.lambdify(r, v, 'numpy')
    r_vals = np.linspace(0, 3, 100)
    plt.plot(r_vals, v_func(r_vals))
    plt.show()

    return v.subs({r: 2})
