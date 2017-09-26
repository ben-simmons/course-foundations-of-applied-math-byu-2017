## Course Info

* Foundations of Applied Mathematics
* BYU
* 2017
* https://foundations-of-applied-mathematics.github.io/

## Setup

1. `mkvirtualenv --python=/usr/local/bin/python3 byu-applied-math`
1. `pip3 install --upgrade pip`
1. `pip3 install numpy`
1. `pip3 install matplotlib`
1. `pip3 install sympy`
1. `pip3 install jupyter`

## Matplotlib

You may have an issue importing `matplotlib`: `Python is not installed as a framework`.
See https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python for resolution.

1. Make file `~/.matplotlib/matplotlibrc`
1. Add code: `backend: TkAgg`
