## Course Info

* Foundations of Applied Mathematics
* BYU
* 2017
* https://foundations-of-applied-mathematics.github.io/

## Setup

1. `mkvirtualenv --python=/usr/local/bin/python3 byu-applied-math`
1. `pip3 install --upgrade pip`
1. `pip3 install -r requirements.txt`

## Matplotlib

You may have an issue importing matplotlib on Mac: `Python is not installed as a framework`.
See https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python for resolution.

1. Make file `~/.matplotlib/matplotlibrc`
1. Add code: `backend: TkAgg`

## Jupyter

Useful commands:

* Start server: `jupyter notebook`
* Command palette: `Command + Shift + P`
* Run section: `Control + Enter`
