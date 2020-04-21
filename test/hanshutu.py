import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import csv
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from mpl_toolkits.basemap import Basemap,cm

a = np.linspace(-10, 10)
b = 1/(1+np.exp(-a))
plt.grid()
plt.plot(a,b)
plt.show()
c = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
plt.grid()
plt.plot(a,c)
plt.show()