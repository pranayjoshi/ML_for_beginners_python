import numpy as np
import matplotlib.pyplot as plt 

# testing objects
greyhounds = 500
labs = 500

# testing data height
grey_height = 28+4*np.random.randn(greyhounds)
lab_height = 24+4*np.random.randn(labs)
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()