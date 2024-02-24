import numpy as np
import matplotlib.pyplot as plt
import math

left = range(100)

prob = [0]*100
for i in range(100):
    prob[i] = 2.0*i/(100.0*(99.0))


vals = list(np.random.choice(100,size=1000000, p=prob))



height = [0]*100
for i in range(100):
    height[i] = vals.count(i)

plt.bar(left,height)

plt.show()