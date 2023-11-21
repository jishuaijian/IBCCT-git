import numpy as np

img = [[1,2],[3],[4,5]]

img = np.array(img).repeat(3,axis=0)
print(img)