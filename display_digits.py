# Display images from the digits data set

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
digits = datasets.load_digits()

im = digits.images[1]
print(im.shape)
plt.gray() 
plt.matshow(digits.images[1]) 
plt.matshow(digits.images[2]) 
plt.show() 
