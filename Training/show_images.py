import matplotlib.pyplot as plt
import pylab # show the img in cli
import numpy as np

from dataset import *
# functions to show an image

def imshow(img):
    img = img / 2 + 0.5 # unormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# pring labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
pylab.show()