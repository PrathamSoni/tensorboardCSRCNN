from skimage import io
import numpy as np

img1=np.asarray(io.imread('image2.png'))
img2=np.asarray(io.imread('image3.png'))
GT=np.asarray(io.imread('cropped.png'))
bicubic=np.asarray(io.imread('bicubic.png'))


#MSE
print(np.max(GT))
mse1=(np.sum(np.square(GT-img1)))/(252*252)
mse2=(np.sum(np.square(GT-img2)))/(252*252)
mse3=(np.sum(np.square(GT-bicubic)))/(252*252)
print('mse170',mse1)
print('mse1020',mse2)
print('msebicubic',mse3)
