## PG reconstruction
import matplotlib.pyplot as plt
import numpy as np

#Importing the fft and inverse fft functions from fftpackage
from scipy import fftpack

#For convolutions
from scipy import ndimage
from astropy.io import fits
import os, sys

class Image(object):
    def __init__(self, image, mask=0, reconstructed=False, \
                freq_cut=0, fits_header=None):

        self.img = image

        if mask == 0:
            self.mask = Mask(np.zeros(image.shape))
        elif isinstance(mask, Mask):
            self.mask = mask
        elif isinstance(mask,(np.ndarray, np.generic)):
            self.mask = Mask(mask)

        self.reconstructed = reconstructed

        if freq_cut == 0: freq_cut=image.shape[0]/2
        self.freq_cut = freq_cut


    def from_file(filename, nanzero=True):
        cube=fits.open(filename)#data_dir+cube_name)
        img=cube[0].data
        if nanzero == True:
            min_v=np.nanmin(img)
            for i in range(0,cube_size[0]):
                for j in range(0,cube_size[1]):
                    if img[i,j] > min_v:
                        pass
                    else:
                        img[i,j]=0
        return Image(img,fits_header=cube[0].header)

    def recon_img(image, window, smoothed_window, f_window):
        mask=np.ones(window.shape)
        mask -= window

        ####multiply the window with generated
        i_a = image * window

        #Apply smoothed window to the observation
        smoothed_input = i_a * smoothed_window * window

        #Caluculating the Fourier components
        c0=fftpack.fft2(smoothed_input)
        est=fftpack.ifft2(c0)

        i=0
        while i<1500: # Iteration limiting condition
            arr=(est*mask)+(i_a*window)
            f_est = fftpack.fft2(arr)
            f_est = fftpack.fftshift(f_est)*f_window
            f_est = fftpack.ifftshift(f_est)
            est=fftpack.ifft2(f_est)
            i+=1

        o_a=est.astype(float)
        return o_a

    def reconstruct(img_obj):
        if not isinstance(img_obj, image):
            return print("invalid object type")

        if not((img_obj.Mask.mask).all()):
            return print("enter mask")

        if img_obj.img.ndim == 2:
            shp_x, shp_y = img_obj.img.shape
            f_wind = np.ones(shp_x, shp_y)

            for (y,x), value in np.ndenumerate(f_wind):
                if (x-shp_x+1)**2+(y-shp_y+1)**2>img_obj.freq_cut**2:
                    f_wind[y,x]=0

            reconstructed_img = recon_img(img_obj.img, img_obj.mask.window, \
                                        img_obj.mask.s_wind, f_wind)

        else: print("Invalid number of dimensions")

        return Image(reconstructed_img, img_obj.mask, reconstructed=True, \
                    img_obj.freq_cut)

    def save_image(img_obj, filename, filedir="", filetype="fits"):
        if filetype == "fits":
            hdu=fits.PrimaryHDU(img_obj.img[:,:])
            hdu.header=orig_header
            im=fits.HDUList([hdu])
            save_file = filedir + filename + ".fits"
            im.writeto(save_file, clobber=True)

class Mask(object):
    '''

    mask
    error_arr
    nan_arr

    from image
    define error line


    '''

    def __init__(self, wind_arr, error_arr=0, nan_arr=0):

        self.window = wind_arr
        self.s_wind = smooth_boundary(wind_arr)
        self.error = error_arr #0 for error line
        self.nan = nan_arr #0 for nan

    def find_mask(window):
        mask = np.ones(window.shape)
        mask -= window
        return mask

    def smooth_boundary(i_a, sigma=2):
        shp_x, shp_y = i_a.shape
        x, y = np.meshgrid(np.linspace(-shp_x,shp_x,2*shp_x), \
                            np.linspace(-shp_y,shp_y,2*shp_y))
        sigma_x, sigma_y = sigma, sigma
        gauss = (np.exp(-( (x**2)/(2.0*sigma_x**2) + (y**2)/(2.0*sigma_y**2) ))
                    /(2*np.pi*sigma_x*sigma_y))

        #Generating a smoothed window
        o_a=np.copy(i_a)
        i=0
        while i<5:
            o_a = ndimage.convolve((o_a*i_a), gauss, mode='nearest')
            i+=1
        return o_a

    def error_line(mask_obj, err_line, elwidth = 1):
        mask_obj.error=np.ones(mask_obj.window.shape)
        err_line_min = err_line - elwidth
        mask_obj.error[err_line_min:err_line][:]=0
        mask_obj.window[err_line_min:err_line][:]=0

        self.s_wind = smooth_boundary(mask_obj.window)
        return pass

    def nan_arr_from_image(filename, nanzero=True):
        cube=fits.open(filename)#data_dir+cube_name)
        img=cube[0].data
        nan_arr = np.ones(img.shape)
        if nanzero == True:
            min_v=np.nanmin(img)
            for i in range(0,cube_size[0]):
                for j in range(0,cube_size[1]):
                    if img[i,j] > min_v:
                        pass
                    else:
                        nan_arr[i,j]=0
        return nan_arr
