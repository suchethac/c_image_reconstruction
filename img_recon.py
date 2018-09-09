## PG reconstruction
import matplotlib.pyplot as plt
import numpy as np

#Importing the fft and inverse fft functions from fftpackage
from scipy import fftpack

#For convolutions
from scipy import ndimage
from astropy.io import fits
import os, sys

class Image():
    def __init__(self, image, fits_header=None):
        self.data = image

    def from_file(filename, nanzero=True):
        cube=fits.open(filename)#data_dir+cube_name)
        img=cube[0].data
        if nanzero == True:
            min_v=np.nanmin(img)
            for (i,j),val in np.ndenumerate(img):
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

    def f_window(img, freq_cut):
        shp_x, shp_y = img.shape
        f_wind = np.ones(shp_x, shp_y)
        for (y,x), value in np.ndenumerate(f_wind):
            if (x-shp_x+1)**2+(y-shp_y+1)**2>freq_cut**2:
                f_wind[y,x]=0
        return f_wind

    def reconstruct(img, msk, freq_cut=None):
        if not isinstance(img, Image):
            return print("invalid image object type")
        if not isinstance(msk, (Mask, MaskObj)):
            return print("invalid mask object type")
        if freq_cut is None: freq_cut=img.shape[0]/2

        s_wind = smooth_boundary(msk.window)
        f_wind = f_window(img.data, freq_cut)
        reconstructed_img = recon_img(img.data, msk.window, s_wind, f_wind)

        return ImageObj(reconstructed_img, msk, reconstructed=True, \
                    freq_cut=freq_cut, fits_header=img.fits_header)

    # def _reconstruct(img_obj):#Not used
    #     if not isinstance(img_obj, Image):
    #         return print("invalid object type")
    #
    #     if not((img_obj.Mask.mask).all()):
    #         return print("enter mask")
    #
    #     if img_obj.img.ndim == 2:
    #         shp_x, shp_y = img_obj.img.shape
    #         f_wind = np.ones(shp_x, shp_y)
    #
    #         for (y,x), value in np.ndenumerate(f_wind):
    #             if (x-shp_x+1)**2+(y-shp_y+1)**2>img_obj.freq_cut**2:
    #                 f_wind[y,x]=0
    #
    #         reconstructed_img = recon_img(img_obj.img, img_obj.mask.window, \
    #                                     img_obj.mask.s_wind, f_wind)
    #
    #     else: print("Invalid number of dimensions")
    #
    #     return ImageObject(reconstructed_img, img_obj.mask, reconstructed=True, \
    #                 img_obj.freq_cut)

    def save_image(img_obj, filename, filedir="", filetype="fits"):
        if filetype == "fits":
            hdu=fits.PrimaryHDU(img_obj.data[:,:])
            hdu.header=orig_header
            im=fits.HDUList([hdu])
            save_file = filedir + filename + ".fits"
            im.writeto(save_file, clobber=True)

class ImageObj(Image):
    """
    Contains the image and the necessary information from reconstruction.
    """
    def __init__(self, image, mask, reconstructed=False, \
                freq_cut=0, fits_header=None):

        super().__init__(image, fits_header=None)

        if isinstance(mask, (Mask, MaskObj)):
            self.mask = mask
        elif isinstance(mask,(np.ndarray, np.generic)):
            self.mask = Mask(mask)

        self.reconstructed = reconstructed

        if freq_cut == 0: freq_cut=image.shape[0]/2
        self.freq_cut = freq_cut


class Mask():
    '''

    mask
    error_arr
    nan_arr

    from image
    define error line


    '''

    def __init__(self, wind_arr):
        self.window = wind_arr

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

    def error_def_line(img, hline=None, vline=None):
        img_shp = img.shape
        error_array = np.ones(img_shp)

        if hline is not None:
            if isinstance(hline, int):
                hline_min = hline - 1
                error_array[hline_min:hline][:]=0
            elif isinstance(hline, list):
                hline_min = hline - 1
                for a,b in zip(hline_min, hline):
                    error_array[hline_min:hline][:]=0

        if vline is not None:
            if isinstance(vline, int):
                vline_min = vline - 1
                error_array[:][vline_min:vline]=0
            elif isinstance(vline, list):
                vline_min = vline - 1
                for a,b in zip(vline_min, vline):
                    error_array[:][vline_min:vline]=0

        return error_array


    def error_line(mask_obj, err_line, elwidth = 1):
        mask_obj.error=np.ones(mask_obj.window.shape)
        err_line_min = err_line - elwidth
        mask_obj.error[err_line_min:err_line][:]=0
        mask_obj.window[err_line_min:err_line][:]=0

        self.s_wind = smooth_boundary(mask_obj.window)

class MaskObj(Mask):
    def __init__(self, wind_arr, error_arr=None, nan_arr=None):
        super().__init__(wind_arr)
        self.error = error_arr #0 for error line
        self.nan = nan_arr

    def nan_arr_from_image(filename, nanzero=True):
        cube=fits.open(filename)#data_dir+cube_name)
        img=cube[0].data
        nan_arr = np.ones(img.shape)
        if nanzero == True:
            min_v=np.nanmin(img)
            for (i,j),val in np.ndenumerate(img):
                if img[i,j] > min_v:
                    pass
                else:
                    nan_arr[i,j]=0
        return nan_arr
