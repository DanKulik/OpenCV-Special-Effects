import cv2
import numpy as np
from wand.image import Image
from scipy.interpolate import UnivariateSpline
from perlin_numpy import generate_perlin_noise_2d


class draw:

    def create_LUT(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    @staticmethod
    def warm_col(image):
        """
        Warms colors in image
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        incr_ch_lut = draw.create_LUT([0, 69, 137, 199, 255], [0, 75, 144, 210, 255])
        decr_ch_lut = draw.create_LUT([0, 60, 115, 180, 255],[0, 50,  105, 170, 224])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        c_r, c_g, c_b = cv2.split(image)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb,  cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

        image = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
        
        return image

    def inc_satnbr(image, sat=1.28, br=1.03):
        """
        Increase the image saturation and brightness 
        input: image (3D numpy array uint8)
               sat (saturation value >=1.0; default=1.28)
               br (brightness value >=1.0; default=1.03)
        return: (3D numpy array uint8)
        """

        if sat<1.0:
            sat=1.0

        if br<1.0:
            br=1.0

        hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        mask = np.where((hsvImg[...,1]*sat)<=255.0)
        hsvImg[mask[0],mask[1],1] = hsvImg[mask[0],mask[1],1]*sat

        mask = np.where((hsvImg[...,2]*br)<=255.0)
        hsvImg[mask[0],mask[1],2] = hsvImg[mask[0],mask[1],2]*br

        image = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

        return image

    def sharpen(image, copy):
        """
        Sharpen image
        input: image (3D numpy array uint8)
               copy (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        image = cv2.addWeighted(cv2.GaussianBlur(image,(3,3),0), 1.5, cv2.GaussianBlur(copy,(21,21),0), -0.5,0)

        return image

    def thicken_lines(image):
        """
        Increase black line thickness around borders/corners
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255-gray
        blur = cv2.GaussianBlur(inv, (5,5), 0)
        inv = 255-blur
        gray = cv2.divide(gray, inv, scale=255.0)
        gray[gray<255] = np.uint8(gray[gray<255]-30)
        gray[gray<200] = np.uint8(15)
        image[gray<=15] = [5,5,5]

        return image

    def resize(image):
        """
        Resize image
        input: image (3D or 2D numpy array uint8)
        return: (3D numpy array uint8)
        """

        image = cv2.resize(image,(960,720),interpolation=cv2.INTER_LANCZOS4)
        return image

    def perlin_noise(image, weight=0.97):
        """
        Add perlin noise to image
        input: image (2D numpy array uint8)
               weight (Set normal image weight value; default=0.97)
        return: (3D numpy array uint8)
        """

        noise = generate_perlin_noise_2d((720, 960), (24, 24))
        fr = np.uint8((noise-np.min(noise))/(np.max(noise)-np.min(noise))*255)
        fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        image = np.uint8(image*weight + fr*(1-weight))

        return image

    @staticmethod
    def poisson_noise(image, weight=0.87):
        """
        Add poisson noise to image
        input: image (3D numpy array uint8)
               weight (Set normal image weight value; default=0.87)
        return: (3D numpy array uint8)
        """

        fr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with Image.from_array(fr) as img:
            img.noise("poisson", attenuate = 0.99)
            poissonNoise = np.asarray(img)
        poissonNoise = cv2.cvtColor(poissonNoise, cv2.COLOR_RGB2BGR)
        
        image = np.uint8(image*weight + poissonNoise*(1-weight))

        return image

    @staticmethod
    def chrom_abbr(image, px_shift=1):
        """
        Add chromatic abberation to image
        input: image (3D numpy array uint8)
               px_shift (pixel shift for colors; default=1)
        return: (3D numpy array uint8)
        """

        image[:,:,0] = np.roll(image[:,:,0], -px_shift*2)
        image[:,:,1] = np.roll(image[:,:,1], -px_shift)

        return image

    def col_squeeze(image):
        """
        Soften blackest and whitest colors
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        mask1 = np.where((image[:,:,0]>=242)&(image[:,:,2]>=242)&(image[:,:,2]>=242))
        mask2 = np.where((image[:,:,0]<=12)&(image[:,:,2]<=12)&(image[:,:,2]<=12))
        image[mask1] = [242,242,242]
        image[mask2] = [12,12,12]

        return image

    @staticmethod
    def std_run(image):
        """
        Run main sequence of retrofying image
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """
        copy = image
        image = draw.inc_satnbr(image)
        image = draw.warm_col(image)
        image = draw.sharpen(image, copy)
        image = draw.thicken_lines(image)
        image = draw.resize(image)
        image = draw.perlin_noise(image)
        image = draw.poisson_noise(image)
        image = draw.chrom_abbr(image)
        image = draw.col_squeeze(image)

        return image
        

        

