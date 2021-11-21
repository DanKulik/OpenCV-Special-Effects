import cv2
import numpy as np
from wand.image import Image


class draw:

    @staticmethod
    def ink(image):
        """
        Create pen style drawing of image
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255-gray
        blur = cv2.GaussianBlur(inv, (21,21), 0)
        inv = 255-blur
        gray = cv2.divide(gray, inv, scale=255.0)
        gray[gray<255] = np.uint8(gray[gray<255]-30)
        gray[gray<200] = 15

        return gray

    @staticmethod
    def inknshadow(image):
        """
        Create pen style drawing of image with black shading
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = draw.ink(image)
        image[gray<85] = 15
        
        return image

    @staticmethod
    def manga(image, screentones=True, second_filter=True):
        """
        Create manga style drawing of image
        input: image (3D numpy array uint8)
               screentones (Bool: True adds screentones,
                                  False uses grayscale and poisson noise;
                            default=True)
               secondfilter (Bool: True add white values, False all screentone;
                             default=True)
        return: (3D numpy array uint8)
        """

        tile1 = np.rint(image.shape[1]/8).astype(int)
        tile2 = np.rint(image.shape[0]/4).astype(int)
        image = cv2.resize(image,(tile1*8,tile2*4),interpolation=cv2.INTER_LANCZOS4)
          
        gray = draw.inknshadow(image)

        if screentones:
            
            kernel = np.array([[1,1,1,0,0,1,1,1],
                               [1,1,1,0,0,1,1,1],
                               [0,1,1,1,1,1,1,0],
                               [0,1,1,1,1,1,1,0]])
            
            kernel = np.tile(kernel,tile1)
            repetitions = tile2
            kernel = np.tile(kernel,(repetitions, 1))

            filter1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

            filter1 = filter1*kernel
            filter1[filter1<45] = 15

            filter2 = filter1.copy()
            if second_filter:
                filter2[filter1==15] = 255
            
            avg = np.uint8(np.average(gr))
            if (avg<70):
                gray[gr<60] = 15
            else:
                gray[gr<85] = 15

                    
            gray[gray>15] = filter1[gray>15]
            mask = np.where((gr>140) & (gr<=180))
            gray[mask] = filter2[mask]

            if (avg>140) & (avg<160):
                gray[gr>175] = 255
            else:
                gray[gr>167] = 255

        else:

            weight = 0.6

            fr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with Image.from_array(fr) as img:
                img.noise("poisson", attenuate = 0.99)
                poissonNoise = np.asarray(img)
            poissonNoise = cv2.cvtColor(poissonNoise, cv2.COLOR_RGB2BGR)
        
            gr = np.uint8(image*weight + poissonNoise*(1-weight))
            gr = cv2.cvtColor(gr, cv2.COLOR_BGR2GRAY)

            gr[gray<=15] = gray[gray<=15]
            gr[gr>175] = 255
            gray = gr

        return gray

    @staticmethod
    def manga_color(image, screentones=True, second_filter=True, bg=255):
        """
        Create color manga style drawing of image
        input: image (3D numpy array uint8)
               screentones (Bool: True adds screentones,
                                  False uses grayscale and poisson noise;
                            default=True)
               secondfilter (Bool: True add white values, False all screentone;
                             default=True)
               bg (Set background color when using screentones; default=255)
        return: (3D numpy array uint8)
        """

        if bg>255:
            bg = 255
        elif bg<0:
            bg =0
            
        tile1 = np.rint(image.shape[1]/8).astype(int)
        tile2 = np.rint(image.shape[0]/4).astype(int)
        image = cv2.resize(image,(tile1*8,tile2*4),interpolation=cv2.INTER_LANCZOS4)

        gray = draw.manga(image, screentones, second_filter)
        new = np.ones((image.shape), dtype=np.uint8)*bg

        gray_col = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        new[gray<255] = image[gray<255]
        new[gray<=15] = gray_col[gray<=15]

        return new

    @staticmethod
    def centerfy(image):
        """
        Create pen style drawing of image and add color using
        a splotch color finder to paint image by finding color centers
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        inked = draw.ink(image)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros(img.shape, dtype=np.uint8)
        
        q1 = np.quantile(img, 0.125)
        q2 = np.quantile(img, 0.25)
        q3 = np.quantile(img, 0.375)
        q4 = np.quantile(img, 0.50)
        q5 = np.quantile(img, 0.625)
        q6 = np.quantile(img, 0.75)
        q7 = np.quantile(img, 0.375)

        mask1 = np.where((img>q1) & (img<q2))
        mask2 = np.where((img>q2) & (img<q3))
        mask3 = np.where((img>q3) & (img<q4))
        mask4 = np.where((img>q4) & (img<q5))
        mask5 = np.where((img>q5) & (img<q6))
        mask6 = np.where((img>q6) & (img<q7))

        img2[img>q7] = 255
        img[mask6] = 219
        img[mask5] = 183
        img[mask4] = 147
        img[mask3] = 111
        img[mask2] = 75
        img[mask1] = 39
        img[img<q1] = 0
        
        ret,thresh = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        score = np.ones(image.shape)*[255,255,255]

        for cnt in contours:
            
            area = cv2.contourArea(cnt)
            
            if (area>250.0)&(area<120000):
                
                test = np.zeros(img.shape)
                
                M = cv2.moments(cnt)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                
                cv2.drawContours(test, [cnt], -1, 255,-1)

                mask = np.where(test==255)
                dist = np.sqrt(np.max(np.abs(y-mask[0]))**2 + np.max(np.abs(x-mask[1]))**2)
                xx = mask[0]
                yy = mask[1]
                
                for j in range(len(xx)):
                    
                    far = np.sqrt((y-xx[j])**2 + (x-yy[j])**2)
                    weight = far/dist #*1.5
                    image[xx[j],yy[j]] = np.uint8(image[y,x]*(1-weight) + score[xx[j],yy[j]]*(weight)) 

        image[inked<=15] = [0,0,0]         

        return image


    @staticmethod
    def cartoonize(image):
        """
        Create pen style drawing of image and color with edgepreserving
        input: image (3D numpy array uint8)
        return: (3D numpy array uint8)
        """

        mask = draw.ink(image)
        mask[mask<=15] = 0
        mask[mask>15] = 255

        dst = cv2.edgePreservingFilter(image, flags=1, sigma_s=130, sigma_r=0.95)
        image = cv2.bitwise_and(dst, dst, mask=mask)

        return image
        
        

        

