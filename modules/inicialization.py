
import math
import numpy as np



class Init:
  def __init__(self, image, boolImage):
    
    # OpenCL configuration
    self.blockSize = 32
    self.fileCL = "prog.cl" 
    
    # Image specifications
    if(boolImage):
        self.height, self.width = image.shape[:2]
        self.padding = image.strides[0] - image.strides[1] * self.width
    else:
       self.height = 1080
       self.width = 1440
       self.padding = 0
    
    # Canny inputs
    self.lowThresh = 50  
    self.highThresh = 70 
    
    # Hough inputs 
    self.thetas1 = np.arange(0, 350, 1)
    self.thetas = np.concatenate((self.thetas1[20:55], self.thetas1[305:340])) #angle iteration
    self.lowH = int(0.40*self.height) 
    self.highH = int(0.84*self.height)
    self.minAcc = 15 #min value of HS position to be accepted
    self.maxRho = 800 
    self.minRho = 255
    
    
    # Viola Jone inputs
    self.classifier = "images/cars.xml"
    self.scale_factor = 1.1
    self.minNeighbors = 4
    self.min_size = (60, 60)
    self.highWindow = int(0.8*self.height)
    
    
    
    #----------------------------------------------------------------------------
    # Gaussian Blur constants
    self.arrayGaussian = np.array([[0.0625, 0.125, 0.0625], [0.1250, 0.250, 0.1250], [0.0625, 0.125, 0.0625]], dtype=float).flatten()
    self.maskSize = 3
    
    # Hough constants
    self.arrayMaxLeft = np.zeros([3], dtype=np.int32)
    self.arrayMaxRight = np.zeros([3], dtype=np.int32)
    self.arrayLines = np.empty([8], dtype=np.int32)
    self.lin_left_old  = np.zeros([3], dtype=np.int32)
    self.lin_right_old = np.zeros([3], dtype=np.int32)
    
    self.diag_len = round(math.sqrt(self.height**2 + self.width**2))*2
    self.cos_thetas = np.cos(np.deg2rad(self.thetas), dtype=np.float32)
    self.sin_thetas = np.sin(np.deg2rad(self.thetas), dtype=np.float32)
    self.N = len(self.thetas)    
    self.accumulator = np.zeros((self.diag_len, self.N), dtype=np.int32).flatten() 
    

    
    
