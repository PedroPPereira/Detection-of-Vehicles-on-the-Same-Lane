

from matplotlib import pyplot as plt
from   tkinter import *
from  tkinter import messagebox
import pyopencl as cl
import math
import cv2 as cv
import numpy as np



#Show two images side by side
# receives the images and a BGR boolean to specify if the image is BGR or RGB format
def showSideBySideImages(img1, img2, title1, title2, title="", BGR1=True, BGR2=True):
    img1 = img1 if BGR1 else cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = img2 if BGR2 else cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img1)
    ax.set_title(title1)
    plt.axis("off")
    if (len(img1.shape) < 3): #grayscale
        imgplot.set_cmap('gray')

    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img2)
    ax.set_title(title2)
    plt.axis("off")
    if (len(img2.shape) <3): #grayscale
        imgplot.set_cmap('gray')
    plt.show()
    


def showImage(img1):
    fig = plt.figure('Img1')
    imgplot = plt.imshow(img1)

    plt.axis("off")


def showMessageBox(title, message):
    app = Tk()
    app.withdraw()
    messagebox.showinfo(title, message=message)









class OpenCL_init:
  def __init__(self, in1):
    try:
        # setup the Kernel
        plaforms = cl.get_platforms()
        plaform = plaforms[0]
        devices = plaform.get_devices()
        device = devices[0]
        self.ctx = cl.Context(devices)  # or dev_type=cl.device_type.ALL)
        self.commQ = cl.CommandQueue(self.ctx, device)
        # build the Kernel
        file = open(in1.fileCL, "r")
        self.prog = cl.Program(self.ctx, file.read())
        self.prog.build()
        #launch config
        self.workItemSize = (in1.blockSize, in1.blockSize)
        self.workGroupSize = (math.ceil(in1.width / in1.blockSize) * in1.blockSize,
                        math.ceil(in1.height / in1.blockSize) * in1.blockSize)
        self.imgFormat = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)
    except Exception as e:
        print('EXCEPTION setup_opencl')
        print(e)
        





def grayscale(gpu, height, width, img1):
    try:
        # select kernel function
        kernelName = gpu.prog.Grayscale
        # create image objects
        imgFormat = cl.ImageFormat(
            cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)
        bufferFilter = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = imgFormat, shape = (width, height),
            pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferFilterOut = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format = imgFormat, shape = (width, height),
            pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        # set kernel arguments
        kernelName.set_arg(0, bufferFilter)
        #kernelName.set_arg(1, np.int32(width))
        #kernelName.set_arg(2, np.int32(height))
        kernelName.set_arg(1, bufferFilterOut)

        
        # launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        # copy the arrays from the device to the Host and show them on screen
        tmp_out = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, tmp_out, bufferFilterOut, origin = (0, 0), region = (width, height), is_blocking = True)
        bufferFilterOut.release()
        return tmp_out
    
    except Exception as e:
        print('EXCEPTION grayscale')
        print(e)





















