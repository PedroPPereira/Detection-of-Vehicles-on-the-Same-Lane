# -*- coding: utf-8 -*-
#!/usr/bin/python
import pyopencl as cl
import numpy as np



##############################################################################################################################
#0. Class Sobel
def sobel_threshold(gpu, in1, img1, threshold_1, threshold_2):
    try:
        # select kernel function
        kernelName = gpu.prog.sobel_threshold
        # create image objects
        imgFormat = cl.ImageFormat(
            cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)
        bufferFilter = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = imgFormat, shape = (in1.width, in1.height),
            pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferFilterOut = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format = imgFormat, shape = (in1.width, in1.height),
            pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        # set kernel arguments
        kernelName.set_arg(0, bufferFilter)
        kernelName.set_arg(1, np.int32(in1.width))
        kernelName.set_arg(2, np.int32(in1.height))
        kernelName.set_arg(3, bufferFilterOut)
        kernelName.set_arg(4, np.int32(threshold_1))
        kernelName.set_arg(5, np.int32(threshold_2))
        
        # launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        # copy the arrays from the device to the Host and show them on screen
        tmp_out = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, tmp_out, bufferFilterOut, origin = (0, 0), region = (in1.width, in1.height), is_blocking = True)
        bufferFilterOut.release()
        return tmp_out
    
    except Exception as e:
        print('EXCEPTION sobel_threshold')
        print(e)










##############################################################################################################################
#1. Gaussian Blur
def gaussian_blur(gpu, in1, img1):
    try:
        #select kernel function
        kernelName = gpu.prog.gaussian_blur
        
        #create image objects
        bufferImg = cl.Image(gpu.ctx, flags = cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferImgOut = cl.Image(gpu.ctx, flags = cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferGaussian = cl.Buffer(gpu.ctx,flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=in1.arrayGaussian)
        
        #set kernel arguments
        kernelName.set_arg(0, bufferImg)
        kernelName.set_arg(1, bufferGaussian)
        kernelName.set_arg(2, bufferImgOut)
        kernelName.set_arg(3, np.int32( (in1.maskSize-1)/2 ))
        kernelName.set_arg(4, np.int32(in1.width))
        kernelName.set_arg(5, np.int32(in1.height))
        
        #launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, 
                                                 local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        
        #get Device objects to the Host
        img_empty = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, img_empty, bufferImgOut,  origin = (0,0), region = (in1.width, in1.height), is_blocking=True)
        
        #release buffers
        bufferImg.release()
        bufferImgOut.release()
        bufferGaussian.release()
        return img_empty
    
    except Exception as e:
        print("EXCEPTION gaussian_blur")
        print(e)






##############################################################################################################################
#2. Sobel
def sobel_transform(gpu, in1, img1):
    try:
        # select kernel function
        kernelName = gpu.prog.sobel_filter
        
        # create image objects
        bufferImg = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferImgOut = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferTheta = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, img1.nbytes)
        
        # set kernel arguments
        kernelName.set_arg(0, bufferImg)
        kernelName.set_arg(1, np.int32(in1.width))
        kernelName.set_arg(2, np.int32(in1.height))
        kernelName.set_arg(3, bufferImgOut)
        kernelName.set_arg(4, bufferTheta)
        kernelName.set_arg(5, np.int32(in1.padding)) 
        
        # launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        
        #get Device objects to the Host
        img_empty = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, img_empty, bufferImgOut, origin = (0, 0), region = (in1.width, in1.height), is_blocking = True)
        img_empty_theta = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, img_empty_theta, bufferTheta)
        
        #release buffers
        bufferImg.release() 
        bufferImgOut.release() 
        bufferTheta.release()        
        return img_empty, img_empty_theta
    
    except Exception as e:
        print('EXCEPTION sobel_transform')
        print(e)







##############################################################################################################################
#3. Non Max Supression
def non_max_suppression(gpu, in1, img1, theta):
    try:
        # select kernel function
        kernelName = gpu.prog.non_max_suppression
        
        # create image objects
        bufferImg = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferImgOut = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format = gpu.imgFormat, 
            shape = (in1.width, in1.height), pitches = (img1.strides[0], img1.strides[1]), hostbuf = img1.data)
        bufferTheta = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf = theta)
        
        # set kernel arguments
        kernelName.set_arg(0, bufferImg)
        kernelName.set_arg(1, np.int32(in1.width))
        kernelName.set_arg(2, np.int32(in1.height))
        kernelName.set_arg(3, bufferImgOut)
        kernelName.set_arg(4, bufferTheta)
        kernelName.set_arg(5, np.int32(in1.padding)) 
        kernelName.set_arg(6, np.int32(in1.lowThresh))
        kernelName.set_arg(7, np.int32(in1.highThresh)) 
        
        # launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        
        #get Device objects to the Host
        img_empty = np.empty_like(img1)
        cl.enqueue_copy(gpu.commQ, img_empty, bufferImgOut, origin = (0, 0), region = (in1.width, in1.height), is_blocking = True)
        
        #release buffers
        bufferImg.release()
        bufferImgOut.release() 
        bufferTheta.release()
        return img_empty
    
    except Exception as e:
        print('EXCEPTION non_max_suppression')
        print(e)
        
        
        
        
        
        
        