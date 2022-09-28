# -*- coding: utf-8 -*-
#!/usr/bin/python
import pyopencl as cl
import numpy as np
import cv2 as cv
import math



def hough_transform(gpu, in1, edge_image):
    try:
        #select kernel function
        kernelName = gpu.prog.hough_transform_polar
        
        #create image objects
        buffImage = cl.Image(gpu.ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format = gpu.imgFormat,
            shape = (in1.width, in1.height), pitches = (edge_image.strides[0], edge_image.strides[1]), hostbuf = edge_image.data)
        buffCos = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf = in1.cos_thetas)
        buffSen = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf = in1.sin_thetas)
        buffAcc = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = in1.accumulator)
        buffMaxLeft = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = in1.arrayMaxLeft)
        buffMaxRight = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = in1.arrayMaxRight)
        
        #set kernel arguments
        kernelName.set_arg(0, buffImage)
        kernelName.set_arg(1, np.int32(in1.width))
        kernelName.set_arg(2, np.int32(in1.height))
        kernelName.set_arg(3, buffCos)
        kernelName.set_arg(4, buffSen)
        kernelName.set_arg(5, np.int32(in1.N))
        kernelName.set_arg(6, buffAcc)
        kernelName.set_arg(7, np.int32(in1.lowH))
        kernelName.set_arg(8, np.int32(in1.highH))
        kernelName.set_arg(9, buffMaxLeft)
        kernelName.set_arg(10, buffMaxRight)  
        kernelName.set_arg(11, np.int32(in1.maxRho))
        kernelName.set_arg(12, np.int32(in1.minRho))

        #launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = gpu.workGroupSize, local_work_size = gpu.workItemSize)
        kernelEvent.wait()
        
        #get Device objects to the Host
        out_left = np.empty_like(in1.arrayMaxLeft)
        cl.enqueue_copy(gpu.commQ, out_left, buffMaxLeft)
        out_right = np.empty_like(in1.arrayMaxRight)
        cl.enqueue_copy(gpu.commQ, out_right, buffMaxRight)


        #release buffers
        buffImage.release()
        buffCos.release()
        buffSen.release()
        buffMaxLeft.release()
        buffMaxRight.release()
        buffAcc.release()
        return out_left, out_right
    
    except Exception as e:
        print('EXCEPTION hough_transform')
        print(e)








def hough_vote(gpu, lin_left, lin_right, in1, image):
    try:
        #select kernel function
        kernelName = gpu.prog.hough_vote      
        
        #create objects
        buffLeft = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = lin_left)
        buffRight = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = lin_right)
        buffCos = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf = in1.cos_thetas)
        buffSen = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf = in1.sin_thetas)
        bufferLines = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, in1.arrayLines.nbytes)
        buffMaxLeftOld  = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = in1.lin_left_old)
        buffMaxRightOld = cl.Buffer(gpu.ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf = in1.lin_right_old)

        #set kernel arguments
        kernelName.set_arg(0, buffLeft)
        kernelName.set_arg(1, buffRight)
        kernelName.set_arg(2, buffCos)
        kernelName.set_arg(3, buffSen)
        kernelName.set_arg(4, bufferLines)
        kernelName.set_arg(5, buffMaxLeftOld)
        kernelName.set_arg(6, buffMaxRightOld)
        kernelName.set_arg(7, np.int32(in1.minAcc))

        #launch the kernel
        kernelEvent = cl.enqueue_nd_range_kernel(gpu.commQ, kernelName, global_work_size = (math.ceil(10 / 5) * 5 + 5, 1), 
                    local_work_size = (5, 1))
        kernelEvent.wait()
        
        #get Device objects to the Host
        array_line = np.empty_like(in1.arrayLines)
        cl.enqueue_copy(gpu.commQ, array_line, bufferLines)
        cl.enqueue_copy(gpu.commQ, in1.lin_left_old, buffMaxLeftOld)
        cl.enqueue_copy(gpu.commQ, in1.lin_right_old, buffMaxRightOld)
        
        
        #release buffers
        buffLeft.release()
        buffRight.release()
        buffCos.release()
        buffSen.release()
        bufferLines.release()
        buffMaxLeftOld.release()
        buffMaxRightOld.release()
        
        #draw lines
        cv.line(image, (array_line[0], array_line[1]), (array_line[2], array_line[3]), (255, 0, 0), 3)
        cv.line(image, (array_line[4], array_line[5]), (array_line[6], array_line[7]), (255, 0, 0), 3)
        return image, array_line
    
    except Exception as e:
        print('EXCEPTION hough_vote')
        print(e)









