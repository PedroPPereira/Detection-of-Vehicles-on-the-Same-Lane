#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
  CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
  CLK_FILTER_NEAREST;







__kernel void hough_transform_polar(__read_only image2d_t image, 
                                      int w, int h,
                                      __global float* cos_thetas, __global float* sin_thetas,
                                      int N, __global int* accumulator, 
                                      int lowH, int highH,
                                      __global int* max_left, __global int* max_right, 
                                      int maxRho, int minRho)
{
    const int iX = get_global_id(0);
    const int iY = get_global_id(1);
    const int pixelValue = read_imageui(image, sampler, (int2)(iX,iY)).x;

    if (pixelValue == 255 && (iX >= 0) && (iX < w) && (iY >= lowH) && (iY < highH)) {
        int rho;
    
        for(int i = 0; i < N; i++) {
          //calculate distance for every angle
          rho = (int)round(iX*cos_thetas[i] + iY*sin_thetas[i]); 
          atomic_add(&accumulator[rho*N + i], 1);
          
          //best left line update 
          if(i < N/2 && accumulator[rho*N + i] > max_left[0] && rho < maxRho && rho > minRho) {
              max_left[0] = accumulator[rho*N + i];
              max_left[1] = rho;
              max_left[2] = i;
          }
          //best left line update 
          else if(i > N/2 && accumulator[rho*N + i] > max_right[0] && rho < maxRho && rho > minRho) {
              max_right[0] = accumulator[rho*N + i];
              max_right[1] = rho;
              max_right[2] = i;
          }           
        }
   }
}



__kernel void hough_vote(__global int* max_left, __global int* max_right, 
                        __global float* cos_thetas, __global float* sin_thetas,
                        __global int* array_lines,
                        __global int* max_left_old, __global int* max_right_old, 
                        int minAcc)
{      
    
    //use old value if the current one is bad
    if (max_left[0] > minAcc) { 
        for(int loop = 0; loop < 3; loop++) 
          max_left_old[loop] = max_left[loop];
    }
    else { 
        for(int loop = 0; loop < 3; loop++) 
          max_left[loop] = max_left_old[loop];    
    }
    
         
    if (max_right[0] > minAcc) { 
        for(int loop = 0; loop < 3; loop++) 
          max_right_old[loop] = max_right[loop];
    }
    else { 
        for(int loop = 0; loop < 3; loop++) 
          max_right[loop] = max_right_old[loop];    
    }
                  
    
    //convert from the Hough space to cartasian for both lines
    float a0 = cos_thetas[max_left[2]];
    float b0 = sin_thetas[max_left[2]];
    float x0 = a0*max_left[1];
    float y0 = b0*max_left[1];
    array_lines[0] = (int)(x0 + 2000*(-b0)); //x1
    array_lines[1] = (int)(y0 + 2000*(a0)); //y1
    array_lines[2] = (int)(x0 - 2000*(-b0)); //x2
    array_lines[3] = (int)(y0 - 2000*(a0)); //y2
            
    a0 = cos_thetas[max_right[2]];
    b0 = sin_thetas[max_right[2]];
    x0 = a0*max_right[1];
    y0 = b0*max_right[1];
    array_lines[4] = (int)(x0 + 2000*(-b0)); //x1
    array_lines[5] = (int)(y0 + 2000*(a0)); //y1
    array_lines[6] = (int)(x0 - 2000*(-b0)); //x2
    array_lines[7] = (int)(y0 - 2000*(a0)); //y2
}








//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------

__kernel void gaussian_blur(__read_only image2d_t image, __global float * mask, __write_only image2d_t imageOut, int maskSize, int w, int h)
{
    const int iX = get_global_id(0);
    const int iY = get_global_id(1);
    
    float sum = 0.0f;
    
    if ((iX >= 0)&&(iX < w) && (iY >= 0)&&(iY < h)) {
    
        for(int a = -maskSize; a < maskSize+1; a++) {
            for(int b = -maskSize; b < maskSize+1; b++) {
                sum += mask[a + maskSize + (b+maskSize)*(maskSize*2+1)]*read_imageui(image, sampler, (int2)(iX+a,iY+b)).z
                    + mask[a+1,b+1]*read_imageui(image, sampler, (int2)(iX+a, iY+b)).z;
            }
        }
        sum /= (maskSize*2 + 1)*(maskSize*2 + 1);
        
        write_imageui( imageOut, (int2)(iX, iY) , (uint4)((int)sum, (int)sum, (int)sum , 0) );
    }
}





__kernel void sobel_filter(__read_only image2d_t image, int w, int h, __write_only image2d_t imageOut, __global uchar* theta, int padding)
{
  const int iX = get_global_id(0);
  const int iY = get_global_id(1);

  if ((iX >= 0)&&(iX < w) && (iY >= 0)&&(iY < h)) {
    //----------------------------------  SOBEL  ----------------------------------
    //3x3 window
    uint pixelA = read_imageui( image, sampler, (int2)(iX-1,iY-1)).x;
    uint pixelB = read_imageui( image, sampler, (int2)(iX,iY-1)).x;
    uint pixelC = read_imageui( image, sampler, (int2)(iX+1,iY-1)).x;
    uint pixelD = read_imageui( image, sampler, (int2)(iX-1,iY)).x;
    uint pixelE = read_imageui( image, sampler, (int2)(iX,iY)).x;
    uint pixelF = read_imageui( image, sampler, (int2)(iX+1,iY)).x;
    uint pixelG = read_imageui( image, sampler, (int2)(iX-1,iY+1)).x;
    uint pixelH = read_imageui( image, sampler, (int2)(iX,iY+1)).x;
    uint pixelI = read_imageui( image, sampler, (int2)(iX+1,iY+1)).x;
    //Sx and Sy
    uint sumX = (pixelA+2*pixelD+pixelG) - (pixelC+2*pixelF+pixelI);
    uint sumY = (pixelG+2*pixelH+pixelI) - (pixelA+2*pixelB+pixelC);
    //absolute Sx and Sy and final sum
    uint sum = sqrt((float)(sumX * sumX + sumY * sumY));
    //bound SOBEL values
    if(sum > 255) sum = 255;
    //update pixel
    write_imageui( imageOut, (int2)(iX,iY) , (uint4)((int)sum, (int)sum, (int)sum , 0));

    //-----------------------------------------------------------------
    const float PI = 3.14159265;
    //compute the direction angle theta in radians
    float angle = atan2((float)sumY, (float)sumX);
    if (angle < 0)
        //shift the range to (0, 2PI) by adding 2PI to the angle, then perform modulo operation of 2PI
        angle = fmod((angle + 2*PI),(2*PI));
    //round the angle to one of four possibilities: 0, 45, 90, 135 degrees
    theta[iY * (w + padding) + iX] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
  }
}





__kernel void non_max_suppression(__read_only image2d_t image, int w, int h, __write_only image2d_t imageOut, 
                                    __global uchar* theta, int padding, int lowThresh, int highThresh)
{
  const int iX = get_global_id(0);
  const int iY = get_global_id(1);

  if ((iX >= 0) && (iX < w) && (iY >= 0) && (iY < h)) {
    uint my_magnitude = read_imageui( image, sampler, (int2)(iX,iY)).x;
    uint value = 0;

	//----------------------------------  NON MAX SUPRESSION  ----------------------------------
    switch (theta[iY * (w + padding) + iX])
    {
        // A gradient angle of 0 degrees = an edge that is North/South (Check neighbors East and West)
        case 0:
            if (my_magnitude <= read_imageui( image, sampler, (int2)(iX,iY+1)).x || // east
                my_magnitude <= read_imageui( image, sampler, (int2)(iX,iY-1)).x)   // west
                value = 0;
            else
                value = my_magnitude;
            break;  
        // A gradient angle of 45 degrees = an edge that is NW/SE (Check neighbors NE and SW)
        case 45:
            if (my_magnitude <= read_imageui( image, sampler, (int2)(iX-1,iY+1)).x || // north east
                my_magnitude <= read_imageui( image, sampler, (int2)(iX+1,iY-1)).x)   // south west
                value = 0;
            else
                value = my_magnitude;
            break;     
        // A gradient angle of 90 degrees = an edge that is E/W (Check neighbors North and South)
        case 90:
            if (my_magnitude <= read_imageui( image, sampler, (int2)(iX-1,iY)).x || // north
                my_magnitude <= read_imageui( image, sampler, (int2)(iX+1,iY)).x)   // south
                value = 0;
            else
                value = my_magnitude;
            break;     
        // A gradient angle of 135 degrees = an edge that is NE/SW (Check neighbors NW and SE)
        case 135:
            if (my_magnitude <= read_imageui( image, sampler, (int2)(iX-1,iY-1)).x || // north west
                my_magnitude <= read_imageui( image, sampler, (int2)(iX+1,iY+1)).x)   // south east
                value = 0;
            else
                value = my_magnitude;
            break;   
        default:
            value = my_magnitude;
            break;
    } 

    // ------------------------------------- DOUBLE THRESHOLD -------------------------------------
    if (value >= highThresh) //strong
        value = 255;
    else if (value <= lowThresh) //zeros
        value = 0;
    else //weak
    {
	//----------------------------------  HYSTERESIS  ----------------------------------
        if (value >= (highThresh + lowThresh)/2)
            value = 255;
        else
            value = 0;
    }
    write_imageui( imageOut, (int2)(iX,iY) , (uint4)((int)value, (int)value, (int)value , 0));
  }
}















__kernel void Grayscale(__read_only image2d_t input, __write_only image2d_t output) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int2 size = get_image_dim(input);

    if(all(gid < size)){
        uint4 pixel = read_imageui(input, sampler, gid);
        float4 color = convert_float4(pixel) / 255;
        color.xyz = 0.2126*color.x + 0.7152*color.y + 0.0722*color.z;
        pixel = convert_uint4_rte(color * 255);
        write_imageui(output, gid, pixel);
    }
}



__kernel void sobel_threshold(__read_only image2d_t image, int w, int h, __write_only image2d_t imageOut, int t1, int t2)
{
  int iX = get_global_id(0);
  int iY = get_global_id(1);

  if ((iX >= 0) && (iX < w) && (iY >= 0) && (iY < h)) {
    //----------------------------------  SOBEL  ----------------------------------
    //3x3 window
    uint4 pixelA = read_imageui( image, sampler, (int2)(iX-1,iY-1));
    uint4 pixelB = read_imageui( image, sampler, (int2)(iX,iY-1));
    uint4 pixelC = read_imageui( image, sampler, (int2)(iX+1,iY-1));
    uint4 pixelD = read_imageui( image, sampler, (int2)(iX-1,iY));
    uint4 pixelE = read_imageui( image, sampler, (int2)(iX,iY));
    uint4 pixelF = read_imageui( image, sampler, (int2)(iX+1,iY));
    uint4 pixelG = read_imageui( image, sampler, (int2)(iX-1,iY+1));
    uint4 pixelH = read_imageui( image, sampler, (int2)(iX,iY+1));
    uint4 pixelI = read_imageui( image, sampler, (int2)(iX+1,iY+1));
    //Sx and Sy
    uint4 sumX = (pixelA+2*pixelD+pixelG) - (pixelC+2*pixelF+pixelI);
    uint4 sumY = (pixelG+2*pixelH+pixelI) - (pixelA+2*pixelB+pixelC);
    //absolute Sx and Sy and final sum
    uint4 sum = (uint4)(0, 0, 0, 0);
    sum.x = sqrt((float)(sumX.x * sumX.x + sumY.x * sumY.x));
	sum.y = sqrt((float)(sumX.y * sumX.y + sumY.y * sumY.y));
	sum.z = sqrt((float)(sumX.z * sumX.z + sumY.z * sumY.z));
    //bound SOBEL values
    if(sum.x > 255) sum.x = 255;
    if(sum.y > 255) sum.y = 255;
    if(sum.z > 255) sum.z = 255;

    //----------------------------------  THRESHOLD  ----------------------------------
    uint diff = (sum.z-sum.y) + (sum.z-sum.x) + (sum.y-sum.x);
    uint avg_sobel = (sum.x+sum.y+sum.z)/3;
    if(diff > t1 && avg_sobel > t2) { //set to white
      sum = (uint4)(255, 255, 255 , 0);
    }
    else { //set to black
      sum = (uint4)(0, 0, 0 , 0);
    }
    //update pixel
    write_imageui(imageOut, (int2)(iX,iY) , sum);
  }
}





























