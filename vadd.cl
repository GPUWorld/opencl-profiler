/*
 * vadd.cl
 *
 *  Created on: 29/5/2015
 *      Author: diego
 */

__kernel void vadd(
   __global float* a,
   __global float* b,
   __global float* c,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)
       c[i] = a[i] + b[i];
}
