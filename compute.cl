#pragma OPENCL EXTENSION cl_arm_printf : enable

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

 __kernel
void computeTA (const int wA,
                          __global TYPE* A,
                          __global TYPE* B,
                          __global TYPE* C)
{
	 const unsigned int col = get_global_id(0);
	 const unsigned int row = get_global_id(1);

	 unsigned int k;

	 TYPE partialC = 0.f;
	 TYPE2 a1;
	 TYPE2 b1;
	 TYPE2 c1 = 0;

	 for ( k=0; k<wA; k+=2 )
	 {
		 a1 = vload2(0,&A[col*wA+k]);
		 b1 = vload2(0,&B[row*wA+k]);
		 c1 += a1*b1;
	 }

	 C[row*wA+col] = c1.s0 + c1.s1;
}

__kernel 
void computeTB (const int wA,
                          __global TYPE* A,
                          __global TYPE* B,
                          __global TYPE* C)
{
	 const unsigned int col = get_global_id(0);
	 const unsigned int row = get_global_id(1);

	 unsigned int k;

	 TYPE partialC = 0.f;
	 TYPE2 a1;
	 TYPE2 b1;
	 TYPE2 c1 = 0;

	 for ( k=0; k<wA; k+=2 )
	 {
		 a1 = vload2(0,&A[k*wA+col]);
		 b1 = vload2(0,&B[k*wA+row]);
		 c1 += a1*b1;
	 }

	 C[row*wA+col] = c1.s0 + c1.s1;
}

__kernel
void compute_orig (const int wA,
                            __global TYPE* A,
                            __global TYPE* B,
                            __global TYPE* C)
{
    int row = get_global_id(1);

    int col = get_global_id(0);

    TYPE cSum = 0.0f;

    for (int i=0; i<wA; i++)
    {
        cSum += B[row*wA+ i] * A[i*wA+col];
    }

    C[row*wA+col] = cSum+C[row*wA+col];
}

__kernel
void trans_kernel (const int lda,
                   __global TYPE* in,
		   __global TYPE* out)
{
	const unsigned int i = get_global_id(0);
	const unsigned int j = get_global_id(1);
	out[j*lda+i] = in[i*lda+j];
}


__kernel
void dgemmcl(int M, 
             int N, 
             int K, 
             TYPE alpha, 
             __global TYPE4* A, 
             int lda, 
             __global TYPE4* B, 
             int ldb, 
             TYPE beta, 
             __global TYPE4* C, 
             int ldc)
{
    const unsigned int row = get_global_id(0); // Current row
    const unsigned int col = get_global_id(1); // Current column

    TYPE4 a1, a2, a3, a4, b, c, c1;

    const unsigned int col4 = col*4;

    for ( unsigned int k=0; k<K; k+=4 )
    {
        // Load A 4x4 double block
        a1 = A[(((0+k)*lda)+(col4))/4];
        a2 = A[(((1+k)*lda)+(col4))/4];
        a3 = A[(((2+k)*lda)+(col4))/4];
        a4 = A[(((3+k)*lda)+(col4))/4];

        // This computes row by row of our block
        for ( unsigned int bRow=0; bRow<4; bRow++ )
        {
            // Load B row, we reuse A rows
            b = B[((row+bRow)*ldb+k)/4];
            
            // Reset c
            c = 0.f;

            c.s0 += a1.s0*b.s0+
                    a2.s0*b.s1+
                    a3.s0*b.s2+
                    a4.s0*b.s3;

            c.s1 += a1.s1*b.s0+
                    a2.s1*b.s1+
                    a3.s1*b.s2+
                    a4.s1*b.s3;

            c.s2 += a1.s2*b.s0+
                    a2.s2*b.s1+
                    a3.s2*b.s2+
                    a4.s2*b.s3;

            c.s3 += a1.s3*b.s0+
                    a2.s3*b.s1+
                    a3.s3*b.s2+
                    a4.s3*b.s3;

            C[(((row*4)+bRow)*ldc+(col4))/4] += c;
        }
    }
}
