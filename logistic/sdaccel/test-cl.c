/*******************************************************************************
Vendor: Xilinx 
Associated Filename: test-cl.c
Purpose: OpenCL Host Code for Matrix Multiply Example
Revision History: July 1, 2013 - initial release

 *******************************************************************************
 Copyright (C) 2013 XILINX, Inc.

 This file contains confidential and proprietary information of Xilinx, Inc. and 
 is protected under U.S. and international copyright and other intellectual 
 property laws.

 DISCLAIMER
 This disclaimer is not a license and does not grant any rights to the materials 
 distributed herewith. Except as otherwise provided in a valid license issued to 
 you by Xilinx, and to the maximum extent permitted by applicable law: 
 (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX 
 HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
 INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
 FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether 
 in contract or tort, including negligence, or under any other theory of 
 liability) for any loss or damage of any kind or nature related to, arising under 
 or in connection with these materials, including for any direct, or any indirect, 
 special, incidental, or consequential loss or damage (including loss of data, 
 profits, goodwill, or any type of loss or damage suffered as a result of any 
 action brought by a third party) even if such damage or loss was reasonably 
 foreseeable or Xilinx had been advised of the possibility of the same.

 CRITICAL APPLICATIONS
 Xilinx products are not designed or intended to be fail-safe, or for use in any 
 application requiring fail-safe performance, such as life-support or safety 
 devices or systems, Class III medical devices, nuclear facilities, applications 
 related to the deployment of airbags, or any other applications that could lead 
 to death, personal injury, or severe property or environmental damage 
 (individually and collectively, "Critical Applications"). Customer assumes the 
 sole risk and liability of any use of Xilinx products in Critical Applications, 
 subject only to applicable laws and regulations governing limitations on product 
 liability. 

 THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT 
 ALL TIMES.

 *******************************************************************************/
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static matrix for simplicity
//
#define LABEL_SIZE		10
#define FEATURE_SIZE	784
#define DUP 10


////////////////////////////////////////////////////////////////////////////////

    int
load_file_to_memory(const char *filename, char **result)
{ 
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) 
    { 
        *result = NULL;
        return -1; // -1 means file opening fail 
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) 
    { 
        free(*result);
        return -2; // -2 means file reading fail 
    } 
    fclose(f);
    (*result)[size] = 0;
    return size;
}

int main(int argc, char** argv)
{
    float learning_rate = 0.13;
    int L = LABEL_SIZE;
    int D = FEATURE_SIZE;
    int n = 60000;
    int m = 10000;
    int iter, i, j, k;

    float* global_data = (float*)malloc(n*(D+L)*sizeof(float));
    float* test_data = (float*)malloc(m*(D+L)*sizeof(float));
    FILE* pFile = fopen("/curr/bjxiao/projects/logistic/data/train_data.bin", "r");
    if (!pFile) {
        printf("cannot find training file\n");
    }
    fread(global_data,sizeof(float),n*(D+L),pFile);
    fclose(pFile);
    pFile = fopen("/curr/bjxiao/projects/logistic/data/test_data.bin", "r");
    if (!pFile) {
        printf("cannot find test file\n");
    }
    fread(test_data,sizeof(float),m*(D+L),pFile);
    fclose(pFile);
    printf("finish reading train data.\n");

    float* global_weights		= (float*)malloc(L*(D+1)*sizeof(float));
    memset(global_weights,0, L*(D+1)*sizeof(float));
    float* global_gradient 	= (float*)malloc(L*(D+1)*sizeof(float));

    int err;                            // error code returned from api calls

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem input_weights;                     // device memory used for the input array
    cl_mem input_data;                     // device memory used for the input array
    cl_mem output_gradient;                      // device memory used for the output array

    if (argc != 2){
        printf("%s <inputfile>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Connect to first platform
    //
    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_NAME %s\n",cl_platform_name);

    // Connect to a compute device
    //
    int fpga = 0;
#if defined (FPGA_DEVICE)
    fpga = 1;
#endif
    err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
            1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    int status;

    // Create Program Objects
    //

    // Load binary from disk
    unsigned char *kernelbinary;
    char *xclbin=argv[1];
    printf("loading %s\n", xclbin);
    int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    size_t n_t = n_i;
    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device_id, &n_t,
            (const unsigned char **) &kernelbinary, &status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "mmult", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input_weights = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * L*(D+1), NULL, NULL);
    input_data = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * n*(D+L), NULL, NULL);
    output_gradient = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * L*(D+1), NULL, NULL);
    if (!input_weights || !input_data || !output_gradient)
    {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }    

    for (iter=0; iter<100; iter++) {
        printf("iteration %d starting...\n", iter);
        // Write our data set into the input array in device memory 
        //
        err = clEnqueueWriteBuffer(commands, input_weights, CL_TRUE, 0, sizeof(int) * L*(D+1), global_weights, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array a!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        // Write our data set into the input array in device memory 
        //
        err = clEnqueueWriteBuffer(commands, input_data, CL_TRUE, 0, sizeof(int) * n*(D+L), global_data, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array b!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        // Set the arguments to our compute kernel
        //
        err = 0;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &n);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_weights);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_data);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_gradient);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //

        err = clEnqueueTask(commands, kernel, 0, NULL, NULL);

        if (err)
        {
            printf("Error: Failed to execute kernel! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        // Read back the results from the device to verify the output
        //
        cl_event readevent;
        err = clEnqueueReadBuffer( commands, output_gradient, CL_TRUE, 0, sizeof(int) * L*(D+1), global_gradient, 0, NULL, &readevent );  
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        clWaitForEvents(1, &readevent);

        for (i = 0; i < L; i++) {
            for (j = 0; j < D+1; j++) {
                global_weights[i*(D+1)+j] -= learning_rate * global_gradient[i*(D+1)+j] / n;
                //printf("gradient(%d,%d)=%f\n",i,j,global_gradient[i*(D+1)+j]);
            }
        }
        int errors = 0;
        for(i = 0; i < m; i++)
        {
            float max_possibility = -10000;
            int likely_class = 0;
            for(j = 0; j < L; j++)
            {
                float dot = global_weights[j*(D+1)];
                for(k=0; k < D; k++)
                {
                    dot += global_weights[j*(D+1)+k+1] * test_data[i*(D+L)+k+L];
                }
                if(dot > max_possibility)
                {
                    max_possibility = dot;
                    likely_class = j;
                }
            }
            if( test_data[i*(D+L)+likely_class] < 0.5 )
            {
                errors++;
            }
        }
        printf("error rate: %f\%\n", ((float)errors)/ m * 100.);
    }

    // Shutdown and cleanup
    //
    clReleaseMemObject(input_weights);
    clReleaseMemObject(input_data);
    clReleaseMemObject(output_gradient);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(test_data);
    free(global_data);
    free(global_weights);
    free(global_gradient);

    return EXIT_SUCCESS;
}
