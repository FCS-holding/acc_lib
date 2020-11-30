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
#include <sys/time.h>

#include "vgg16.hpp"

////////////////////////////////////////////////////////////////////////////////
data_t absub(data_t a, data_t b) {
    return (a>b)?(a-b):(b-a);
}
data_t abadd(data_t a, data_t b) {
    data_t tmp1 = (a>(data_t)0)?(data_t)a:(data_t)((data_t)-1*a);
    data_t tmp2 = (b>(data_t)0)?(data_t)b:(data_t)((data_t)-1*b);
    data_t tmp3 = (data_t)((data_t)(tmp1+tmp2)/(data_t)2);
    return tmp3;
}
float cmpratio(data_t a, data_t b) {
    float diff = (float)absub(a, b);
    float aver = (float)abadd(a, b);
    float ratio = diff/aver;
    return ratio;
}

template<typename data_t>
void memset_int(data_t *m, data_t val, int addr, int length) {
    for(int i=0; i<length; i++) {
        m[i + addr] = val;
    }
}

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
  int err;                            // error code returned from api calls

  unsigned int correct;               // number of correct results returned

   data_t *DRAM_sw;
   int swDATA_VOL = (WEIGHTSIZE+ INFM+ PL53);
   DRAM_sw = (data_t*) malloc(swDATA_VOL*sizeof(data_t));
   if (DRAM_sw==NULL) {
       printf("malloc failure\n");
       exit (1);
   }
   memset_int<data_t>(DRAM_sw, (data_t)0.001, 0, swDATA_VOL);
   memset_int<data_t>(DRAM_sw, (data_t)0.0004, 0, Layer11);
   memset_int<data_t>(DRAM_sw, (data_t)0.0005, Layer11, Layer12-Layer11);
   memset_int<data_t>(DRAM_sw, (data_t)0.0005, Layer42, Layer43-Layer42);

   printf("0. DRAM_sw init finish\n");

   data_t *DRAM_result;
   int DATA_VOL = (WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);

   DRAM_result = (data_t*) malloc(DATA_VOL*sizeof(data_t));
   if (DRAM_result==NULL) {
       printf("malloc failure\n");
       exit (1);
   }
    memset_int<data_t>(DRAM_result, (data_t)0.001, 0, DATA_VOL);
    memset_int<data_t>(DRAM_result, (data_t)0.0004, 0, Layer11);
    memset_int<data_t>(DRAM_result, (data_t)0.0005, Layer11, Layer12-Layer11);
    memset_int<data_t>(DRAM_result, (data_t)0.0005, Layer42, Layer43-Layer42);

   printf("1. DRAM_result init finish\n");

  size_t global[2];                   // global domain size for our calculation
  size_t local[2];                    // local domain size for our calculation

  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id 
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel
   
  char cl_platform_vendor[1001];
  char cl_platform_name[1001];
   
  cl_mem DRAM;//[DATA_VOL];              // device memory used for the input/output data

  if (argc != 2){
    printf("%s <inputfile>\n", argv[0]);
    return EXIT_FAILURE;
  }

   printf("2. device DRAM init finish\n");

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
 
   printf("3. connect to platform finish\n");
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

  printf("4. create device  finish\n");

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
  size_t n = n_i;
  // Create the compute program from offline
  program = clCreateProgramWithBinary(context, 1, &device_id, &n,
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

  printf("5. build executable finish\n");
  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "vgg16", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  //
  DRAM = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(data_t) * DATA_VOL, NULL, NULL);
  if (!DRAM)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }    
    
  //cl_event writeEvent;

  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(data_t) * DATA_VOL, DRAM_result, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
 
  printf("6. write buffer finish\n");

  //clWaitForEvents(1, &writeEvent);
  //cl_ulong startWrite, endWrite;
  //clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startWrite, NULL);
  //clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endWrite, NULL);
  //cl_ulong kernelWriteTimeNs = endWrite-startWrite;
  //printf("kernal write time :%8.6f ms\n ", kernelWriteTimeNs*1e-6 );

 // Set the arguments to our compute kernel
  //
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &DRAM);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
    //--measure runtime
    //cl_event Eve_kernel;

#ifdef C_KERNEL
  err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  global[0] = 1;
  global[1] = 1;
  local[0] = 1;
  local[1] = 1;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, 
                               (size_t*)&global, (size_t*)&local, 0, NULL, NULL);
#endif

//  clFinish(commands);
//  cl_ulong startTime, endTime;
//  clGetEventProfilingInfo(Eve_kernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
//  clGetEventProfilingInfo(Eve_kernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
//  cl_ulong kernelExecTimeNs = endTime-startTime;
//  printf("kernal exec time :%8.6f ms\n ", kernelExecTimeNs*1e-6 );


  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Read back the results from the device to verify the output
  //
  cl_event readevent;
  err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, 0, sizeof(int) * DATA_VOL, DRAM_result, 0, NULL, &readevent );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  clWaitForEvents(1, &readevent);
  //cl_ulong startRead, endRead;
  //clGetEventProfilingInfo(readevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startRead, NULL);
  //clGetEventProfilingInfo(readevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endRead, NULL);
  //cl_ulong kernelReadTimeNs = endRead-startRead;
  //printf("kernal read time :%8.6f ms\n ", kernelReadTimeNs*1e-6 );

  printf("7. read buffer finish\n");

  // Validate our results
  //
  timeval startSw, endSw;
  gettimeofday(&startSw, NULL);

  vgg16_sw2(DRAM_sw);

  gettimeofday(&endSw, NULL);
  printf("software time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );
  printf("Compute Golden Result Finish\n");

  printf("8. vgg software execute finish\n");

  int cnt=0;
  data_t tmp_1;
  data_t tmp_2;
  for(int i=0; i<PL53; i++) {
    tmp_1 = DRAM_sw[WEIGHTSIZE+ INFM+ i];
    tmp_2 = DRAM_result[WEIGHTSIZE+ INFM+ FM11+FM12+PL12+ FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+ i];
    if (cmpratio(tmp_1, tmp_2)>1E-4) {
      printf("%f, %f, %f\n", (float)tmp_1, (float)tmp_2, (float)cmpratio(tmp_1, tmp_2));
      cnt+=1;
    }
  }

  printf("9. compare finish\n");

  free(DRAM_result);
  free(DRAM_sw);
    
  // Print a brief summary detailing the results
  //
  printf("\n%d diff\n", cnt);
    
  // Shutdown and cleanup
  //
  clReleaseMemObject(DRAM);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  if(cnt == 0){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
}
