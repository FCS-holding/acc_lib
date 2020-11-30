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

int kernel_cl(data_t *dinput, int bit_flag)
{
  int err;                            // error code returned from api calls
  unsigned int correct;               // number of correct results returned

  int DATA_VOL = (WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);

/* data_t *DRAM_result;
   DRAM_result = (data_t*) malloc(DATA_VOL*sizeof(data_t));
   if (DRAM_result==NULL) {
       printf("malloc failure\n");
       exit (1);
   }
   memcpy(DRAM_result, dinput, sizeof(data_t)*(WEIGHTSIZE+INFM)); */


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
  //printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
  //printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
 
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

if(bit_flag > 0){
  // Load binary from disk
  unsigned char *kernelbinary;
  char binname[200] = "/curr/chenzhang/tool/caffe/FPGA/xclbin/vgg16.xclbin";
  char *xclbin= binname;
  //printf("loading %s\n", xclbin);
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

  // Write our data set into the input array in device memory 
  //
  cl_event event;
  struct timeval t0, t1, t2, t3, t4, t5;

  err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(data_t) * DATA_VOL, dinput, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array a!\n");
    printf("Test failed\n");
    return EXIT_FAILURE;
  }
 
  clWaitForEvents(1, &event);

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device

  //printf("init iteration!\n");
  gettimeofday(&t0, NULL);
  //run the first time
  //err = clEnqueueTask(commands, kernel, 0, NULL, &event);
  //clWaitForEvents(1, &event);
  //gettimeofday(&t1, NULL);

  //run the second time
  int reps=1;
  for(int r=0; r<reps; r++) {
    printf("%d iteration!\n", r+1);
    err = clEnqueueTask(commands, kernel, 0, NULL, &event);
    clWaitForEvents(1, &event);
  }
  gettimeofday(&t2, NULL);
  //float time_pcie = (t1.tv_sec-t0.tv_sec)*1e+3 + (t1.tv_usec-t0.tv_usec)*1e-03 ;
  //float time_kernel = (t2.tv_sec-t1.tv_sec)*1e+3 + (t2.tv_usec-t1.tv_usec)*1e-03 ;
  float time_kernel = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
  time_kernel = time_kernel/reps;
  //time_pcie = time_pcie - time_kernel;
  //printf("PCIe time: %8.6f\t FPGA time :%8.6f ms\n ", time_pcie, time_kernel);
  printf("FPGA time :%8.6f ms\n ", time_kernel);

  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, 0, sizeof(int) * DATA_VOL, dinput, 0, NULL, &event );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    return EXIT_FAILURE;
  }

  clWaitForEvents(1, &event);
  gettimeofday(&t3, NULL);
    
    
  // Shutdown and cleanup
  //
  clReleaseMemObject(DRAM);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

/*  if(cnt == 0){
    printf("Test passed!\n");
    return EXIT_SUCCESS;
  }
  else{
    printf("Test failed\n");
    return EXIT_FAILURE;
  } */
}
