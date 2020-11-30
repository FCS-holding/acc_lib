#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>

#include <CL/opencl.h>

class OpenCLEnv {

public:

  OpenCLEnv(
      const char* bin_path,
      const char* kernel_name):
    num_platforms(0),
    num_devices(0)
  {
    // start platform setting up
    int err;

    cl_platform_id platform_id[32];
    cl_device_id device_id[32];

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    // Connect to first platform
    err = clGetPlatformIDs(8, platform_id, &num_platforms);

    if (err != CL_SUCCESS || num_platforms == 0) {
      throw std::runtime_error(
          "Failed to find an OpenCL platform");
    }

    printf("Found %d OpenCL Platform\n", num_platforms);

    for (int p=0; p<num_platforms; p++) {
      err = clGetPlatformInfo(
          platform_id[p], 
          CL_PLATFORM_VENDOR, 
          1000, 
          (void *)cl_platform_vendor,NULL);

      if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!");
      }

      err = clGetPlatformInfo(platform_id[p],CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);

      if (err != CL_SUCCESS) {
        throw std::runtime_error("clGetPlatformInfo(CL_PLATFORM_NAME) failed!");
      }

      printf("OpenCL Vendor: %s\n", cl_platform_vendor);
      printf("OpenCL Platform: %s\n", cl_platform_name);
    }

    // Connect to a compute device
    // Assuming platform 0 is GPU
    err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 8, device_id, &num_devices);

    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create a device group!");
    }

    printf("Found %d devices\n", num_devices);

    // Create a compute context for all devices on this platform
    context = clCreateContext(0, num_devices, device_id, NULL, NULL, &err);

    if (!context) {
      throw std::runtime_error("Failed to create a compute context!");
    }

    // Create command queues
    commands = new cl_command_queue[num_devices];

    for (int d=0; d<num_devices; d++) {
      commands[d] = clCreateCommandQueue(context, device_id[d], 0, &err);
      if (!commands[d]) {
        throw std::runtime_error("Failed to create a command queue");
      }
    }

    // Create Program Objects
    // Load source from disk
    char *kernelSource;

    int n_i = load_file(bin_path, &kernelSource);

    if (n_i < 0) {
      throw std::runtime_error(
          "failed to load kernel from xclbin");
    }
    size_t n_t = n_i;

    int status = 0;

    // Create the compute program from offline
    program = clCreateProgramWithSource(context, 1,
        (const char **) &kernelSource, &n_t, &err);

    if ((!program) || (err!=CL_SUCCESS)) {
      throw std::runtime_error(
          "Failed to create compute program from source");
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
      if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
      }
      throw std::runtime_error("Failed to build program executable!");
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);

    if (!kernel || err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create compute kernel!");
    }
  }

  ~OpenCLEnv() {

    clReleaseProgram(program);
    clReleaseKernel(kernel);

    for (int d=0; d<num_devices; d++) {
      clReleaseCommandQueue(commands[d]);
    }
    delete [] commands;

    clReleaseContext(context);
  }

  cl_program& getProgram() { return program; }
  cl_context& getContext() { return context; }

  cl_command_queue& getCmdQueue(int idx) { return commands[idx]; }

  cl_kernel& getKernel() { return kernel; }

  unsigned int getNumDevices() { return num_devices; }

private:

  int load_file(
      const char *filename, 
      char **result)
  { 
    int size = 0;
    FILE *f = fopen(filename, "r");
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

  unsigned int num_platforms;
  unsigned int num_devices;

  cl_context        context;
  cl_command_queue* commands;
  cl_program        program;
  cl_kernel         kernel;
};
