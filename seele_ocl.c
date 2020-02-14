#include "sph_keccak.h"
#include <CL/opencl.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define HASH_ROWS 429

#define MATRIX_DIM 30
#define HASH_TRIES (HASH_ROWS - MATRIX_DIM + 1)
#define HASH_COL_NUM 256
#define DET_LEN 225

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  FILE *knl_fp;
  knl_fp = fopen("mpow.cl", "rb");
  fseek(knl_fp, 0, SEEK_END);
  long fsize = ftell(knl_fp);
  fseek(knl_fp, 0, SEEK_SET); /* same as rewind(f); */

  char *knl_src = malloc(fsize + 1);
  fread(knl_src, 1, fsize, knl_fp);
  fclose(knl_fp);
  knl_src[fsize] = 0;

  int err; // error code returned from api calls

  uint64_t results[8];   // results returned from device
  float float_mats[256]; // results returned from device
  unsigned int correct;  // number of correct results returned

  char *header =
      "\xf9\x01\x00\xa0\x93\x54\x69\x17\x41\x5f\xa4\xac\x1d\x4b\xb0\xda\x26\x95"
      "\xd6\x3e\x46\xaa\xd4\x94\x94\x18\x34\x43\x64\x3e\x0c\x67\x02\x8b\xd0\x47"
      "\x94\xba\xd5\x58\xfc\x41\xfa\xb4\x29\xac\xf3\x9c\xf8\xae\x1c\x6e\xe9\x5d"
      "\x52\xb0\xa1\xa0\x9d\x81\x21\x1c\xe9\x41\x2c\x59\x6b\xdb\x32\x0f\xa8\xbb"
      "\x68\x5b\xa9\x08\x33\x10\x11\xff\x7b\x11\x61\x69\x80\xd3\xb0\xfa\xa6\x0e"
      "\xa0\xf1\x88\x78\x00\xe0\x6a\x2d\xbf\x41\xc6\x5a\xab\x7a\x92\x6e\x57\xd1"
      "\x69\x8d\x9e\xef\xa9\xaf\x57\xeb\xfa\xa0\x56\x22\xb6\xe9\xa0\xa0\x17\xaa"
      "\xdb\x58\x87\xfe\xe6\xc7\xa4\x72\x1c\xbb\xaf\xd5\xf4\xf6\x09\xc8\x56\x70"
      "\xb8\xc2\x1f\xee\xbf\xf4\xf2\x9e\xb3\x2f\x05\x94\xa0\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x84\x03\x80\x36\xfe\x83\x14\x39\x3e\x84\x5d\xd1"
      "\x5f\x75\x93\x31\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30"
      "\x30\x30\x30\x30\x80\x80\x80";

  char *mutableheader;
  mutableheader = malloc(259);
  memcpy(mutableheader, header, 259);

  size_t global; // global domain size for our calculation
  size_t local;  // local domain size for our calculation

  cl_platform_id cpPlatform[5];
  cl_context context;        // compute context
  cl_command_queue commands; // compute command queue
  cl_program program;        // compute program
  cl_kernel kernels[4];      // compute kernels[0]

  cl_mem in_header; // device memory used for the input array
  cl_mem keccak_res;
  cl_mem expanded_floats; // device memory used for the output array
  cl_mem det_buffer;
  cl_mem chance_buffer;
  // cl_mem checkdet_res_buffer;
  cl_mem checkdet_det_buffer;

  // Fill our data set with random float values
  //
  int i = 0;

  //cl_uint num_platforms = 0;
  //cl_platform_id platform_id = 0;
  //cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
  cl_device_id device_id = 0;
  cl_uint num_devices = 0;
  //ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
  // 0, 0, &num_devices);
  //cl_uint num_default_device = 0;
  //ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
  // 1, &device_id, &num_default_device);

  // Connect to a compute device
  cl_uint numPlatforms = 0;
  //
  err = clGetPlatformIDs(5, cpPlatform, &numPlatforms);
  err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to create a device group! err: %d\n", err);
    return EXIT_FAILURE;
  }

  // Create a compute context
  //
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if (!context)
  {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }

  // Create a command commands
  //
  commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
  if (!commands)
  {
    printf("Error: Failed to create a command commands!\n");
    return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  //
  program = clCreateProgramWithSource(context, 1, (const char **)&knl_src, NULL,
                                      &err);
  if (!program)
  {
    printf("Error: Failed to create compute program!, err: %d\n", err);
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
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }

  kernels[0] = clCreateKernel(program, "genkeccakmat", &err);
  if (!kernels[0] || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernels[0]!\n");
    exit(1);
  }
  kernels[1] = clCreateKernel(program, "qr", &err);
  if (!kernels[1] || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernels[1]!\n");
    exit(1);
  }
  kernels[2] = clCreateKernel(program, "check_chance", &err);
  if (!kernels[2] || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernels[2]!\n");
    exit(1);
  }
  kernels[3] = clCreateKernel(program, "check_dets", &err);
  if (!kernels[2] || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernels[3]!\n");
    exit(1);
  }

  // Create the input and output arrays in device memory for our calculation
  //
  in_header = clCreateBuffer(context, CL_MEM_READ_ONLY, 240, NULL, NULL);
  keccak_res = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * HASH_ROWS, NULL, NULL);
  expanded_floats =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     32 * HASH_ROWS * 8 * sizeof(float), NULL, NULL);
  det_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(float) * DET_LEN * HASH_TRIES, NULL, NULL);
  chance_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(int) * HASH_TRIES, NULL, NULL);
  // checkdet_res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
  //                                      sizeof(int) * HASH_TRIES, NULL, NULL);
  checkdet_det_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       sizeof(float) * HASH_TRIES, NULL, NULL);

  if (!in_header || !keccak_res || !expanded_floats || !det_buffer || !chance_buffer ||
      !checkdet_det_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  err = 0;
  cl_ulong nonce_start = 999;
  err = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &in_header);
  err |= clSetKernelArg(kernels[0], 1, sizeof(nonce_start), &nonce_start);
  err |= clSetKernelArg(kernels[0], 2, sizeof(cl_mem), &expanded_floats);
  err |= clSetKernelArg(kernels[0], 3, sizeof(cl_mem), &keccak_res);

  if (err < 0)
  {
    printf("Couldn't set a kernel[0] argument, err: %d", err);
    exit(1);
  };
  // err = clSetKernelArg(kernels[1], 0, MATRIX_DIM * sizeof(float), NULL);
  err = clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &expanded_floats);
  err |= clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &det_buffer);
  err |= clSetKernelArg(kernels[1], 2, sizeof(cl_mem), &chance_buffer);
  if (err < 0)
  {
    printf("Couldn't set a kernel[1] argument, err: %d", err);
    exit(1);
  };
  float target = 800000000.0;
  err = clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &det_buffer);
  err |= clSetKernelArg(kernels[2], 1, sizeof(target), &target);
  err |= clSetKernelArg(kernels[2], 2, sizeof(cl_mem), &chance_buffer);
  if (err < 0) {
      printf("Couldn't set a kernel[2] argument, err: %d", err);
      exit(1);
  };

  err = clSetKernelArg(kernels[3], 0, sizeof(cl_mem), &det_buffer);
  err |= clSetKernelArg(kernels[3], 1, sizeof(target), &target);
  // err |= clSetKernelArg(kernels[3], 2, sizeof(cl_mem), &checkdet_res_buffer);
  err |= clSetKernelArg(kernels[3], 2, sizeof(cl_mem), &checkdet_det_buffer);
  if (err < 0)
  {
    printf("Couldn't set a kernel[3] argument, err: %d", err);
    exit(1);
  };

  size_t tries = 0;

  clock_t begin = clock();

  //for(int q=0; q<2; q++)
  for (int kk = 0; kk < 1000000000; kk++)
  {
    // printf("header: ");
    // for (int i = 0; i < 259; i++)
    // {
    //   printf("%02x", mutableheader[i] & 0xff);
    // }
    // printf("\n");

    ////////////////////////////////Expand Header into Float Matrix Begin////////////////////////
    // mutableheader[238] = (char)(0x30 + (kk / 10) % 10);
    // mutableheader[239] = (char)(0x30 + kk % 10);
    nonce_start = kk*HASH_TRIES;
    err = clSetKernelArg(kernels[0], 1, sizeof(nonce_start), &nonce_start);
    // err = clEnqueueWriteBuffer(commands, in_header, CL_TRUE, 0, 240,
    //                            mutableheader, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write to source array!\n");
      exit(1);
    }

    global = HASH_ROWS;
    local = global;
    err = clEnqueueNDRangeKernel(commands, kernels[0], 1, NULL, &global, &local,
                                 0, NULL, NULL);
    if (err)
    {
      printf("Error: Failed to execute kernels[0]! err: %d\n", err);
      return EXIT_FAILURE;
    }
    // tries += HASH_TRIES;

    clFinish(commands);
    ////////////////////////////////Expand Header into Float Matrix End////////////////////////


    ////////////////////////////////Calc Dets in speculer mode [last 20 dets] Begin////////////////////////
    
    
    size_t qr_global_size[3];
    size_t qr_global_offset[3];
    size_t qr_local_size[3] = {MATRIX_DIM, 1, 1};

    qr_global_size[0] = MATRIX_DIM;
    qr_global_size[1] = 20;
    qr_global_size[2] = HASH_TRIES;
    qr_global_offset[0] = 0;
    qr_global_offset[1] = 205;
    qr_global_offset[2] = 0;
    
    err = clEnqueueNDRangeKernel(commands, kernels[1], 3, qr_global_offset, qr_global_size,
                                 qr_local_size, 0, NULL, NULL);
    if (err)
    {
      printf("Error: Failed to calc dets in speculer mode! err: %d\n", err);
      return EXIT_FAILURE;
    }
    float det_res[DET_LEN * HASH_TRIES];
    // err = clEnqueueReadBuffer(commands, det_buffer, CL_TRUE, 0,
    //                           sizeof(det_res), det_res, 0, NULL, NULL);

    // for (int hashtries = 0; hashtries < HASH_TRIES; hashtries++)
    // {
    //   printf("det: ");
    //   for (int q = 0; q < DET_LEN; q++)
    //   {
    //     printf("%.2f, ", det_res[hashtries * DET_LEN + q]);
    //   }
    //   printf("\n\n");
    // }

    // cl_uint chance_buff[HASH_TRIES];
    // err = clEnqueueReadBuffer(commands, chance_buffer, CL_TRUE, 0,
    //                           sizeof(chance_buff), chance_buff, 0, NULL, NULL);
    // if(err)
    // {
    //   printf("Error: Failed to readback chance buff, %d\n",err);
    //   return EXIT_FAILURE;
    // }
    // for(int i = 0; i < HASH_TRIES; i++){
    //   printf("chance buff readback: %d\n", chance_buff[i]);
    // }
    // printf("\n");
    ////////////////////////////////Calc Dets in speculer mode [last 20 dets] End////////////////////////


    ////////////////////////////////Check speculer chance Begin////////////////////////
    size_t global_checkdet = HASH_TRIES;
    target=8000000000;
    clSetKernelArg(kernels[2], 1, sizeof(target), &target);
    clSetKernelArg(kernels[3], 1, sizeof(target), &target);
    err = clEnqueueNDRangeKernel(commands, kernels[2], 1, NULL,
                                 &global_checkdet, NULL, 0, NULL, NULL);
    if (err < 0)
    {
      perror("Couldn't enqueue the kernel");
      exit(1);
    }

    cl_uint chance[HASH_TRIES];

    err = 0;
    err = clEnqueueReadBuffer(commands, chance_buffer, CL_TRUE, 0,
                              sizeof(chance), chance, 0, NULL, NULL);
    if (err < 0)
    {
      perror("Couldn't read the buffers");
      exit(1);
    }
    // for(int i=0; i< HASH_TRIES; i++){
    //   printf("chance: %d\n",chance[i]);
    // }

    clFinish(commands);
    ////////////////////////////////Check speculer chance End////////////////////////


    ////////////////////////////////Calc Dets in normal mode [first 205 dets] Begin////////////////////////
    qr_global_size[0] = MATRIX_DIM;
    qr_global_size[1] = 205;
    qr_global_size[2] = HASH_TRIES;
    qr_global_offset[0] = 0;
    qr_global_offset[1] = 0;
    qr_global_offset[2] = 0;
    err = clEnqueueNDRangeKernel(commands, kernels[1], 3, qr_global_offset, qr_global_size,
                                 qr_local_size, 0, NULL, NULL);
    if (err)
    {
      printf("Error: Failed to calc dets! err: %d\n", err);
      return EXIT_FAILURE;
    }
    // err = clEnqueueReadBuffer(commands, det_buffer, CL_TRUE, 0,
    //                           sizeof(det_res), det_res, 0, NULL, NULL);

    // for (int hashtries = 0; hashtries < HASH_TRIES; hashtries++)
    // {
    //   printf("det: ");
    //   for (int q = 0; q < DET_LEN; q++)
    //   {
    //     printf("%.2f, ", det_res[hashtries * DET_LEN + q]);
    //   }
    //   printf("\n\n");
    // }
    ////////////////////////////////Calc Dets in normal mode [first 205 dets] End////////////////////////

    ////////////////////////////////Check Nonce Begin////////////////////////
    global_checkdet = HASH_TRIES;
    // cl_uint nonce[HASH_TRIES];
    float dets[HASH_TRIES] = {0.0};
    err = clEnqueueNDRangeKernel(commands, kernels[3], 1, NULL,
                                 &global_checkdet, NULL, 0, NULL, NULL);
    if (err < 0)
    {
      perror("Couldn't enqueue the kernel");
      exit(1);
    }
    clFinish(commands);

    err = 0;
    // err = clEnqueueReadBuffer(commands, checkdet_res_buffer, CL_TRUE, 0,
    //                           sizeof(nonce), nonce, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commands, checkdet_det_buffer, CL_TRUE, 0,
                               sizeof(dets), dets, 0, NULL, NULL);

    if (err < 0)
    {
      perror("Couldn't read the buffers");
      exit(1);
    }
    for (int i = 0; i < HASH_TRIES; i++)
    {
      if (dets[i] < target)
      {
          continue;
      }
      // printf("offset: %d, res: %d, det: %.2f\n", i + HASH_TRIES * kk, nonce[i], dets[i]);
      printf("offset: %d, det: %.2f\n", i + HASH_TRIES * kk, dets[i]);
    }
    ////////////////////////////////Check Nonce End////////////////////////
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("clock: %lf\n", time_spent);
  // Shutdown and cleanup
  //
  clReleaseMemObject(in_header);
  clReleaseMemObject(keccak_res);
  clReleaseMemObject(expanded_floats);
  clReleaseMemObject(det_buffer);
  clReleaseMemObject(chance_buffer);
  // clReleaseMemObject(checkdet_res_buffer);
  clReleaseMemObject(checkdet_det_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernels[0]);
  clReleaseKernel(kernels[1]);
  clReleaseKernel(kernels[2]);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  getchar();
  return 0;
}
