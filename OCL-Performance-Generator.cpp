/*
 ============================================================================
 Name        : OCL-Performance-Generator.cpp
 Author      : Diego Nieto
 Version     :
 Copyright   : GNU License
 Description : Hello World in C++,
 ============================================================================
 */

#include <iostream>
#include <CL/cl.h>

#include "launcher/launcher.hpp"
#include "launcher/execution.hpp"

using namespace std;

int main(void) {
	Launcher launcher(2,64,1,2);

	launcher.initKernel("vadd.cl","vadd");

	cl_mem d_a, d_b, d_c;
	int err;

	int size = 4;
	unsigned int totalBytes = (sizeof(float)*4);

  d_a = clCreateBuffer(launcher.getContext(), CL_MEM_READ_ONLY, totalBytes, NULL, &err);
  Launcher::clCheckError(err, "BUF1");
  d_b = clCreateBuffer(launcher.getContext(), CL_MEM_READ_ONLY, totalBytes, NULL, &err);
  Launcher::clCheckError(err, "BUF2");
  d_c = clCreateBuffer(launcher.getContext(), CL_MEM_READ_WRITE, totalBytes, NULL, &err);
  Launcher::clCheckError(err, "BUF3");

  err = clSetKernelArg(launcher.getKernel(), 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(launcher.getKernel(), 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(launcher.getKernel(), 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(launcher.getKernel(), 3, sizeof(cl_int), &size);

  size_t globalWorkSize[1];
  globalWorkSize[0] = 1024;

  launcher.launch(globalWorkSize);

  Execution *best = launcher.getBest();

	cout << "Number of executions: " << launcher.getNumberOfExecutions() << endl; /* prints Hello World */
	cout << "Best time: " << best->getTime() << ". x=" << best->getX() << ", y=" << best->getY() << endl;
	return 0;
}
