/*
 ============================================================================
 Name        : OCL-Performance-Generator.cpp
 Author      : Diego Nieto
 Version     :
 Copyright   : GNU License
 ============================================================================
 */

#include "launcher/launcher.hpp"
#include "launcher/execution.hpp"

using namespace std;

int main(void) {
  Launcher launcher(2, 64, 1, 2);

  launcher.initKernel("vadd.cl", "vadd");

  int size = 4;
  unsigned int totalBytes = (sizeof(float) * size);

  size_t globalWorkSize[1];
  globalWorkSize[0] = 1024;

  launcher.setArgs(OclArg(totalBytes, BUFFER, CL_MEM_READ_ONLY),
      OclArg(totalBytes, BUFFER, CL_MEM_READ_ONLY),
      OclArg(totalBytes, BUFFER, CL_MEM_READ_WRITE),
      OclArg(sizeof(cl_int), OTHER, NULL, (void**) &size));

  launcher.launch(globalWorkSize);

  launcher.printStats();
  return 0;
}
