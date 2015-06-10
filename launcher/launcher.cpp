/*
 * launcher.cpp
 *
 *  Created on: 27/5/2015
 *      Author: diego
 */

#include "launcher.hpp"
#include <iostream>

void Launcher::launch(const size_t *globalWork)
{
	if ( _isInit )  {
		const unsigned int xMax = _max;
		const unsigned int yMax = _max;
		const unsigned int zMax = _max;

		size_t *localWorkSize;
		localWorkSize = new size_t[_dim];

		switch ( _dim )
		{
		// One dimension
			case 1:
				_global_work[0] = globalWork[0];
				for ( unsigned int x=_min; x<=xMax; x*=_step )
				{
					localWorkSize[0] = x;
					update(runKernel(localWorkSize));
				}
				break;
		// Two dimensions
			case 2:
				_global_work[0] = globalWork[0];
				_global_work[1] = globalWork[1];
				for ( unsigned int x=_min; x<=xMax; x*=_step )
				{
					for ( unsigned int y=_min; y<=yMax; y*=_step )
					{
                        std::cout << "x=" << x << "y=" << y << std::endl;
                        localWorkSize[0] = x;
                        localWorkSize[1] = y;
                        update(runKernel(localWorkSize));
					}
				}
				break;
		// Three dimensions
			case 3:
				_global_work[0] = globalWork[0];
				_global_work[1] = globalWork[1];
				_global_work[2] = globalWork[2];
				for ( unsigned int x=_min; x<=xMax; x*=_step )
				{
					for ( unsigned int y=_min; y<=yMax; y*=_step )
					{
						for ( unsigned int z=_min; z<=zMax; z*=_step )
						{

						}
					}
				}
				break;
			default:
				throw;
		}

		delete localWorkSize;
	} else {
		throw;
	}
}

void Launcher::update(Execution &execution)
{
	if ( execution < (*_best) )
	{
		_best = &execution;
	}
	_executions.push(execution);
}

void Launcher::init()
{
	initPlatforms();
	initDevices();
	initContext();
	initCommandQueue();

	_isInit = true;
}

void Launcher::initKernel(const char *filename, const char *kernelName)
{
	cl_int err;
  char* kernelSource;
  FILE *kernelFile;
  kernelFile = fopen(filename, "r");
  if ( kernelFile == NULL )
  	throw;
  fseek(kernelFile, 0, SEEK_END);
  size_t size = ftell(kernelFile);
  fseek(kernelFile, 0, SEEK_SET);

  kernelSource = new char[size+1];
  fread(kernelSource, 1, size, kernelFile);
  fclose(kernelFile);

	_program = clCreateProgramWithSource(_context, 1, (const char **) &kernelSource, NULL, &err);
	clCheckError(err, "CREATE_PROGRAM 1\n");
  err  =  clBuildProgram(_program, 0, NULL, "-DTYPE=double -DTYPE2=double2 -DTYPE4=double4", NULL, NULL);
    //err  =  clBuildProgram(program, 0, NULL, "-DTYPE=double-g -s \\home\diego\workspace_cpp\OCL-gemm\compute.cl'", NULL, NULL);
	//err  =  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
			char buildLog[16384];
			clGetProgramBuildInfo(_program, _currentDevice, CL_PROGRAM_BUILD_LOG,
					sizeof(buildLog), buildLog, NULL);
			printf("%s\b", buildLog);
			clReleaseProgram(_program);
			throw;
	}

	_kernel = clCreateKernel(_program, kernelName, &err);
	clCheckError(err, "Kernel");
}

Execution& Launcher::runKernel(size_t *localWorkSize)
{
	cl_uint clErr;
	cl_event ev;
	cl_ulong startTime, endTime;

	clCheckError(clEnqueueNDRangeKernel (_commandQueue, _kernel, _dim, 0, _global_work, localWorkSize, 0, NULL, &ev),
			"Error on kernel launch");

	clWaitForEvents(1, &ev);

  clCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL),"Error reading start execution");
  clCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL),"Error reading end execution");

  unsigned int x, y, z;
  x = localWorkSize[0];
  if ( _dim > 1)
  	y = localWorkSize[1];
  else if ( _dim > 2)
  	z =localWorkSize[2];

  new Execution(_dim,x,y,z,(cl_ulong)endTime-startTime);
}

void Launcher::initDevices()
{
	if ( _platforms.size() != 0 ) {
		unsigned int numDevices;

		for ( unsigned int j=0; j<_platforms.size(); j++ )
		{
			clCheckError(clGetDeviceIDs(_platforms[j], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices), "DEVICE CHECK 1 ERROR\n");

			cl_device_id *clDevices = new cl_device_id[numDevices];

			clCheckError(clGetDeviceIDs(_platforms[j], CL_DEVICE_TYPE_ALL, numDevices, clDevices, NULL), "DEVICE CHECK 2 ERROR\n");

			for ( unsigned int i=0; i<numDevices; i++ )
			{
				_devices.push_back(clDevices[i]);
			}

			delete[] clDevices;
		}
		_currentDevice = _devices[0];
	} else {
		throw;
	}
}

Launcher::~Launcher()
{
	delete[] _global_work;
}

void Launcher::initContext()
{
	cl_int err;
	if ( _currentDevice != NULL ) {
		_context = clCreateContext(0, 1, &_currentDevice, NULL, NULL, &err);
		clCheckError(err, "CONTEXT\n");
	} else {
		throw;
	}
}

void Launcher::initPlatforms()
{
	unsigned int numPlatforms;
	clCheckError(clGetPlatformIDs(NULL, NULL, &numPlatforms), "PLATFORM CHECK 1 ERROR\n");

	cl_platform_id *clPlatforms = new cl_platform_id[numPlatforms];

	clCheckError(clGetPlatformIDs(numPlatforms, clPlatforms, NULL), "PLATFORM CHECK 2 ERROR\n");

	for ( unsigned int i=0; i<numPlatforms; i++ )
	{
		_platforms.push_back(clPlatforms[i]);
	}

	delete[] clPlatforms;
}

void Launcher::initCommandQueue()
{
	cl_int err;
	if ( _context != NULL &&& _currentDevice != NULL ) {
		_commandQueue = clCreateCommandQueue(_context, _currentDevice, CL_QUEUE_PROFILING_ENABLE, &err);
		clCheckError(err, "COMMAND_QUEUE\n");
	} else {
		throw;
	}
}

template <typename... OpenCLArgument>
void Launcher::setArgs(OpenCLArgument... args)
{
	// TODO: use this way
}
