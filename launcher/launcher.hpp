/*
 * launcher.hpp
 *
 *  Created on: 27/5/2015
 *      Author: diego
 */

#ifndef LAUNCHER_HPP_
#define LAUNCHER_HPP_

#include <CL/cl.h>
#include <queue>
#include <vector>
#include "execution.hpp"
#include <stdio.h>
#include <limits.h>
#include <iostream>

using namespace std;

enum ArgType {
	BUFFER,
	OTHER
};

class OclArg {
private:
	size_t _size;
	ArgType _argType;
	cl_mem_flags _clMemFlags;
	void **_ptr;
public:
	OclArg(size_t size, ArgType argType, cl_mem_flags clMemFlags, void **ptr=NULL) :
		_size(size), _argType(argType), _clMemFlags(clMemFlags), _ptr(ptr) {};

	ArgType getArgType() const {
		return _argType;
	}

	void setArgType(ArgType argType) {
		this->_argType = argType;
	}

	cl_mem_flags getClMemFlag() const {
		return _clMemFlags;
	}

	void setClMemFlag(cl_mem_flags clMemFlag) {
		this->_clMemFlags = clMemFlag;
	}

	size_t getSize() const {
		return _size;
	}

	void setSize(size_t size) {
		this->_size = size;
	}

	void** getPtr() const {
		return _ptr;
	}

	void setPtr(void** ptr) {
		this->_ptr = ptr;
	}
};

class Launcher {
private:
	// Workgroup values
	unsigned int _min; 		/* Minimum block size 	*/
	unsigned int _max;		/* Maximum block size 	*/
	unsigned int _dim;		/* Number of dimensions */
	unsigned int _step;		/* Iteration step				*/

	// Global size
	cl_uint _work_dim;
	size_t *_global_work;

	// Executions
	std::queue<Execution> _executions;
	Execution *_best;

	// OpenCL model
	cl_context _context;
	std::vector<cl_platform_id> _platforms;
	std::vector<cl_device_id> _devices;
	std::vector<cl_mem> _buffers;
	cl_command_queue _commandQueue;
	cl_device_id _currentDevice;
	cl_kernel _kernel;
	cl_program _program;
	int _numArgs;

	// State flags
	bool _isInit;

	// Profiling flags
	unsigned int _iterations;

	// Private functions
	void init();
	void initPlatforms();
	void initDevices();
	void initContext();
	void initCommandQueue();
	Execution& runKernel(size_t *);
	void update(Execution &);

public:
	inline Launcher(unsigned int min, unsigned int max, unsigned int dim, unsigned int step) :
		_min(min), _max(max), _dim(dim), _step(step), _best(NULL), _isInit(false), _numArgs(0)
	{
		init();
		_global_work = new size_t[_dim];
		_iterations = 3;
		_best = new Execution(1,1,1,1,LLONG_MAX);
	};

	~Launcher();

	cl_command_queue getCommandQueue() { return _commandQueue; };

	cl_context getContext() { return _context; };

	cl_device_id getDevice() { return _currentDevice; };

	cl_kernel getKernel() { return _kernel; };

	void initKernel(const char *filename, const char *kernelName);

	void launch(const size_t *global_work);

	long long unsigned int getNumberOfExecutions() { return _executions.size(); }

	Execution * getBest() { return _best; }

	void allocAndSet(OclArg arg);

	void set(OclArg arg);

	void setArgs(OclArg arg) {
		if (arg.getArgType() == BUFFER) {
			allocAndSet(arg);
		} else {
			set(arg);
		}
	}

	template <class... T>
	void setArgs(OclArg arg, T... args)
	{
		if ( arg.getArgType() == BUFFER ) {
			allocAndSet(arg);
		} else {
			set(arg);
		}

		setArgs(args...);
	}

	static inline void clCheckError(cl_int clError, char* errorString) {
		if (clError != CL_SUCCESS) {
			fprintf(stderr,"ERROR: %d, %s\n", clError, errorString);
			exit( EXIT_FAILURE );
		}
	}
};
#endif /* LAUNCHER_HPP_ */
