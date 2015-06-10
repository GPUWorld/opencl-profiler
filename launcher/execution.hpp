/*
 * execution.hpp
 *
 *  Created on: 27/5/2015
 *      Author: diego
 */

#ifndef EXECUTION_HPP_
#define EXECUTION_HPP_

class Execution {
private:
	unsigned char _ndims;
	unsigned int _x;
	unsigned int _y;
	unsigned int _z;
	long long int _time;

public:
	Execution(unsigned int ndims, unsigned int x, unsigned int y, unsigned int z, long long int time) :
			_ndims(ndims), _x(x), _y(y), _z(z), _time(time) { }

	unsigned char getNdims() const {
		return _ndims;
	}

	long long int getTime() const {
		return _time;
	}

	unsigned int getX() const {
		return _x;
	}

	unsigned int getY() const {
		return _y;
	}

	unsigned int getZ() const {
		return _z;
	}

	bool operator<(const Execution& execution) { return _time < execution.getTime(); }
};

#endif /* EXECUTION_HPP_ */
