#pragma once

#include "tbb/tbb.h"

using namespace tbb;


template <typename T_numtype>
class LoopObject {
	T_numtype *data;
public:
	void operator()(const blocked_range<size_t>& r) const {
		T_numtype *a = data;
		for (size_t i = r.begin(); i != r.end(); ++i)
			Foo(a[i]);
	}
	LoopObject(T_numtype a[]) :
		data(a)
	{}
};
