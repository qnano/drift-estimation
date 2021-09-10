// C++ multithread helper code
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <mutex>
#include <atomic>
#include <future>
#include "ctpl_stl.h"

template<typename TAction>
void LockedAction(std::mutex& m, TAction fn) {
	std::lock_guard<std::mutex> l(m);
	fn();
}
template<typename TFunc>
auto LockedFunction(std::mutex& m, TFunc fn) -> decltype(fn()) {
	std::lock_guard<std::mutex> l(m);
	return fn();
}

template<typename Function>
void ParallelFor(int n, Function f)
{
	ctpl::thread_pool pool(std::thread::hardware_concurrency());

	for (int i = 0; i < n; i++)
		pool.push([=](int id) { f(i); });

	pool.stop(true);
}
