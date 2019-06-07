#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>


	class threadpool
	{
		using Task = std::function<void()>;

		std::vector<std::thread> pool;
		

	public:
		threadpool(unsigned short size = 4) 
		{
			
		}
		~threadpool()
		{
			joinAll();
		}

		void joinAll() {
			for (std::thread& thread : pool) {
				if (thread.joinable())
					thread.join(); 
			}
			pool.clear();
		}

		template<class F, class... Args>
		auto commit(F&& f, Args&&... args) 
		{
			pool.emplace_back(thread(std::bind(std::forward<F>(f), std::forward<Args>(args)...)));
			

			return;
		}


	};



threadpool g_pool(8);



#endif