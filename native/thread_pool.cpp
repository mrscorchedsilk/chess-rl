#include "thread_pool.h"

#include <algorithm>

namespace chess_rl_native {

int ThreadPool::clamp_threads(int desired) noexcept {
    const unsigned hw = std::thread::hardware_concurrency();
    int capped = desired;
    if (hw > 0)
        capped = std::min(desired, static_cast<int>(hw));
    return std::max(1, capped);
}

ThreadPool::ThreadPool(int threads) {
    const int n = clamp_threads(threads);
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        workers_.emplace_back([this]() { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    job_cv_.notify_all();
    for (std::thread& t : workers_)
        if (t.joinable()) t.join();
}

void ThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(mu_);
        job_cv_.wait(lock, [&]() { return stop_ || job_id_ != seen; });
        if (stop_) return;
        seen = job_id_;
        const std::function<void(int)>* fn = fn_;
        const int n = n_;
        lock.unlock();

        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            try {
                (*fn)(i);
            } catch (...) {
                std::lock_guard<std::mutex> elock(mu_);
                if (!error_) error_ = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> dlock(mu_);
            if (--active_ == 0) done_cv_.notify_all();
        }
    }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // Single worker, or a single item: run inline and skip the handshake.
    if (workers_.empty() || n == 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }

    std::exception_ptr err;
    {
        std::unique_lock<std::mutex> lock(mu_);
        fn_ = &fn;
        n_ = n;
        next_.store(0, std::memory_order_relaxed);
        active_ = static_cast<int>(workers_.size());
        error_ = nullptr;
        ++job_id_;
        job_cv_.notify_all();
        done_cv_.wait(lock, [&]() { return active_ == 0; });
        fn_ = nullptr;
        n_ = 0;
        err = error_;
        error_ = nullptr;
    }
    if (err) std::rethrow_exception(err);
}

}  // namespace chess_rl_native
