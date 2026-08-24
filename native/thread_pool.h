#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace chess_rl_native {

// A persistent worker pool for index-parallel work.
//
// The Actor previously spawned and joined a fresh std::vector<std::thread> on
// EVERY gather_leaves and EVERY advance — thousands of thread creations per
// game, and it sized the pool to the game count, so 20 concurrent games
// oversubscribed an 8-core/16-thread CPU.  This pool is created once and
// reused: workers park on a condition variable between jobs.
//
// Determinism: parallel_for makes no ordering guarantee about WHEN each index
// runs, only that fn(i) has run exactly once for every i in [0, n) before it
// returns.  Callers that need deterministic output (the Actor's merged CSR)
// must write into per-index slots and merge serially afterwards, which is
// exactly what Actor::gather_leaves does.
//
// Exceptions thrown by fn are captured and rethrown on the calling thread
// after all indices have been attempted, so a throwing game can never leave
// workers spinning or the pool half-joined.
class ThreadPool {
  public:
    explicit ThreadPool(int threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Runs fn(i) for every i in [0, n) and blocks until all have completed.
    // n <= 0 returns immediately.  Not re-entrant and not safe to call
    // concurrently from multiple threads on the same pool.
    void parallel_for(int n, const std::function<void(int)>& fn);

    [[nodiscard]] int size() const noexcept {
        return static_cast<int>(workers_.size());
    }

    // Clamp a desired worker count to something the machine can actually run:
    // at least 1, never more than std::thread::hardware_concurrency() (when
    // that is reported).  Oversubscribing only adds context switches.
    [[nodiscard]] static int clamp_threads(int desired) noexcept;

  private:
    void worker_loop();

    std::vector<std::thread> workers_;

    std::mutex mu_;
    std::condition_variable job_cv_;    // workers wait for a new job
    std::condition_variable done_cv_;   // caller waits for job completion

    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0;
    std::uint64_t job_id_ = 0;          // monotonically increasing job counter
    std::atomic<int> next_{0};          // next index to claim
    int active_ = 0;                    // workers still inside the current job
    bool stop_ = false;
    std::exception_ptr error_;          // first exception thrown by fn
};

}  // namespace chess_rl_native
