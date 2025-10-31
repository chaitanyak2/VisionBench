// core/benchmark_recorder.hpp
#pragma once
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <chrono>

#include "metadata.hpp"

struct sqlite3;

namespace visionbench {

struct BenchmarkEntry {
    uint64_t metadata_id = 0;
    std::string module_name; // e.g., "preprocess.resize"
    // params serialized as simple key=value; pairs separated by ';' for now
    std::string params_serialized;
    uint64_t timestamp_unix_ms = 0;
    uint64_t thread_id = 0;
};

class BenchmarkRecorder {
public:
    // singleton accessor
    static BenchmarkRecorder& instance();

    // initialize with db path; creates table if necessary and sets WAL mode
    // returns true if init ok
    bool init(const std::string& db_path);

    // push an entry; non-blocking and returns immediately
    void record(const BenchmarkEntry& entry);

    // fetch all rows (for testing / inspection). Blocks while performing read.
    std::vector<BenchmarkEntry> fetch_all();

    // graceful shutdown: flush queue, stop background thread
    void shutdown();

    // For tests: access queue size (not part of public API usually)
    size_t queue_size();

    // destructor ensures shutdown
    ~BenchmarkRecorder();

    // non-copyable
    BenchmarkRecorder(const BenchmarkRecorder&) = delete;
    BenchmarkRecorder& operator=(const BenchmarkRecorder&) = delete;

private:
    BenchmarkRecorder();

    void writer_thread_fn();
    void open_db();
    void close_db();
    void ensure_schema();
     bool reopen_for_read();

    // internals
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::deque<BenchmarkEntry> queue_;
    std::unique_ptr<std::thread> writer_thread_;
    std::atomic<bool> stop_flag_ {false};

    bool done_ = false;

    // sqlite handle
    sqlite3* db_ = nullptr;
    std::string db_path_;
    std::mutex db_mutex_; // protects sqlite handle for reads (writes happen only in writer thread)
    size_t batch_size_ = 64;
    std::chrono::milliseconds batch_wait_ms_ {200}; // flush every 200ms if queue non-empty
};

} // namespace vis