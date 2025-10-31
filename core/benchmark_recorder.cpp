// core/benchmark_recorder.cpp
#include "include/benchmark_recorder.hpp"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <thread>

namespace visionbench {

static inline uint64_t now_unix_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

BenchmarkRecorder& BenchmarkRecorder::instance() {
    static BenchmarkRecorder inst;
    return inst;
}

BenchmarkRecorder::BenchmarkRecorder() = default;

bool BenchmarkRecorder::init(const std::string& db_path) {
    db_path_ = db_path;
    {
        std::lock_guard<std::mutex> lock(db_mutex_);
        int rc = sqlite3_open(db_path_.c_str(), &db_);
        if (rc != SQLITE_OK) {
            std::cerr << "Cannot open sqlite db: " << sqlite3_errmsg(db_) << std::endl;
            sqlite3_close(db_);
            db_ = nullptr;
            return false;
        }
        // set WAL mode and pragmas
        char* err = nullptr;
        sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &err);
        if (err) { sqlite3_free(err); err = nullptr; }
        sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, &err);
        if (err) { sqlite3_free(err); err = nullptr; }
        sqlite3_exec(db_, "PRAGMA temp_store=MEMORY;", nullptr, nullptr, &err);
        if (err) { sqlite3_free(err); err = nullptr; }
    }

    ensure_schema();

    stop_flag_.store(false);
    writer_thread_.reset(new std::thread(&BenchmarkRecorder::writer_thread_fn, this));
    return true;
}

void BenchmarkRecorder::ensure_schema() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    if (!db_) return;
    const char* create_sql = R"sql(
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata_id INTEGER,
            module_name TEXT,
            params TEXT,
            timestamp_ms INTEGER,
            thread_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_metadata_id ON benchmarks(metadata_id);
    )sql";
    char* err = nullptr;
    int rc = sqlite3_exec(db_, create_sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "Error creating schema: " << (err ? err : "unknown") << std::endl;
        if (err) sqlite3_free(err);
    }
}



void BenchmarkRecorder::writer_thread_fn() {
    std::vector<BenchmarkEntry> batch;
    batch.reserve(batch_size_);
    while (!stop_flag_.load()) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (queue_.empty()) {
                cv_.wait_for(lock, batch_wait_ms_);
            }
            while (!queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        if (!batch.empty()) {
            // perform batched insert
            std::lock_guard<std::mutex> db_lock(db_mutex_);
            if (!db_) { batch.clear(); continue; }
            char* err = nullptr;
            sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err);
            if (err) { sqlite3_free(err); err = nullptr; }
            const char* insert_sql = "INSERT INTO benchmarks (metadata_id, module_name, params, timestamp_ms, thread_id) VALUES (?, ?, ?, ?, ?);";
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_, insert_sql, -1, &stmt, nullptr) != SQLITE_OK) {
                std::cerr << "Failed to prepare insert stmt: " << sqlite3_errmsg(db_) << std::endl;
            } else {
                for (auto &e : batch) {
                    sqlite3_reset(stmt);
                    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)e.metadata_id);
                    sqlite3_bind_text(stmt, 2, e.module_name.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(stmt, 3, e.params_serialized.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int64(stmt, 4, (sqlite3_int64)e.timestamp_unix_ms);
                    sqlite3_bind_int64(stmt, 5, (sqlite3_int64)e.thread_id);
                    if (sqlite3_step(stmt) != SQLITE_DONE) {
                        std::cerr << "Insert step failed: " << sqlite3_errmsg(db_) << std::endl;
                    }
                }
                sqlite3_finalize(stmt);
            }
            sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err);
            if (err) { sqlite3_free(err); err = nullptr; }
            batch.clear();
        }
    }

    // flush remaining
    {
        std::vector<BenchmarkEntry> rem;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            while (!queue_.empty()) {
                rem.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }
        if (!rem.empty()) {
            std::lock_guard<std::mutex> db_lock(db_mutex_);
            char* err = nullptr;
            sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err);
            if (err) { sqlite3_free(err); err = nullptr;  std::cerr <<" SQLITE ERR" << std::endl;}
            const char* insert_sql = "INSERT INTO benchmarks (metadata_id, module_name, params, timestamp_ms, thread_id) VALUES (?, ?, ?, ?, ?);";
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(db_, insert_sql, -1, &stmt, nullptr) == SQLITE_OK) {
                for (auto &e : rem) {
                    sqlite3_reset(stmt);
                    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)e.metadata_id);
                    sqlite3_bind_text(stmt, 2, e.module_name.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(stmt, 3, e.params_serialized.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int64(stmt, 4, (sqlite3_int64)e.timestamp_unix_ms);
                    sqlite3_bind_int64(stmt, 5, (sqlite3_int64)e.thread_id);
                    sqlite3_step(stmt);
                }
                sqlite3_finalize(stmt);
            }
            sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err);
            if (err) { sqlite3_free(err); err = nullptr; 
            std::cerr <<" SQLITE ERR" << std::endl;
        }
        }
    }

    // close DB (do not close here; externally in shutdown)
}

void BenchmarkRecorder::record(const BenchmarkEntry& entry) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // augment entry timestamp/thread if not provided
        BenchmarkEntry copy = entry;
        if (copy.timestamp_unix_ms == 0) copy.timestamp_unix_ms = now_unix_ms();
        if (copy.thread_id == 0) copy.thread_id = (uint64_t) std::hash<std::thread::id>{}(std::this_thread::get_id());
        queue_.push_back(std::move(copy));
    }
    cv_.notify_one();
}

std::vector<BenchmarkEntry> BenchmarkRecorder::fetch_all() {
    std::vector<BenchmarkEntry> results;
    {
        std::lock_guard<std::mutex> lock(db_mutex_);
        if (!db_) {
            if (!reopen_for_read()) {
                std::cerr << "fetch_all(): unable to reopen DB for read\n";
                return results;
            }
        }

        const char* q = "SELECT metadata_id, module_name, params, timestamp_ms, thread_id "
                        "FROM benchmarks ORDER BY id ASC;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "fetch_all prepare failed: " << sqlite3_errmsg(db_) << std::endl;
            return results;
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            BenchmarkEntry e;
            e.metadata_id = (uint64_t) sqlite3_column_int64(stmt, 0);
            const unsigned char* txt = sqlite3_column_text(stmt, 1);
            e.module_name = txt ? reinterpret_cast<const char*>(txt) : "";
            const unsigned char* p = sqlite3_column_text(stmt, 2);
            e.params_serialized = p ? reinterpret_cast<const char*>(p) : "";
            e.timestamp_unix_ms = (uint64_t) sqlite3_column_int64(stmt, 3);
            e.thread_id = (uint64_t) sqlite3_column_int64(stmt, 4);
            results.push_back(std::move(e));
        }
        sqlite3_finalize(stmt);
    }
    return results;
}

bool BenchmarkRecorder::reopen_for_read() {
    
    if (sqlite3_open_v2(db_path_.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to reopen database for read: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    return true;
}

void BenchmarkRecorder::shutdown() {
    stop_flag_.store(true);
    cv_.notify_all();
    
    if (writer_thread_ && writer_thread_->joinable()) {
        writer_thread_->join();   // ensures flush finished
    } else {
        std::cerr << "[WARN] Writer thread not joinable during shutdown.\n";
    }

    cv_.notify_all(); 

    {
       
        std::lock_guard<std::mutex> lock(db_mutex_);
        if (db_) {
            sqlite3_exec(db_, "PRAGMA wal_checkpoint(FULL);", nullptr, nullptr, nullptr);
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }
    
}

BenchmarkRecorder::~BenchmarkRecorder() {
    shutdown();
}

size_t BenchmarkRecorder::queue_size() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

} // namespace visionbench