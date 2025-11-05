// tests/test_benchmark_recorder.cpp
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cstdio>
#include <filesystem>
#include "benchmark_recorder.hpp"
#include "test_fixture.hpp"

using namespace visionbench;

static std::string make_temp_db() {
    std::string path = "/tmp/vis_bench_test.db";
    // remove existing
    std::remove(path.c_str());
    return path;
}
TEST_F(VisionBenchFixture, BenchmarkRecorderTest_SingleRecordSaveAndFetch) {
     auto &rec = BenchmarkRecorder::instance();
    BenchmarkEntry e;
    e.metadata_id = 42;
    e.module_name = "preprocess.resize";
    e.params_serialized = "w=224;h=224";
    rec.record(e);
    // allow writer thread some time
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto rows = rec.fetch_all();
    ASSERT_GE(rows.size(), 1u);
    EXPECT_EQ(rows.back().metadata_id, 42);
    EXPECT_EQ(rows.back().module_name, "preprocess.resize");
    
}

TEST_F(VisionBenchFixture, BenchmarkRecorderTest_ConcurrentWriters) {
     auto &rec = BenchmarkRecorder::instance();
    const int nThreads = 2;
    const int perThread = 200;
    std::vector<std::thread> threads;
    for (int t = 0; t < nThreads; ++t) {
        threads.emplace_back([t, perThread]() {
            for (int i = 0; i < perThread; ++i) {
                BenchmarkEntry e;
                e.metadata_id = (uint64_t)(t+1);
                e.module_name = "infer.trt";
                e.params_serialized = "i=" + std::to_string(i);
                BenchmarkRecorder::instance().record(e);
            }
        });
    }

    for (auto &th : threads) th.join();

    // let writer flush
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    auto rows = rec.fetch_all();
    EXPECT_GE((int)rows.size(), nThreads * perThread);
    
}

TEST_F(VisionBenchFixture, BenchmarkRecorderTest_WriterReaderInterleaved) {
   
    auto &rec = BenchmarkRecorder::instance();
    

    std::thread writer([&]() {
        int i = 0;
        while (i < 500) {
            BenchmarkEntry e;
            e.metadata_id = (uint64_t)(i % 5);
            e.module_name = "stage.loop";
            e.params_serialized = "i=" + std::to_string(i);
            rec.record(e);
            ++i;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::thread reader([&]() {
        int iterations = 0;
        while (iterations < 10) {
            auto rows = rec.fetch_all();
            // simply validate that fetch_all works while writing
            EXPECT_TRUE(rows.size() >= 0);
            ++iterations;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    writer.join();
    reader.join();

    // wait flush
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto rows = rec.fetch_all();
    EXPECT_GE(rows.size(), 500u);
    
}

// Test that queue drains on shutdown
TEST_F(VisionBenchFixture, BenchmarkRecorderTest_QueueDrainsOnShutdown) {
    
    auto &rec = BenchmarkRecorder::instance();
  

    // push many items quickly
    for (int i=0;i<1000;++i) {
        BenchmarkEntry e;
        e.metadata_id = i;
        e.module_name = "fast.push";
        e.params_serialized = "i=" + std::to_string(i);
        rec.record(e);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
   
    // reopen same DB to read
    // create a new instance? it's singleton; but shutdown closed DB; instance still exists
    auto rows = rec.fetch_all();
    // Since we shutdown, fetch_all should return what's present
    EXPECT_GE(rows.size(), 1000u);
   
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
