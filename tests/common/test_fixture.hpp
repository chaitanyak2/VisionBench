#pragma once
#include <gst/gst.h>
#include "benchmark_recorder.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>


using namespace visionbench;
static std::string getConfigValue(const std::string& key) {
    std::ifstream f("tests/config/config.json");
    nlohmann::json j;
    f >> j;
    return j[key];
}

static std::filesystem::path resolve_relative_to_project_root(const std::string &relative) {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();

    // Walk up directories until we find one containing "data"
    fs::path probe = cwd;
    for (int i = 0; i < 5; ++i) { // up to 5 levels up
        if (fs::exists(probe / "data")) {
            return probe / relative;
        }
        probe = probe.parent_path();
    }

    // fallback: assume relative to current directory
    return cwd / relative;
}

class VisionBenchFixture : public ::testing::Test {
    protected:
    static inline std::string data_dir;
    static inline std::string nn_model_path;
    static inline std::string db_path;
    static inline std::string faiss_index_path;
    static inline std::string faiss_metadata_jsonpath;

public:
    static void SetUpTestSuite() {
        static bool initialized = false;
        if (initialized) return;
        initialized = true;
        std::string config_path = "tests/config/config.json";
        // 1) Initialize GStreamer safely
        int argc = 0;
        char **argv = nullptr;
        if (!gst_init_check(&argc, &argv, nullptr)) {
            std::cerr << "[VisionBenchFixture] Warning: gst_init_check failed or already initialized\n";
        }
        // Load JSON config once
        // 2) Load JSON config if it exists (robustly)
        std::ifstream f(config_path);
        if (f.good()) {
            try {
                nlohmann::json cfg;
                f >> cfg;
                if (cfg.contains("data_dir") && cfg["data_dir"].is_string()) data_dir = cfg["data_dir"];
                if (cfg.contains("model_path") && cfg["model_path"].is_string()) nn_model_path = cfg["model_path"];
                if (cfg.contains("db_path") && cfg["db_path"].is_string()) db_path = cfg["db_path"];
                if (cfg.contains("faiss_index_path") && cfg["faiss_index_path"].is_string()) faiss_index_path = cfg["faiss_index_path"];
                if (cfg.contains("faiss_metadata_jsonpath") && cfg["faiss_metadata_jsonpath"].is_string()) faiss_metadata_jsonpath = cfg["faiss_metadata_jsonpath"];

                // After parsing JSON config, normalize all relevant paths:
nn_model_path = resolve_relative_to_project_root(nn_model_path).string();
data_dir = resolve_relative_to_project_root(data_dir).string();
db_path = resolve_relative_to_project_root(db_path).string();
faiss_index_path = resolve_relative_to_project_root(faiss_index_path).string();
faiss_metadata_jsonpath = resolve_relative_to_project_root(faiss_metadata_jsonpath).string();
std::cout << "model_path "<< nn_model_path << std::endl;
std::cout << "data_dir "<< data_dir << std::endl;
std::cout << "db_path "<< db_path << std::endl;
std::cout << "faiss_index_path "<< faiss_index_path << std::endl;
std::cout << "faiss_metadata_jsonpath "<< faiss_index_path << std::endl;

std::cerr << "[VisionBenchFixture] Using data_dir=" << data_dir << "\n";
std::cerr << "[VisionBenchFixture] Using model_path=" << nn_model_path << "\n";
            } catch (const nlohmann::json::parse_error &e) {
                std::cerr << "[VisionBenchFixture] Warning: failed to parse " << config_path
                          << " — using defaults. parse_error: " << e.what() << "\n";
            } catch (const std::exception &e) {
                std::cerr << "[VisionBenchFixture] Warning: exception while reading " << config_path
                          << " — using defaults. exception: " << e.what() << "\n";
            }
        }else {
            std::cerr << "[VisionBenchFixture] Info: config " << config_path << " not found — using defaults\n";
        }

        try {
            auto &recorder = BenchmarkRecorder::instance();
            recorder.init(db_path);
        } catch (const std::exception &e) {
            std::cerr << "[VisionBenchFixture] Error: BenchmarkRecorder::init threw: " << e.what()
                      << " — continuing tests but recorder may be non-functional\n";
        } catch (...) {
            std::cerr << "[VisionBenchFixture] Error: unknown exception during recorder.init() — continuing\n";
        }
    }
    

    static void TearDownTestSuite() {
        // 1) Try to shutdown recorder gracefully
        try {
            auto &recorder = BenchmarkRecorder::instance();
            recorder.shutdown();
            // small wait to allow background threads to join - helpful to avoid "not joinable" warnings
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        } catch (const std::exception &e) {
            std::cerr << "[VisionBenchFixture] Warning: exception during recorder.shutdown(): " << e.what() << "\n";
        } catch (...) {
            std::cerr << "[VisionBenchFixture] Warning: unknown exception during recorder.shutdown()\n";
        }

        // 2) Optionally cleanup temporary artifacts created by tests (safe-guard)
        // std::filesystem::remove(db_path); // uncomment if you want to delete DB after entire suite
        // std::filesystem::remove(faiss_index_path);
    }
};