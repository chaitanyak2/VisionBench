// core/metadata.hpp
#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <unordered_map>

namespace visionbench {

enum class Stage : uint8_t { READ=0, PREPROCESS=1, INFERENCE=2, POSTPROCESS=3, SINK=4, EMBEDDING=5 };
enum class StageStatus : uint8_t { VACANT=0, QUEUED=1, DONE=2, FAILED=3, DROPPED=4  };
enum class OverallStatus : uint8_t { IN_PROGRESS=0, DONE=1, FAILED=2, DROPPED=3, TO_FREE=4 };
enum class LocationType : uint8_t { LOCAL=0, GDRIVE=1, S3=2, STREAM=3 };
enum class ChannelType : uint8_t { GRAY=1, RGB=3, RGBA=4 };
enum class DeviceType : uint8_t { CPU=0, CUDA=1 };

struct PreprocessMeta {
    // Simple POD; in real-world this holds a shared_ptr to tensor/mat
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    // additional fields can be added: format, dtype, pointer to buffer, etc.
};

struct InferenceMeta {
    uint64_t embedding_id = 0;    // id which will map into FAISS / vectordb
    uint32_t embedding_dim = 0;
    std::string model_name;
    double inference_latency_ms = 0.0;
    DeviceType device = DeviceType::CPU;
    // pointer to embedding buffer may be added later as shared_ptr
};

struct CoreMetadata {
    uint64_t metadata_id = 0;    // unique internal id (use as FAISS id if desired)
    std::string input_id;        // external id (e.g., filename or UUID)
    std::string image_location;  // path or URI
    LocationType location_type = LocationType::LOCAL;
    Stage stage = Stage::READ;
    StageStatus stage_status = StageStatus::QUEUED;
    OverallStatus overall_status = OverallStatus::IN_PROGRESS;
    uint64_t timestamp_unix_ms = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    ChannelType channels = ChannelType::RGB;

    // optional per-stage data
    std::shared_ptr<PreprocessMeta> preprocess_meta;
    std::shared_ptr<InferenceMeta> inference_meta;

    // free-form attributes (if needed)
    std::unordered_map<std::string, std::string> attributes;
};

} // namespace vis
