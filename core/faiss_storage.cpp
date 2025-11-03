#include "faiss_storage.hpp"
#include <faiss/index_io.h>   // ✅ required for read_index / write_index
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
using namespace visionbench;

FaissStorage::FaissStorage(int dim, const std::string &index_type)
    : dim_(dim)
{
    if (index_type == "Flat")
        index_ = std::make_unique<faiss::IndexFlatL2>(dim);
    else
        throw std::invalid_argument("Unsupported FAISS index type: " + index_type);
}

void FaissStorage::add(const std::vector<float> &embedding,
                       const std::string &image_path,
                       const CoreMetadata &meta)
{
    if (embedding.size() != static_cast<size_t>(dim_))
        throw std::runtime_error("Embedding dimension mismatch.");

    index_->add(1, embedding.data());
    entries_.push_back(Entry{image_path, meta});
}

std::vector<std::pair<std::string, float>>
FaissStorage::search(const std::vector<float> &query, int k) const
{
    if (query.size() != static_cast<size_t>(dim_))
        throw std::runtime_error("Query dimension mismatch.");

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);   // ✅ fixed type

    index_->search(1, query.data(), k, distances.data(), indices.data());

    std::vector<std::pair<std::string, float>> results;
    for (int i = 0; i < k; ++i) {
        if (indices[i] < 0 || indices[i] >= (faiss::idx_t)entries_.size())
            continue;
        results.emplace_back(entries_[indices[i]].image_path, distances[i]);
    }
    return results;
}

void FaissStorage::saveIndex(const std::string &path) const
{
    faiss::write_index(index_.get(), path.c_str());   // ✅ now works
}

void FaissStorage::loadIndex(const std::string &path)
{
    index_.reset(faiss::read_index(path.c_str()));    // ✅ now works
}

void FaissStorage::exportMetadata(const std::string &json_path) const
{
    json j = json::object();

    for (size_t i = 0; i < entries_.size(); ++i) {
        const auto &entry = entries_[i];
        const auto &meta = entry.metadata;   // ✅ consistent with struct definition

        json rec;
        rec["image"] = entry.image_path;
        rec["width"] = meta.image_width;
        rec["height"] = meta.image_height;
        rec["channels"] = static_cast<int>(meta.channels);
        rec["stage"] = static_cast<int>(meta.stage);
        rec["stage_status"] = static_cast<int>(meta.stage_status);
        rec["timestamp_unix_ms"] = meta.timestamp_unix_ms;

        j[std::to_string(i)] = rec;
    }

    std::ofstream ofs(json_path);
    ofs << std::setw(4) << j;
}
