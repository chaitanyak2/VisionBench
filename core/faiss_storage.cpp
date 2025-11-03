#include "faiss_storage.hpp"
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

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
    entries_.push_back({image_path, meta});
}

std::vector<std::pair<std::string, float>>
FaissStorage::search(const std::vector<float> &query, int k) const
{
    if (query.size() != static_cast<size_t>(dim_))
        throw std::runtime_error("Query dimension mismatch.");

    std::vector<float> distances(k);
    std::vector<faiss::Index::idx_t> indices(k);

    index_->search(1, query.data(), k, distances.data(), indices.data());

    std::vector<std::pair<std::string, float>> results;
    for (int i = 0; i < k; ++i) {
        if (indices[i] < 0 || indices[i] >= (faiss::Index::idx_t)entries_.size())
            continue;
        results.emplace_back(entries_[indices[i]].image_path, distances[i]);
    }
    return results;
}

void FaissStorage::saveIndex(const std::string &path) const
{
    faiss::write_index(index_.get(), path.c_str());
}

void FaissStorage::loadIndex(const std::string &path)
{
    index_.reset(faiss::read_index(path.c_str()));
}

void FaissStorage::exportMetadata(const std::string &json_path) const
{
    json j;
    for (size_t i = 0; i < entries_.size(); ++i) {
        const auto &entry = entries_[i];
        j[std::to_string(i)] = {
            {"image", entry.image_path},
            {"width", entry.metadata.image_width},
            {"height", entry.metadata.image_height},
            {"channels", static_cast<int>(entry.metadata.channels)},
            {"stage", static_cast<int>(entry.metadata.stage)},
            {"stage_status", static_cast<int>(entry.metadata.stage_status)},
            {"timestamp_unix_ms", entry.metadata.timestamp_unix_ms}
        };
    }

    std::ofstream ofs(json_path);
    ofs << std::setw(4) << j;
}
