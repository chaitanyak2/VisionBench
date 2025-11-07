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

FaissStorage::FaissStorage(int dim, const std::string &index_type, const std::string &index_path)
    : dim_(dim), current_index_path_(index_path)
{
    namespace fs = std::filesystem;

    if (fs::exists(index_path)) {
        try {
            std::cout << "[FaissStorage] Loading existing FAISS index from: " << index_path << std::endl;
            index_.reset(faiss::read_index(index_path.c_str()));
            index_loaded_ = true;
        } catch (const std::exception &e) {
            std::cerr << "[FaissStorage] Failed to load index (" << e.what()
                      << "), creating a new one instead." << std::endl;
            createNewIndex(index_type);
        }
    } else {
        createNewIndex(index_type);
    }
}

void FaissStorage::createNewIndex(const std::string &index_type) {
    std::cout << "[FaissStorage] Creating new FAISS index of type: " << index_type << std::endl;
    if (index_type == "Flat")
        index_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    else
        throw std::invalid_argument("Unsupported FAISS index type: " + index_type);
    index_loaded_ = true;
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
    namespace fs = std::filesystem;

    if (fs::exists(path)) {
        std::cerr << "[FaissStorage] Index file already exists (" << path
                  << "), skipping overwrite.\n";
        return;
    }

    std::string tmp_path = path + ".tmp";
    faiss::write_index(index_.get(), tmp_path.c_str());
    fs::rename(tmp_path, path); //
    std::cout << "[FaissStorage] Index saved to: " << path << std::endl;
}

bool FaissStorage::indexExists(const std::string &path) const {
    return std::filesystem::exists(path);
}

bool FaissStorage::loadIndex(const std::string &path, bool force_reload)
{
    namespace fs = std::filesystem;

    // --- Step 1: Safety checks
    if (!fs::exists(path)) {
        std::cerr << "[FaissStorage] loadIndex(): File does not exist: " << path << std::endl;
        return false;
    }

    // --- Step 2: Prevent accidental reload if index already loaded
    if (index_loaded_ && !force_reload) {
        std::cout << "[FaissStorage] loadIndex(): Index already loaded, skipping reload." << std::endl;
        return true;
    }

    try {
        // --- Step 3: Attempt to load
        std::unique_ptr<faiss::Index> new_index(faiss::read_index(path.c_str()));
        if (!new_index) {
            std::cerr << "[FaissStorage] loadIndex(): Failed to read FAISS index from " << path << std::endl;
            return false;
        }

        // --- Step 4: Commit swap
        index_.swap(new_index);
        index_loaded_ = true;
        std::cout << "[FaissStorage] loadIndex(): Successfully loaded index from " << path << std::endl;
        return true;
    }
    catch (const std::exception &e) {
        std::cerr << "[FaissStorage] loadIndex(): Exception while reading index: " << e.what() << std::endl;
        return false;
    }
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
