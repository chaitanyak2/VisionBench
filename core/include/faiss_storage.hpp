#pragma once
#include <faiss/IndexFlat.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "metadata.hpp"  // your existing CoreMetadata struct

namespace visionbench{
class FaissStorage {
public:
    explicit FaissStorage(int dim, const std::string &index_type = "Flat");

    void add(const std::vector<float> &embedding,
             const std::string &image_path,
             const CoreMetadata &meta);

    std::vector<std::pair<std::string, float>> search(const std::vector<float> &query, int k = 5) const;

    void saveIndex(const std::string &path) const;
    void loadIndex(const std::string &path);
    void exportMetadata(const std::string &json_path) const;

    size_t size() const { return entries_.size(); }

private:
    std::unique_ptr<faiss::Index> index_;
    int dim_;

    struct Entry {
        std::string image_path;
        visionbench::CoreMetadata metadata;   // ✅ make sure it's named exactly this
    };

    std::vector<Entry> entries_;
};
}