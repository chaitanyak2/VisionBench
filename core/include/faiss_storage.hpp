#pragma once
#include <faiss/IndexFlat.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include "core_metadata.h"  // your existing CoreMetadata struct

class FaissStorage {
public:
    explicit FaissStorage(int dim, const std::string &index_type = "Flat");

    // Add one embedding + associated metadata
    void add(const std::vector<float> &embedding,
             const std::string &image_path,
             const CoreMetadata &meta);

    // Search nearest neighbors
    std::vector<std::pair<std::string, float>> search(const std::vector<float> &query, int k = 5) const;

    // Save / load FAISS index
    void saveIndex(const std::string &path) const;
    void loadIndex(const std::string &path);

    // Export metadata mapping to JSON
    void exportMetadata(const std::string &json_path) const;

    // Getters for tests
    size_t size() const { return image_links_.size(); }

private:
    std::unique_ptr<faiss::Index> index_;
    int dim_;

    struct Entry {
        std::string image_path;
        CoreMetadata metadata;
    };
    std::vector<Entry> entries_;
};
