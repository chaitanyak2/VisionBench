#include <gtest/gtest.h>
#include "faiss_storage.hpp"
#include "test_fixture.hpp"

using namespace visionbench;

TEST(FaissStorage, AddAndSearch)
{
    FaissStorage store(4);  // 4D FAISS index for testing

    CoreMetadata meta{};
    meta.image_width = 32;
    meta.image_height = 32;
    meta.stage = Stage::EMBEDDING;

    std::vector<float> e1 = {1.0, 0.0, 0.0, 0.0};
    std::vector<float> e2 = {0.0, 1.0, 0.0, 0.0};

    store.add(e1, "img1.jpg", meta);
    store.add(e2, "img2.jpg", meta);

    ASSERT_EQ(store.size(), 2);

    std::vector<float> query = {0.9, 0.1, 0.0, 0.0};
    auto results = store.search(query, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].first, "img1.jpg");
}

TEST(FaissStorage, ExportMetadata)
{
    FaissStorage store(2);
    CoreMetadata meta{};
    meta.image_width = 64;
    meta.image_height = 64;
    meta.stage_status = StageStatus::DONE;

    store.add({0.1, 0.2}, "test.jpg", meta);
    store.exportMetadata("metadata_test.json");

    std::ifstream f("metadata_test.json");
    ASSERT_TRUE(f.good());
}
