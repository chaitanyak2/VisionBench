#include <gtest/gtest.h>
#include "faiss_storage.hpp"
#include "test_fixture.hpp"

using namespace visionbench;

TEST_F(VisionBenchFixture, FaissStorage_AddAndSearch)
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

TEST_F(VisionBenchFixture, FaissStorage_ExportMetadata)
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

TEST_F(VisionBenchFixture, FaissStorageTest_SafePersistence) {
    
    {
        FaissStorage store(576, "Flat", faiss_index_path);
        std::vector<float> emb(576, 0.1f);
        store.add(emb, "img.png", CoreMetadata{});
        store.saveIndex(faiss_index_path);
        ASSERT_TRUE(std::filesystem::exists(faiss_index_path));
    }

    // Recreate should load, not overwrite
    {
        FaissStorage store(576, "Flat", faiss_index_path);
        ASSERT_TRUE(store.indexExists(faiss_index_path));
        store.saveIndex(faiss_index_path); // Should skip overwrite
    }

    std::filesystem::remove(faiss_index_path);
}

TEST_F(VisionBenchFixture, FaissStorageTest_LoadIndexBehavior)
{
    namespace fs = std::filesystem;
   
    int dim = 576;

    // --- Step 1: Create and save a test index
    {
        FaissStorage store(dim, "Flat", faiss_index_path);
        std::vector<float> emb(dim, 0.5f);
        store.add(emb, "imgA.jpg", CoreMetadata{});
        store.saveIndex(faiss_index_path);
        ASSERT_TRUE(fs::exists(faiss_index_path));
    }

    // --- Step 2: Load existing index successfully
    {
        FaissStorage loader(dim, "Flat", ""); // Empty path → load manually
        bool ok = loader.loadIndex(faiss_index_path);
        EXPECT_TRUE(ok);
    }

    // --- Step 3: Skip reload if already loaded
    {
        FaissStorage loader(dim, "Flat", "");
        loader.loadIndex(faiss_index_path);
        bool skipped = loader.loadIndex(faiss_index_path); // no force
        EXPECT_TRUE(skipped);
    }

    // --- Step 4: Force reload
    {
        FaissStorage loader(dim, "Flat", "");
        loader.loadIndex(faiss_index_path);
        bool forced = loader.loadIndex(faiss_index_path, true);
        EXPECT_TRUE(forced);
    }

    // --- Step 5: Handle missing file gracefully
    {
        FaissStorage loader(dim, "Flat", "");
        bool ok = loader.loadIndex("nonexistent_file.index");
        EXPECT_FALSE(ok);
    }

    fs::remove(faiss_index_path);
}
