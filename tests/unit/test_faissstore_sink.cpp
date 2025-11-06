// tests/test_faissstore_sink.cpp

#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <torch/torch.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"
#include "gstfaissstore.hpp"
#include "faiss_storage.hpp"
#include <filesystem>
#include "test_fixture.hpp"

extern "C" {
    gboolean gst_faiss_store_plugin_init(GstPlugin *plugin);
    GstFlowReturn gst_faiss_store_render(GstBaseSink *sink, GstBuffer *buffer);
    gboolean gst_faiss_store_stop(GstBaseSink *basesink);
    gboolean gst_faiss_store_start(GstBaseSink *basesink);
}

TEST_F(VisionBenchFixture, FaissStore_AttachesMetadataAndStoresEmbedding)
{
    
    // ✅ Register plugin statically
    gboolean registered = gst_plugin_register_static(
        GST_VERSION_MAJOR, GST_VERSION_MINOR,
        "faissstore", "FAISS sink plugin",
        gst_faiss_store_plugin_init,
        "1.0", "LGPL",
        "visionbench", "visionbench",
        "https://github.com/chaitanya.k2/VisionBench");

    ASSERT_TRUE(registered) << "Failed to register faissstore plugin.";

    // ✅ Create element instance
    GstElement *element = gst_element_factory_make("faissstore", nullptr);
    ASSERT_NE(element, nullptr) << "Failed to create faissstore element.";
    // Cast element to base sink
GstBaseSink *sink = GST_BASE_SINK(element);

    // ✅ Configure sink output paths
    g_object_set(G_OBJECT(element),
                 "index-path", faiss_index_path.c_str(),
                 "metadata-path", faiss_metadata_jsonpath.c_str(),
                 "export-on-eos", TRUE,
                 NULL);

    // ✅ Create fake embedding buffer (pretend it came from torchinfer)
    const int embedding_dim = 576;
torch::Tensor embedding = torch::rand({embedding_dim}, torch::kFloat32);

    GstBuffer *buffer = gst_buffer_new_and_alloc(embedding_dim * sizeof(float));
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, embedding.data_ptr<float>(), embedding_dim * sizeof(float));
    gst_buffer_unmap(buffer, &map);

    // ✅ Attach CoreMetadata (as your other elements do)
    GstCoreMeta *meta =
        (GstCoreMeta *)gst_buffer_add_meta(buffer, GST_CORE_META_INFO, nullptr);
    ASSERT_NE(meta, nullptr) << "GstCoreMeta registration failed.";
    meta->core_meta = std::make_shared<visionbench::CoreMetadata>();

    meta->core_meta->stage = visionbench::Stage::INFERENCE;
    meta->core_meta->stage_status = visionbench::StageStatus::DONE;
    meta->core_meta->image_width = 224;
    meta->core_meta->image_height = 224;
    meta->core_meta->channels = visionbench::ChannelType::RGB;
    meta->core_meta->image_location = "dummy";
ASSERT_TRUE(gst_faiss_store_start(sink)) << "gst_base_sink_start failed";
GstFlowReturn ret = gst_faiss_store_render(sink, buffer);

EXPECT_EQ(ret, GST_FLOW_OK) << "Sink render() failed.";

gst_faiss_store_stop(sink);
EXPECT_TRUE(std::filesystem::exists(faiss_index_path));
EXPECT_TRUE(std::filesystem::exists(faiss_metadata_jsonpath));

  // ✅ Cleanup
    gst_buffer_unref(buffer);
    gst_object_unref(element);

}
