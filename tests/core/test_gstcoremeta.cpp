// tests/test_gstcoremeta.cpp
#include <gtest/gtest.h>
#include "gstcoremeta.hpp"
#include "metadata.hpp"
#include "test_fixture.hpp"

TEST_F(VisionBenchFixture, GstCoreMeta_RegisterAndAttach) {
    
    GstBuffer *buffer = gst_buffer_new();
    const GstMetaInfo *info = gst_core_meta_get_info();
    ASSERT_NE(info, nullptr);

    GstCoreMeta *meta = (GstCoreMeta*)gst_buffer_add_meta(buffer, info, nullptr);
    ASSERT_NE(meta, nullptr);
    meta->core_meta = std::make_shared<visionbench::CoreMetadata>();
    meta->core_meta->metadata_id = 42;

    GstCoreMeta *retrieved = gst_buffer_get_core_meta(buffer);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->core_meta->metadata_id, 42u);

    gst_buffer_unref(buffer);
}
