// tests/test_torchpreproc.cpp
#include <gtest/gtest.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <gst/gst.h>
#include <torch/torch.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"

extern "C" {
    GstFlowReturn gst_torchpreproc_transform_frame(GstVideoFilter *filter, GstVideoFrame *inframe, GstVideoFrame *outframe);
}

TEST(TorchPreproc, TensorPreprocessingLibTorch) {
    gst_init(nullptr, nullptr);
    // create dummy RGB image
    int W = 32, H = 32;
    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8);

    // wrap into buffer
    GstBuffer *buffer = gst_buffer_new_and_alloc(W * H * 3);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, img.data_ptr<uint8_t>(), W * H * 3);
    gst_buffer_unmap(buffer, &map);

    // Attach metadata manually
    GstCoreMeta *meta = (GstCoreMeta*)gst_buffer_add_meta(buffer, GST_CORE_META_INFO, nullptr);
    ASSERT_NE(meta, nullptr) << "GST_CORE_META_INFO not registered or gst_init() not called.";

    meta->core_meta = std::make_shared<visionbench::CoreMetadata>();

    // Normally transform_frame uses GstVideoFrame, here we just check we can attach output
    EXPECT_NE(meta->core_meta, nullptr);
    gst_buffer_unref(buffer);
}


TEST(TorchPreproc, PopulatesCoreMetadataCorrectly)
{
    gst_init(nullptr, nullptr);

    const int W = 64, H = 64;

    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8);
    GstBuffer *buffer = gst_buffer_new_and_alloc(W * H * 3);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, img.data_ptr<uint8_t>(), W * H * 3);
    gst_buffer_unmap(buffer, &map);

    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, GST_VIDEO_FORMAT_RGB, W, H);

    GstVideoFrame inframe, outframe;
    ASSERT_TRUE(gst_video_frame_map(&inframe, &vinfo, buffer, GST_MAP_READ));
    GstBuffer *out_buf = gst_buffer_new_and_alloc(W * H * 3);
    ASSERT_TRUE(gst_video_frame_map(&outframe, &vinfo, out_buf, GST_MAP_WRITE));

    // ✅ Create valid plugin instance
    GstElement *element = gst_element_factory_make("torchpreproc", nullptr);
    ASSERT_NE(element, nullptr) << "torchpreproc plugin not registered.";
GstVideoFilter *filter = GST_VIDEO_FILTER(element);

    // ✅ Make sure output buffer is writable
    out_buf = gst_buffer_make_writable(out_buf);

    // ✅ Run transform safely
GstFlowReturn ret = gst_torchpreproc_transform_frame(filter, &inframe, &outframe);
    EXPECT_EQ(ret, GST_FLOW_OK);

    // ✅ Verify metadata
    GstCoreMeta *meta = (GstCoreMeta *)gst_buffer_get_meta(out_buf, GST_CORE_META_API_TYPE);
    ASSERT_NE(meta, nullptr) << "GstCoreMeta not attached to output buffer.";

    auto &core = *meta->core_meta;
    EXPECT_EQ(core.stage, visionbench::Stage::PREPROCESS);
    EXPECT_EQ(core.stage_status, visionbench::StageStatus::DONE);
    EXPECT_EQ(core.image_width, W);
    EXPECT_EQ(core.image_height, H);
    EXPECT_EQ(core.channels, visionbench::ChannelType::RGB);

    ASSERT_NE(core.preprocess_meta, nullptr);
    EXPECT_EQ(core.preprocess_meta->width, W);
    EXPECT_EQ(core.preprocess_meta->height, H);
    EXPECT_EQ(core.preprocess_meta->channels, 3);

    gst_video_frame_unmap(&inframe);
    gst_video_frame_unmap(&outframe);
    gst_buffer_unref(buffer);
    gst_buffer_unref(out_buf);
    gst_object_unref(element);
}
