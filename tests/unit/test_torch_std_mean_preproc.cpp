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
    gboolean plugin_init(GstPlugin *plugin);

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


TEST(TorchPreproc, BasicTransform)
{
    gst_init(nullptr, nullptr);

    gboolean registered = gst_plugin_register_static(
        GST_VERSION_MAJOR, GST_VERSION_MINOR,
        "torchpreproc", "Torch preprocessor plugin",
        plugin_init, "1.0", "LGPL",
        "visionbench", "visionbench",
        "https://github.com/chaitanya.k2/VisionBench");
    ASSERT_TRUE(registered);

    GstElement *element = gst_element_factory_make("torchpreproc", nullptr);
    ASSERT_NE(element, nullptr);
    GstVideoFilter *filter = GST_VIDEO_FILTER(element);

    const int W = 64, H = 64;
    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, GST_VIDEO_FORMAT_RGB, W, H);

    /* ---------- Input buffer ---------- */
    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8);
    GstBuffer *in_buf = gst_buffer_new_and_alloc(W * H * 3);
    GstMapInfo in_map;
    gst_buffer_map(in_buf, &in_map, GST_MAP_WRITE);
    memcpy(in_map.data, img.data_ptr<uint8_t>(), W * H * 3);
    gst_buffer_unmap(in_buf, &in_map);

    /* ---------- Output buffer ---------- */
    GstBuffer *out_buf = gst_buffer_new_and_alloc(W * H * 3);

    /* 🔹 force writable before *anything* touches it */
    out_buf = gst_buffer_make_writable(out_buf);
    g_assert(gst_buffer_is_writable(out_buf));

    GstVideoFrame inframe, outframe;
    ASSERT_TRUE(gst_video_frame_map(&inframe, &vinfo, in_buf, GST_MAP_READ));
    ASSERT_TRUE(gst_video_frame_map(&outframe, &vinfo, out_buf, GST_MAP_WRITE));

    /* 🔹 Pass exactly that writable buffer pointer to the plugin */
    outframe.buffer = out_buf;

    GstFlowReturn ret = gst_torchpreproc_transform_frame(filter, &inframe, &outframe);
    EXPECT_EQ(ret, GST_FLOW_OK);

    gst_video_frame_unmap(&inframe);
    gst_video_frame_unmap(&outframe);
    gst_buffer_unref(in_buf);
    gst_buffer_unref(out_buf);
    gst_object_unref(element);
}
