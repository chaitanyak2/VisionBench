
#include <gtest/gtest.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <torch/script.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"
#include "benchmark_recorder.hpp"
#include "test_fixture.hpp"

using namespace visionbench;

extern "C" {
    GstFlowReturn gst_torchinfer_transform_frame(GstVideoFilter *filter, GstVideoFrame *inframe, GstVideoFrame *outframe);
    gboolean gst_torch_infer_plugin_init(GstPlugin *plugin);
}


TEST_F(VisionBenchFixture, TorchInfer_LoadModelAndInfer)
{
    try {
        auto model = torch::jit::load(nn_model_path, torch::kCPU);
        model.eval();
        torch::Tensor input = torch::rand({1, 3, 224, 224});
        auto output = model.forward({input}).toTensor();
        EXPECT_EQ(output.dim(), 2);
    } catch (...) {
        GTEST_SKIP() << nn_model_path<<" not found, skipping Torch inference test.";
    }
}

TEST_F(VisionBenchFixture, TorchInfer_InferenceMetaUpdate) {
    visionbench::CoreMetadata meta;
    meta.inference_meta = std::make_shared<visionbench::InferenceMeta>();
    meta.inference_meta->embedding_id = 101;
    meta.inference_meta->embedding_dim = 128;
    meta.inference_meta->device = visionbench::DeviceType::CPU;
    EXPECT_EQ(meta.inference_meta->embedding_id, 101u);
    EXPECT_EQ(meta.inference_meta->device, visionbench::DeviceType::CPU);
}

TEST_F(VisionBenchFixture, TorchInfer_TransformFrameRunsAndFillsMetadata)
{
    const int W = 224, H = 224;

    // Create dummy preprocessed float tensor
    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8).contiguous();
    GstBuffer *buffer = gst_buffer_new_and_alloc(W * H * 3 );
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, img.data_ptr<uint8_t>(), W * H * 3 );
    gst_buffer_unmap(buffer, &map);

    // Attach preprocessor metadata
    GstCoreMeta *meta = (GstCoreMeta *)gst_buffer_add_meta(buffer, GST_CORE_META_INFO, nullptr);
    meta->core_meta = std::make_shared<visionbench::CoreMetadata>();
    meta->core_meta->stage = visionbench::Stage::PREPROCESS;
    meta->core_meta->stage_status = visionbench::StageStatus::DONE;
    meta->core_meta->preprocess_meta = std::make_shared<visionbench::PreprocessMeta>();
    meta->core_meta->preprocess_meta->width = W;
    meta->core_meta->preprocess_meta->height = H;
    meta->core_meta->preprocess_meta->channels = 3;

    // // Prepare GstVideoFrames
    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, GST_VIDEO_FORMAT_RGB, W, H);
    GstVideoFrame inframe, outframe;
    ASSERT_TRUE(gst_video_frame_map(&inframe, &vinfo, buffer, GST_MAP_READ));
    GstBuffer *out_buf = gst_buffer_new_and_alloc(W * H * 3 );
    ASSERT_TRUE(gst_video_frame_map(&outframe, &vinfo, out_buf, GST_MAP_WRITE));

   ASSERT_NE((void*)gst_torch_infer_plugin_init, nullptr) << "plugin_init pointer is null";


    gboolean registered = gst_plugin_register_static(
        GST_VERSION_MAJOR, GST_VERSION_MINOR,
        "torchinfer", "TorchScript inference with CoreMetadata integration",
    gst_torch_infer_plugin_init,   // <-- correct init function
    "1.0",
    "LGPL",
    "visionbench", "visionbench",
        "https://github.com/chaitanya.k2/VisionBench"
   );
    ASSERT_TRUE(registered);

//     GstElement *element = gst_element_factory_make("torchinfer", nullptr);
//     ASSERT_NE(element, nullptr);
    
//     g_object_set(G_OBJECT(element),
//              "model-path", nn_model_path.c_str(),
//              NULL);
//     gchar *path = nullptr;
    
//  g_object_get(G_OBJECT(element), "model-path", &path, NULL);
//   EXPECT_STREQ(path, nn_model_path.c_str());
//     g_free(path);
//     // //Run inference transform
//     GstVideoFilter *filter = GST_VIDEO_FILTER(element);
//     GstFlowReturn ret = gst_torchinfer_transform_frame(filter, &inframe, &outframe);
//     EXPECT_EQ(ret, GST_FLOW_OK);

//     // Verify output metadata
//     GstCoreMeta *out_meta = (GstCoreMeta *)gst_buffer_get_meta(outframe.buffer, GST_CORE_META_API_TYPE);
//     ASSERT_NE(out_meta, nullptr);

//     auto &core = *out_meta->core_meta;
//     ASSERT_NE(core.inference_meta, nullptr);
//     EXPECT_EQ(core.stage, visionbench::Stage::INFERENCE);
//     EXPECT_EQ(core.stage_status, visionbench::StageStatus::DONE);
//     EXPECT_EQ(core.inference_meta->device, visionbench::DeviceType::CPU);
//     EXPECT_EQ(core.inference_meta->embedding_dim, 576);  // MobileNet
//     EXPECT_FALSE(core.inference_meta->model_name.empty());

//     // Cleanup
//     gst_video_frame_unmap(&inframe);
//     gst_video_frame_unmap(&outframe);
//     gst_buffer_unref(buffer);
}


TEST_F(VisionBenchFixture, TorchInfer_PreprocBenchmarkWritesEntry)
{
    
    const int W = 224, H = 224;

    // Create dummy preprocessed float tensor
    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8).contiguous();
    GstBuffer *buffer = gst_buffer_new_and_alloc(W * H * 3 );
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, img.data_ptr<uint8_t>(), W * H * 3 );
    gst_buffer_unmap(buffer, &map);

    // Attach preprocessor metadata
    GstCoreMeta *meta = (GstCoreMeta *)gst_buffer_add_meta(buffer, GST_CORE_META_INFO, nullptr);
    meta->core_meta = std::make_shared<visionbench::CoreMetadata>();
    meta->core_meta->stage = visionbench::Stage::PREPROCESS;
    meta->core_meta->stage_status = visionbench::StageStatus::DONE;
    meta->core_meta->preprocess_meta = std::make_shared<visionbench::PreprocessMeta>();
    meta->core_meta->preprocess_meta->width = W;
    meta->core_meta->preprocess_meta->height = H;
    meta->core_meta->preprocess_meta->channels = 3;

    // // Prepare GstVideoFrames
    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, GST_VIDEO_FORMAT_RGB, W, H);
    GstVideoFrame inframe, outframe;
    ASSERT_TRUE(gst_video_frame_map(&inframe, &vinfo, buffer, GST_MAP_READ));
    GstBuffer *out_buf = gst_buffer_new_and_alloc(W * H * 3 );
    ASSERT_TRUE(gst_video_frame_map(&outframe, &vinfo, out_buf, GST_MAP_WRITE));

   

    gboolean registered = gst_plugin_register_static(
        GST_VERSION_MAJOR, GST_VERSION_MINOR,
        "torchinfer", "TorchScript inference with CoreMetadata integration",
    gst_torch_infer_plugin_init,   // <-- correct init function
    "1.0",
    "LGPL",
    "visionbench", "visionbench",
        "https://github.com/chaitanya.k2/VisionBench"
   );
    ASSERT_TRUE(registered);

    GstElement *element = gst_element_factory_make("torchinfer", nullptr);
    ASSERT_NE(element, nullptr);
    
    g_object_set(G_OBJECT(element),
             "model-path", nn_model_path.c_str(),
             NULL);
    gchar *path = nullptr;
    
 g_object_get(G_OBJECT(element), "model-path", &path, NULL);
  EXPECT_STREQ(path, nn_model_path.c_str());
    g_free(path);
    // //Run inference transform
    GstVideoFilter *filter = GST_VIDEO_FILTER(element);
    GstFlowReturn ret = gst_torchinfer_transform_frame(filter, &inframe, &outframe);
    EXPECT_EQ(ret, GST_FLOW_OK);

    // Verify output metadata
    GstCoreMeta *out_meta = (GstCoreMeta *)gst_buffer_get_meta(outframe.buffer, GST_CORE_META_API_TYPE);
    ASSERT_NE(out_meta, nullptr);

    auto &core = *out_meta->core_meta;
    ASSERT_NE(core.inference_meta, nullptr);
    EXPECT_EQ(core.stage, visionbench::Stage::INFERENCE);
    EXPECT_EQ(core.stage_status, visionbench::StageStatus::DONE);
    EXPECT_EQ(core.inference_meta->device, visionbench::DeviceType::CPU);
    EXPECT_EQ(core.inference_meta->embedding_dim, 576);  // MobileNet
    EXPECT_FALSE(core.inference_meta->model_name.empty());

    // Cleanup
    gst_video_frame_unmap(&inframe);
    gst_video_frame_unmap(&outframe);
    gst_buffer_unref(buffer);
    
    // ✅ 6. Verify database and benchmark entry
    auto &recorder = BenchmarkRecorder::instance();
    auto entries = recorder.fetch_all();
    ASSERT_FALSE(entries.empty()) << "No benchmark entries recorded!";
    const auto &e = entries.back();
    EXPECT_EQ(e.module_name, "torchinfer");
    EXPECT_NE(e.params_serialized.find("inference_latency"), std::string::npos);
    recorder.shutdown();
}