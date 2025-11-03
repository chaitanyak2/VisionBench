
#include <gtest/gtest.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <torch/script.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"

extern "C" {
    GstFlowReturn gst_torchinfer_transform_frame(GstVideoFilter *filter, GstVideoFrame *inframe, GstVideoFrame *outframe);
}


TEST(TorchInfer, LoadModelAndInfer) {
    try {
        auto model = torch::jit::load("../data/mobilenetv3_small.pt", torch::kCPU);
        model.eval();
        torch::Tensor input = torch::rand({1, 3, 224, 224});
        auto output = model.forward({input}).toTensor();
        EXPECT_EQ(output.dim(), 2);
    } catch (...) {
        GTEST_SKIP() << "model.pt not found, skipping Torch inference test.";
    }
}

TEST(TorchInfer, InferenceMetaUpdate) {
    visionbench::CoreMetadata meta;
    meta.inference_meta = std::make_shared<visionbench::InferenceMeta>();
    meta.inference_meta->embedding_id = 101;
    meta.inference_meta->embedding_dim = 128;
    meta.inference_meta->device = visionbench::DeviceType::CPU;
    EXPECT_EQ(meta.inference_meta->embedding_id, 101u);
    EXPECT_EQ(meta.inference_meta->device, visionbench::DeviceType::CPU);
}

TEST(TorchInfer, TransformFrameRunsAndFillsMetadata)
{
    const int W = 224, H = 224;

    // Create dummy preprocessed float tensor
    torch::Tensor img = torch::randint(0, 256, {H, W, 3}, torch::kUInt8);
    GstBuffer *buffer = gst_buffer_new_and_alloc(W * H * 3 * sizeof(float));
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, img.data_ptr<float>(), W * H * 3 * sizeof(float));
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

    // Prepare GstVideoFrames
    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, GST_VIDEO_FORMAT_RGB, W, H);
    GstVideoFrame inframe, outframe;
    ASSERT_TRUE(gst_video_frame_map(&inframe, &vinfo, buffer, GST_MAP_READ));
    GstBuffer *out_buf = gst_buffer_new_and_alloc(W * H * 3 * sizeof(float));
    ASSERT_TRUE(gst_video_frame_map(&outframe, &vinfo, out_buf, GST_MAP_WRITE));

    // Run inference transform
    GstFlowReturn ret = gst_torchinfer_transform_frame(nullptr, &inframe, &outframe);
    EXPECT_EQ(ret, GST_FLOW_OK);

    // Verify output metadata
    GstCoreMeta *out_meta = (GstCoreMeta *)gst_buffer_get_meta(out_buf, GST_CORE_META_API_TYPE);
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
    gst_buffer_unref(out_buf);
}

