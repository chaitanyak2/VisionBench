// plugins/torchpreproc/gsttorchpreproc.cpp
#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <torch/torch.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"

using namespace visionbench;

#define GST_TYPE_TORCHPREPROC (gst_torchpreproc_get_type())
G_DECLARE_FINAL_TYPE(GstTorchPreproc, gst_torchpreproc, GST, TORCHPREPROC, GstVideoFilter)

struct _GstTorchPreproc {
    GstVideoFilter parent;
    int out_w;
    int out_h;
    bool input_bgr;
};

G_DEFINE_TYPE(GstTorchPreproc, gst_torchpreproc, GST_TYPE_VIDEO_FILTER)

static gboolean gst_torchpreproc_set_info(GstVideoFilter *filter,
                                          GstCaps *incaps,
                                          GstVideoInfo *in_info,
                                          GstCaps *outcaps,
                                          GstVideoInfo *out_info) {
    GstTorchPreproc *self = GST_TORCHPREPROC(filter);
    self->out_w = 224;
    self->out_h = 224;
    GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(in_info);
    self->input_bgr = (fmt == GST_VIDEO_FORMAT_BGR);
    return TRUE;
}

extern "C" GstFlowReturn gst_torchpreproc_transform_frame(GstVideoFilter *filter,
                                                      GstVideoFrame *inframe,
                                                      GstVideoFrame *outframe) {
    GstTorchPreproc *self = GST_TORCHPREPROC(filter);
    GstBuffer *buf = gst_buffer_ref(inframe->buffer);

    // Get or create CoreMetadata
    GstCoreMeta *meta = gst_buffer_get_core_meta(buf);
    std::shared_ptr<CoreMetadata> core_meta;
    if (meta && meta->core_meta)
        core_meta = meta->core_meta;
    else {
        meta = (GstCoreMeta*)gst_buffer_add_meta(buf, GST_CORE_META_INFO, NULL);
        meta->core_meta = std::make_shared<CoreMetadata>();
        core_meta = meta->core_meta;
        core_meta->stage = Stage::READ;
    }

    // Update metadata for preprocessing
    core_meta->stage = Stage::PREPROCESS;
    core_meta->stage_status = StageStatus::QUEUED;
    core_meta->timestamp_unix_ms =
        static_cast<uint64_t>(g_get_real_time() / 1000);

    uint32_t width  = GST_VIDEO_FRAME_WIDTH(inframe);
    uint32_t height = GST_VIDEO_FRAME_HEIGHT(inframe);
    core_meta->image_width  = width;
    core_meta->image_height = height;
    core_meta->channels = ChannelType::RGB;

    // Wrap into torch tensor
    auto* data = reinterpret_cast<guint8*>(GST_VIDEO_FRAME_PLANE_DATA(inframe, 0));
    torch::Tensor img = torch::from_blob(data, {height, width, 3}, torch::kUInt8)
                            .to(torch::kFloat32)
                            .div_(255.0f);

    if (self->input_bgr)
        img = img.index({"...", torch::tensor({2, 1, 0})});

    img = img.permute({2, 0, 1}).contiguous();  // CHW

    img = torch::nn::functional::interpolate(
              img.unsqueeze(0),
              torch::nn::functional::InterpolateFuncOptions()
                  .size(std::vector<int64_t>{self->out_h, self->out_w})
                  .mode(torch::kBilinear)
                  .align_corners(false))
              .squeeze(0);

    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std  = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
        img[c] = (img[c] - mean[c]) / std[c];

    img = img.contiguous();
    float *buf_data = (float*)g_malloc(img.numel() * sizeof(float));
    memcpy(buf_data, img.data_ptr<float>(), img.numel() * sizeof(float));

    // Append memory
    GstMemory *mem = gst_memory_new_wrapped(
        static_cast<GstMemoryFlags>(0), buf_data, img.numel() * sizeof(float), 0,
        img.numel() * sizeof(float), buf_data, (GDestroyNotify)g_free);
    gst_buffer_append_memory(buf, mem);

    // Fill preprocess_meta
    core_meta->preprocess_meta = std::make_shared<PreprocessMeta>();
    core_meta->preprocess_meta->width = self->out_w;
    core_meta->preprocess_meta->height = self->out_h;
    core_meta->preprocess_meta->channels = 3;

    core_meta->stage_status = StageStatus::DONE;

    return GST_FLOW_OK;
}

static void gst_torchpreproc_class_init(GstTorchPreprocClass *klass) {
    GstVideoFilterClass *vfc = GST_VIDEO_FILTER_CLASS(klass);
    vfc->set_info = GST_DEBUG_FUNCPTR(gst_torchpreproc_set_info);
    vfc->transform_frame = GST_DEBUG_FUNCPTR(gst_torchpreproc_transform_frame);
}

static void gst_torchpreproc_init(GstTorchPreproc *self) {
    self->out_w = 224;
    self->out_h = 224;
    self->input_bgr = false;
}

// Plugin initialization function
static gboolean plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "torchpreproc", GST_RANK_NONE, GST_TYPE_TORCHPREPROC);
}

// Define PACKAGE for non-autotools builds (CMake)
#ifndef PACKAGE
#define PACKAGE "visionbench"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    torchinfer,
    "TorchScript inference with CoreMetadata integration",
    plugin_init,   // <-- correct init function
    "1.0",
    "LGPL",
    PACKAGE,
    "https://github.com/chaitanya.k2/VisionBench"
)

/*GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    torchpreproc,
    "LibTorch preprocessing with CoreMetadata integration",
    gst_element_register_static,
    "1.0",
    "LGPL",
    "gst-torchpreproc",
    "chaitanya.k2@gmail.com"
)
*/