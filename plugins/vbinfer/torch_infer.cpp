#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <torch/script.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"

using namespace visionbench;

#define GST_TYPE_TORCHINFER (gst_torchinfer_get_type())
G_DECLARE_FINAL_TYPE(GstTorchInfer, gst_torchinfer, GST, TORCHINFER, GstVideoFilter)

enum { PROP_0, PROP_MODEL_PATH };

struct _GstTorchInfer {
    GstVideoFilter parent;
    std::shared_ptr<torch::jit::script::Module> module;
    gchar *model_path;
    bool model_loaded;
    int in_c;
    int in_h;
    int in_w;
};

G_DEFINE_TYPE(GstTorchInfer, gst_torchinfer, GST_TYPE_VIDEO_FILTER)

// ---------------------------------------------------------------------------
// Property handling
// ---------------------------------------------------------------------------

static void gst_torchinfer_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec) {
    GstTorchInfer *self = GST_TORCHINFER(object);
    if (prop_id == PROP_MODEL_PATH) {
        g_free(self->model_path);
        self->model_path = g_value_dup_string(value);
    }
}

static void gst_torchinfer_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec) {
    GstTorchInfer *self = GST_TORCHINFER(object);
    if (prop_id == PROP_MODEL_PATH)
        g_value_set_string(value, self->model_path);
}

// ---------------------------------------------------------------------------
// State handling
// ---------------------------------------------------------------------------

static gboolean gst_torchinfer_start(GstBaseTransform *trans) {
    GstTorchInfer *self = GST_TORCHINFER(trans);
    try {
        self->module = std::make_shared<torch::jit::script::Module>(torch::jit::load(self->model_path, torch::kCPU));
        self->module->eval();
        self->model_loaded = true;
        g_print("[TorchInfer] Loaded model: %s\n", self->model_path);
    } catch (const c10::Error &e) {
        g_printerr("[TorchInfer] Failed to load model %s: %s\n", self->model_path, e.what());
        self->model_loaded = false;
    }
    return TRUE;
}

static gboolean gst_torchinfer_stop(GstBaseTransform *trans) {
    GstTorchInfer *self = GST_TORCHINFER(trans);
    self->module.reset();
    self->model_loaded = false;
    return TRUE;
}

// ---------------------------------------------------------------------------
// Core processing
// ---------------------------------------------------------------------------

extern "C" GstFlowReturn gst_torchinfer_transform_frame(GstVideoFilter *filter, GstVideoFrame *inframe, GstVideoFrame *outframe) {
    GstTorchInfer *self = GST_TORCHINFER(filter);
    if (!self->model_loaded) {
        g_printerr("[TorchInfer] Model not loaded.\n");
        return GST_FLOW_ERROR;
    }

    GstBuffer *buf = gst_buffer_ref(inframe->buffer);

    // Retrieve or create CoreMetadata
    GstCoreMeta *meta = gst_buffer_get_core_meta(buf);
    std::shared_ptr<CoreMetadata> core_meta;
    if (meta && meta->core_meta)
        core_meta = meta->core_meta;
    else {
        meta = (GstCoreMeta*)gst_buffer_add_meta(buf, GST_CORE_META_INFO, NULL);
        meta->core_meta = std::make_shared<CoreMetadata>();
        core_meta = meta->core_meta;
    }

    core_meta->stage = Stage::INFERENCE;
    core_meta->stage_status = StageStatus::QUEUED;
    core_meta->timestamp_unix_ms = static_cast<uint64_t>(g_get_real_time() / 1000);

    // Retrieve last memory block (preprocessed tensor)
    guint nmem = gst_buffer_n_memory(buf);
    if (nmem == 0) {
        g_printerr("[TorchInfer] No preprocessed tensor found on buffer\n");
        core_meta->stage_status = StageStatus::FAILED;
        return GST_FLOW_ERROR;
    }

    GstMemory *mem = gst_buffer_peek_memory(buf, nmem - 1);
    GstMapInfo info;
    if (!gst_memory_map(mem, &info, GST_MAP_READ)) {
        core_meta->stage_status = StageStatus::FAILED;
        return GST_FLOW_ERROR;
    }

    // Read tensor dimensions from metadata if available
    int C = 3, H = 224, W = 224;
    if (core_meta->preprocess_meta) {
        W = core_meta->preprocess_meta->width;
        H = core_meta->preprocess_meta->height;
        C = core_meta->preprocess_meta->channels;
    }

    size_t expected_bytes = (size_t)C * H * W * sizeof(float);
    if (info.size < expected_bytes) {
        g_printerr("[TorchInfer] Tensor size mismatch: expected %zu, got %zu\n", expected_bytes, info.size);
        core_meta->stage_status = StageStatus::FAILED;
        gst_memory_unmap(mem, &info);
        return GST_FLOW_ERROR;
    }

    float *data = reinterpret_cast<float*>(info.data);
    torch::Tensor input = torch::from_blob(data, {1, C, H, W}, torch::kFloat32).clone();

    auto start = std::chrono::steady_clock::now();
    torch::Tensor output;
    try {
        std::vector<torch::jit::IValue> inputs{input};
        auto result = self->module->forward(inputs);
        output = result.toTensor();
    } catch (const std::exception &e) {
        g_printerr("[TorchInfer] Inference exception: %s\n", e.what());
        core_meta->stage_status = StageStatus::FAILED;
        gst_memory_unmap(mem, &info);
        return GST_FLOW_ERROR;
    }

    auto end = std::chrono::steady_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Postprocess: extract top-1 prediction
    torch::Tensor probs = torch::softmax(output, 1);
    auto topk = probs.topk(1);
    torch::Tensor values  = std::get<0>(topk);
    torch::Tensor indices = std::get<1>(topk);

    int64_t cls  = indices.item<int64_t>();
    float conf   = values.item<float>();

    // Update metadata with inference results
    core_meta->inference_meta = std::make_shared<InferenceMeta>();
    core_meta->inference_meta->embedding_id = core_meta->metadata_id;
    core_meta->inference_meta->model_name = self->model_path ? self->model_path : "unknown";
    core_meta->inference_meta->inference_latency_ms = latency_ms;
    core_meta->inference_meta->device = DeviceType::CPU;
    core_meta->inference_meta->embedding_dim = output.size(1);

    // Set statuses
    core_meta->stage_status = StageStatus::DONE;
    core_meta->overall_status = OverallStatus::IN_PROGRESS;

    // Attach result to bus
    GstElement *element = GST_ELEMENT(filter);
    GstBus *bus = gst_element_get_bus(element);
    gchar *msg = g_strdup_printf("class=%lld conf=%.3f latency=%.2fms",
                                 (long long)cls, conf, latency_ms);
    GstMessage *m = gst_message_new_application(GST_OBJECT(element),
                      gst_structure_new("torchinfer-result", "result", G_TYPE_STRING, msg, NULL));
    gst_bus_post(bus, m);
    gst_object_unref(bus);
    g_free(msg);

    gst_memory_unmap(mem, &info);
    return GST_FLOW_OK;
}

// ---------------------------------------------------------------------------
// Class boilerplate
// ---------------------------------------------------------------------------

static void gst_torchinfer_class_init(GstTorchInferClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    gobject_class->set_property = gst_torchinfer_set_property;
    gobject_class->get_property = gst_torchinfer_get_property;

    g_object_class_install_property(
        gobject_class, PROP_MODEL_PATH,
        g_param_spec_string("model-path", "Model Path",
                            "Path to TorchScript model file",
                            "model.pt",
                            static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    GstBaseTransformClass *btc = GST_BASE_TRANSFORM_CLASS(klass);
    btc->start = GST_DEBUG_FUNCPTR(gst_torchinfer_start);
    btc->stop = GST_DEBUG_FUNCPTR(gst_torchinfer_stop);

    GstVideoFilterClass *vfc = GST_VIDEO_FILTER_CLASS(klass);
    vfc->transform_frame = GST_DEBUG_FUNCPTR(gst_torchinfer_transform_frame);
}

static void gst_torchinfer_init(GstTorchInfer *self) {
    self->model_path = g_strdup("model.pt");
    self->model_loaded = false;
    self->in_c = 3;
    self->in_h = 224;
    self->in_w = 224;
}

// Plugin initialization function
static gboolean plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "torchinfer", GST_RANK_NONE, GST_TYPE_TORCHINFER);
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

