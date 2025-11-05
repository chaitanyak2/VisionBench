#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <torch/script.h>
#include "metadata.hpp"
#include "gstcoremeta.hpp"
#include "benchmark_recorder.hpp"

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

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw, format=(string)RGB, width=(int)[1,MAX], height=(int)[1,MAX]"));

static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw, format=(string)RGB, width=(int)[1,MAX], height=(int)[1,MAX]"));

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
        if (self->model_path && g_strcmp0(self->model_path, "model.pt") != 0) {
            try {
                self->module = std::make_shared<torch::jit::script::Module>(
                    torch::jit::load(self->model_path, torch::kCPU));
                self->module->eval();
                self->model_loaded = true;
                g_print("[TorchInfer] Auto-loaded model in transform: %s\n", self->model_path);
            } catch (const c10::Error &e) {
                g_printerr("[TorchInfer] Failed to auto-load model %s: %s\n",
                           self->model_path, e.what());
                self->model_loaded = false;
                return GST_FLOW_ERROR;
            }
        }else {
        g_printerr("[TorchInfer] Model not loaded.\n");
        return GST_FLOW_ERROR;
        }
    }

    GstBuffer *outbuf = outframe->buffer;
    // 1) Make writable (copy-on-write if necessary)
    outbuf = gst_buffer_make_writable(outbuf);
 // updat
// 2) Copy any metadata from input to output
gst_buffer_copy_into(outbuf, inframe->buffer, GST_BUFFER_COPY_METADATA, 0, -1);

    GstBuffer *buf = gst_buffer_ref(inframe->buffer);

                     // ---------------------------------------------------------------------------
// Create or update CoreMeta on OUTPUT buffer
// ---------------------------------------------------------------------------
GstCoreMeta *out_meta = gst_buffer_get_core_meta(outframe->buffer);
if (!out_meta) {
    out_meta = (GstCoreMeta*)gst_buffer_add_meta(outframe->buffer, GST_CORE_META_INFO, NULL);
    out_meta->core_meta = std::make_shared<visionbench::CoreMetadata>();

}

// 4) If input had CoreMeta, reuse/share its shared_ptr to preserve preprocess_meta etc.
GstCoreMeta *in_meta = gst_buffer_get_core_meta(inframe->buffer);
if (in_meta && in_meta->core_meta) {
    out_meta->core_meta = in_meta->core_meta; // share pointer
}
// 5) Now update inference fields
auto &core = *out_meta->core_meta;

g_print("[TorchInfer] inframe->buffer=%p outframe->buffer=%p out_buf? (if available)=%p\n",
        (void*)inframe->buffer, (void*)outframe->buffer, (void*)outbuf /*if you have it*/);
g_print("[TorchInfer] out writable? %d in writable? %d\n",
        gst_buffer_is_writable(outframe->buffer),
        gst_buffer_is_writable(inframe->buffer));

    core.stage = Stage::INFERENCE;
    core.stage_status = StageStatus::QUEUED;
    core.timestamp_unix_ms = static_cast<uint64_t>(g_get_real_time() / 1000);

    // Retrieve last memory block (preprocessed tensor)
    guint nmem = gst_buffer_n_memory(buf);
    if (nmem == 0) {
        g_printerr("[TorchInfer] No preprocessed tensor found on buffer\n");
        core.stage_status = StageStatus::FAILED;
        return GST_FLOW_ERROR;
    }

    GstMemory *mem = gst_buffer_peek_memory(buf, nmem - 1);
    GstMapInfo info;
    if (!gst_memory_map(mem, &info, GST_MAP_READ)) {
        core.stage_status = StageStatus::FAILED;
        return GST_FLOW_ERROR;
    }

    // Read tensor dimensions from metadata if available
    int C = 3, H = 224, W = 224;
    if (core.preprocess_meta) {
        W = core.preprocess_meta->width;
        H = core.preprocess_meta->height;
        C = core.preprocess_meta->channels;
    }

    size_t expected_bytes = (size_t)C * H * W ;
    if (info.size < expected_bytes) {
        g_printerr("[TorchInfer] Tensor size mismatch: expected %zu, got %zu\n", expected_bytes, info.size);
        core.stage_status = StageStatus::FAILED;
        gst_memory_unmap(mem, &info);
        return GST_FLOW_ERROR;
    }

    uint8_t *data = reinterpret_cast<uint8_t*>(info.data);
    torch::Tensor input = torch::from_blob(data, {H, W, C}, torch::kUInt8).to(torch::kFloat32).permute({2,0,1}).unsqueeze(0).clone();

    auto start = std::chrono::steady_clock::now();
    torch::Tensor output;
    try {
        std::vector<torch::jit::IValue> inputs{input};
        auto result = self->module->forward(inputs);
        output = result.toTensor();
    } catch (const std::exception &e) {
        g_printerr("[TorchInfer] Inference exception: %s\n", e.what());
        core.stage_status = StageStatus::FAILED;
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
    core.inference_meta = std::make_shared<InferenceMeta>();
    core.inference_meta->embedding_id = in_meta->core_meta->metadata_id;
    core.inference_meta->model_name = self->model_path ? self->model_path : "unknown";
    core.inference_meta->inference_latency_ms = latency_ms;
    core.inference_meta->device = DeviceType::CPU;
    core.inference_meta->embedding_dim = output.size(1);

    // Set statuses
    core.stage_status = StageStatus::DONE;
    core.overall_status = OverallStatus::IN_PROGRESS;
    //outframe->buffer = outbuf;
    // Attach result to bus
    GstElement *element = GST_ELEMENT(filter);
    GstBus *bus = gst_element_get_bus(element);

    if (bus) {
    gchar *msg = g_strdup_printf("class=%lld conf=%.3f latency=%.2fms",
                                 (long long)cls, conf, latency_ms);
    GstMessage *m = gst_message_new_application(GST_OBJECT(element),
                      gst_structure_new("torchinfer-result",
                                        "result", G_TYPE_STRING, msg, NULL));
    gst_bus_post(bus, m);
    gst_object_unref(bus);
    g_free(msg);
} else {
    // No bus (probably running in a unit test)
    g_print("[TorchInfer] No pipeline bus, skipping message post\n");
}

    gst_memory_unmap(mem, &info);

      uint64_t end_time_ms = g_get_real_time() / 1000;
     visionbench::BenchmarkEntry entry;
 entry.metadata_id = (out_meta && out_meta->core_meta) ?core.metadata_id : 0;
 entry.module_name = "torchinfer";
 entry.params_serialized = 
     "{" 
     "\"model_name\":" + core.inference_meta->model_name + "," +
     "\"inference_latency\":" + std::to_string(latency_ms) + "," +
     "}";
 entry.timestamp_unix_ms = end_time_ms;
 visionbench::BenchmarkRecorder::instance().record(entry);

    return GST_FLOW_OK;
}

// ---------------------------------------------------------------------------
// Class boilerplate
// ---------------------------------------------------------------------------

static void gst_torchinfer_class_init(GstTorchInferClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    gobject_class->set_property = gst_torchinfer_set_property;
    gobject_class->get_property = gst_torchinfer_get_property;

    g_object_class_install_property(
        gobject_class, PROP_MODEL_PATH,
        g_param_spec_string("model-path", "Model Path",
                            "Path to TorchScript model file",
                            "model.pt",
                            static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

                            // metadata - required
    gst_element_class_set_static_metadata(
        element_class,
        "Torch Inference",
        "Filter/Effect/Video",
        "Runs TorchScript model inference and attaches CoreMetadata",
        "Chaitanya Khire <you@example.com>");

    GstBaseTransformClass *btc = GST_BASE_TRANSFORM_CLASS(klass);
    btc->start = GST_DEBUG_FUNCPTR(gst_torchinfer_start);
    btc->stop = GST_DEBUG_FUNCPTR(gst_torchinfer_stop);

    GstVideoFilterClass *vfc = GST_VIDEO_FILTER_CLASS(klass);
    vfc->transform_frame = GST_DEBUG_FUNCPTR(gst_torchinfer_transform_frame);
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_template));
}

static void gst_torchinfer_init(GstTorchInfer *self) {
    self->model_path = g_strdup("model.pt");
    self->model_loaded = false;
    self->in_c = 3;
    self->in_h = 224;
    self->in_w = 224;
}

// Plugin initialization function
 extern "C" gboolean plugin_init(GstPlugin *plugin)
{
    static bool initialized = false;
    if (!initialized) {
        
        if (!visionbench::BenchmarkRecorder::instance().init("/tmp/visionbench_benchmarks.db")) {
            g_printerr("Failed to initialize BenchmarkRecorder database\n");
        }
        initialized = true;
    }
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
