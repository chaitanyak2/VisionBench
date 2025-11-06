//gstfaissstore.cpp
#include "gstcoremeta.hpp"
#include "metadata.hpp"// definitions for GstCoreMeta / CoreMetadata
#include "faiss_storage.hpp"     // your FaissStorage class header (adjust path)
#include "gstfaissstore.hpp"
#include <gst/video/gstvideometa.h>

using visionbench::FaissStorage;
using visionbench::CoreMetadata;

extern "C" gboolean gst_faiss_store_plugin_init(GstPlugin *plugin);

typedef struct _GstFaissStorePrivate {
    // optional internal members — you can extend later
    int embedding_dim;
} GstFaissStorePrivate;


extern GType gst_core_meta_api_get_type(void);
#define GST_CORE_META_API_TYPE (gst_core_meta_api_get_type())
extern const GstMetaInfo *GST_CORE_META_INFO;

#ifndef PACKAGE
#define PACKAGE "visionbench"
#endif

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY
);

GST_DEBUG_CATEGORY_STATIC(faiss_store_debug);
#define GST_CAT_DEFAULT faiss_store_debug



/* Forward declarations */
//static gboolean gst_faiss_store_start(GstBaseSink *sink);
//static gboolean gst_faiss_store_stop(GstBaseSink *sink);
static void gst_faiss_store_finalize(GObject *object);

G_DEFINE_TYPE_WITH_CODE(GstFaissStore, gst_faiss_store, GST_TYPE_BASE_SINK, G_ADD_PRIVATE(GstFaissStore)
                        GST_DEBUG_CATEGORY_INIT(faiss_store_debug, "faissstore", 0, "FaissStore element"));

/* GObject property IDs */
enum {
    PROP_0,
    PROP_INDEX_PATH,
    PROP_METADATA_PATH,
    PROP_EXPORT_ON_EOS
};

extern "C" gboolean gst_faiss_store_start(GstBaseSink *basesink) {
     GstFaissStore *self = GST_FAISS_STORE(basesink);
    GST_DEBUG_OBJECT(self, "starting faissstore");
std::lock_guard<std::mutex> lk(self->lock);
    try {
        
        self->storage = FaissStorage(576, "Flat", self->index_path);  
        self->initialized = true;
        GST_DEBUG_OBJECT(self, "started faisstore");
    } catch (const std::exception &e) {
        GST_ERROR_OBJECT(self, "Failed to initialize FaissStorage: %s", e.what());
        return FALSE;
        
    }
    return TRUE;
}


static void gst_faiss_store_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec) {
    GstFaissStore *self = GST_FAISS_STORE(object);
    switch (prop_id) {
        case PROP_INDEX_PATH:
            g_free(self->index_path);
            self->index_path = g_value_dup_string(value);
            break;
        case PROP_METADATA_PATH:
            g_free(self->metadata_path);
            self->metadata_path = g_value_dup_string(value);
            break;
        case PROP_EXPORT_ON_EOS:
            self->export_on_eos = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}

static void gst_faiss_store_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec) {
    GstFaissStore *self = GST_FAISS_STORE(object);
    switch (prop_id) {
        case PROP_INDEX_PATH:
            g_value_set_string(value, self->index_path);
            break;
        case PROP_METADATA_PATH:
            g_value_set_string(value, self->metadata_path);
            break;
        case PROP_EXPORT_ON_EOS:
            g_value_set_boolean(value, self->export_on_eos);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}


/* render: called for each incoming buffer */
extern "C" GstFlowReturn gst_faiss_store_render(GstBaseSink *basesink, GstBuffer *buffer) {
    GstFaissStore *self = GST_FAISS_STORE(basesink);
    
    GST_INFO_OBJECT(self, "Entered render()");

    if (!self->initialized) {
        GST_WARNING_OBJECT(self, "faissstore not initialized");
        return GST_FLOW_ERROR;
    }

    // Make a safe local reference to buffer
    GstBuffer *buf = gst_buffer_ref(buffer);

    // Extract CoreMeta
    GstCoreMeta *coremeta = (GstCoreMeta *)gst_buffer_get_meta(buf, GST_CORE_META_API_TYPE);
    if (!coremeta || !coremeta->core_meta) {
        GST_WARNING_OBJECT(self, "No CoreMeta found in buffer; skipping");
        gst_buffer_unref(buf);
        return GST_FLOW_OK;
    }

    // Try to find embedding memory: simplest approach expects a single contiguous GstMemory appended
    // that contains float32 embedding. Adapt if your pipeline uses different layout.
    guint64 embed_size_bytes = 0;
    float *embedding_ptr = nullptr;
    size_t embedding_dim = 0;

    // Look for a memory block that is writable or readable and big enough (heuristic)
   for (guint i = 0; i < gst_buffer_n_memory(buf); ++i) {
    GstMemory *mem = gst_buffer_peek_memory(buf, i);
    if (!mem) continue;
    gsize sz = gst_memory_get_sizes(mem, nullptr, nullptr);
    if (sz == 0) continue;

    // Try map memory read-only
    GstMapInfo map;
    if (!gst_memory_map(mem, &map, GST_MAP_READ)) continue;

    // Heuristic 1: if size is multiple of sizeof(float), treat as float32 array
    if (sz % sizeof(float) == 0 && (sz / sizeof(float)) > 0) {
        size_t dim = sz / sizeof(float);
        // safe copy into vector<float>
        const float *fptr = reinterpret_cast<const float*>(map.data);
        std::vector<float> emb_vec(fptr, fptr + dim);
        gst_memory_unmap(mem, &map);

        try {
            std::lock_guard<std::mutex> lk(self->lock);
            GST_INFO_OBJECT(self, "Going to add embeddings 1");
            
            self->storage.add(emb_vec,
                              coremeta->core_meta->image_location,
                              *coremeta->core_meta);
        } catch (const std::exception &e) {
            GST_ERROR_OBJECT(self, "FaissStorage add failed: %s", e.what());
            gst_buffer_unref(buf);
            return GST_FLOW_ERROR;
        }
        gst_buffer_unref(buf);
        return GST_FLOW_OK;
    }

    // Heuristic 2: if not float-aligned, maybe it's uint8 bytes (e.g., image bytes).
    // Convert to float vector (normalized) and treat as embedding if length reasonable.
    // We only do this if the data size is not tiny and seems plausible.
    if (sz >= 1) {
        // Create uint8 tensor then convert to float
        try {
            // copy bytes into vector<uint8_t> to avoid lifetime issues
            std::vector<uint8_t> tmp((uint8_t*)map.data, (uint8_t*)map.data + sz);
            gst_memory_unmap(mem, &map);

            // Convert to float vector (normalize to [0,1]) — only if size is sane
            std::vector<float> emb_vec;
            emb_vec.reserve(sz);
            for (size_t j = 0; j < tmp.size(); ++j)
                emb_vec.push_back(static_cast<float>(tmp[j]) / 255.0f);

            // Now call FaissStorage (only if emb_vec length >= 1)
            try {
                GST_INFO_OBJECT(self, "Trying to add embedding to store 2");
                std::lock_guard<std::mutex> lk(self->lock);
                self->storage.add(emb_vec,
                                  coremeta->core_meta->image_location,
                                  *coremeta->core_meta);
            } catch (const std::exception &e) {
                GST_ERROR_OBJECT(self, "FaissStorage add failed (from bytes): %s", e.what());
                gst_buffer_unref(buf);
                return GST_FLOW_ERROR;
            }
            gst_buffer_unref(buf);
            return GST_FLOW_OK;
        } catch (...) {
            // fallback: unmap and continue to next mem
            gst_memory_unmap(mem, &map); // if not already unmapped
            continue;
        }
    }

    // if we reach here, unmap and try next memory
    gst_memory_unmap(mem, &map);
}


    GST_WARNING_OBJECT(self, "No embedding memory found in buffer");
    gst_buffer_unref(buf);
    return GST_FLOW_OK;
}

/* stop: called when element stops */
extern "C" gboolean gst_faiss_store_stop(GstBaseSink *basesink) {
    GstFaissStore *self = GST_FAISS_STORE(basesink);
    GST_DEBUG_OBJECT(self, "stopping faissstore");
    std::lock_guard<std::mutex> lk(self->lock);
    if (self->export_on_eos) {
        // flush/save
        try {
            // adjust to your API
            self->storage.saveIndex(self->index_path);
            self->storage.exportMetadata(self->metadata_path);
        } catch (...) {
            GST_WARNING_OBJECT(self, "Failed to save/export Faiss data on stop");
        }
    }
    self->initialized = false;
    return TRUE;
}

static void gst_faiss_store_class_init(GstFaissStoreClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = gst_faiss_store_set_property;
    gobject_class->get_property = gst_faiss_store_get_property;
    gobject_class->finalize = gst_faiss_store_finalize;

    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
                                       gst_static_pad_template_get(&sink_template));

    base_sink_class->start = gst_faiss_store_start;
    base_sink_class->stop = gst_faiss_store_stop;
    base_sink_class->render = gst_faiss_store_render;

    g_object_class_install_property(gobject_class,
        PROP_INDEX_PATH,
        g_param_spec_string("index-path", "Index file path",
                            "Path to save FAISS index on finalize", "faiss.index",
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class,
        PROP_METADATA_PATH,
        g_param_spec_string("metadata-path", "Metadata JSON path",
                            "Path to write metadata JSON on finalize", "faiss_metadata.json",
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class,
        PROP_EXPORT_ON_EOS,
        g_param_spec_boolean("export-on-eos", "Export on EOS",
                             "If true, save index and metadata when EOS is received", TRUE,
                             (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
        "FaissStore", "Sink/Storage",
        "Store embeddings + metadata to a FAISS index",
        "VisionBench");
}

static void gst_faiss_store_init(GstFaissStore *self) {
    self->index_path = g_strdup("faiss.index");
    self->metadata_path = g_strdup("faiss_metadata.json");
    self->export_on_eos = TRUE;
    self->initialized = false;
}


static void gst_faiss_store_finalize(GObject *object) {
    GstFaissStore *self = GST_FAISS_STORE(object);
    // ensure resources freed
    g_free(self->index_path);
    g_free(self->metadata_path);
    G_OBJECT_CLASS(gst_faiss_store_parent_class)->finalize(object);
}

/* plugin init */
extern "C" gboolean gst_faiss_store_plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "faissstore", GST_RANK_NONE, GST_TYPE_FAISS_STORE);
}



/* export plugin */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    faissstore,
    "FAISS storage sink plugin",
    gst_faiss_store_plugin_init,
    "1.0",
    "LGPL",
    PACKAGE,
    "https://github.com/chaitanya.k2/VisionBench"
)