#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <mutex>
#include <string>
#include "faiss_storage.hpp"

G_BEGIN_DECLS

#define GST_TYPE_FAISS_STORE (gst_faiss_store_get_type())
G_DECLARE_FINAL_TYPE(GstFaissStore, gst_faiss_store, GST, FAISS_STORE, GstBaseSink)

/**
 * GstFaissStore:
 *
 * GStreamer sink element that writes embeddings + metadata to FaissStorage.
 *
 * Properties:
 *  - index-path (string): file path to save FAISS index on finalize (default: "faiss.index")
 *  - metadata-path (string): file path to save metadata JSON on finalize (default: "faiss_metadata.json")
 *  - export-on-eos (boolean): whether to save on EOS (default: TRUE)
 */
struct _GstFaissStore {
    GstBaseSink parent;

    /* properties */
    gchar *index_path;
    gchar *metadata_path;
    gboolean export_on_eos;

    /* internal */
    std::mutex lock;
    bool initialized;
    visionbench::FaissStorage storage;
};

GType gst_faiss_store_get_type(void);

G_END_DECLS
