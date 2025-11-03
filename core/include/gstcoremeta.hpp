// core/gstcoremeta.hpp
#pragma once
#include <gst/gst.h>
#include "metadata.hpp"

G_BEGIN_DECLS

typedef struct _GstCoreMeta {
    GstMeta meta;
    std::shared_ptr<visionbench::CoreMetadata> core_meta;
} GstCoreMeta;

GType gst_core_meta_api_get_type(void);
const GstMetaInfo* gst_core_meta_get_info(void);

#define GST_CORE_META_API_TYPE (gst_core_meta_api_get_type())
#define GST_CORE_META_INFO (gst_core_meta_get_info())
#define gst_buffer_get_core_meta(b) ((GstCoreMeta*)gst_buffer_get_meta((b), GST_CORE_META_API_TYPE))

G_END_DECLS
