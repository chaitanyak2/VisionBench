// core/gstcoremeta.cpp
#include "gstcoremeta.hpp"

GType gst_core_meta_api_get_type(void) {
    static GType type = 0;
    static const gchar *tags[] = { "core-meta", nullptr };
    if (g_once_init_enter(&type)) {
        GType _type = gst_meta_api_type_register("GstCoreMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }
    return type;
}

const GstMetaInfo* gst_core_meta_get_info(void) {
    static const GstMetaInfo *meta_info = nullptr;
    if (g_once_init_enter(&meta_info)) {
        const GstMetaInfo *mi = gst_meta_register(
            GST_CORE_META_API_TYPE, "GstCoreMeta",
            sizeof(GstCoreMeta),
            [](GstMeta *meta, gpointer params, GstBuffer *buffer) -> gboolean {
                GstCoreMeta *m = (GstCoreMeta*)meta;
                new (&m->core_meta) std::shared_ptr<visionbench::CoreMetadata>();
                return TRUE;
            },
            [](GstMeta *meta, GstBuffer *buffer) {
                GstCoreMeta *m = (GstCoreMeta*)meta;
                m->core_meta.~shared_ptr();
            },
            (GstMetaTransformFunction)NULL
        );
        g_once_init_leave(&meta_info, mi);
    }
    return meta_info;
}
