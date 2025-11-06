// tests/test_metadata.cpp
#include <gtest/gtest.h>
#include "metadata.hpp"
#include "test_fixture.hpp"

using namespace visionbench;

TEST_F(VisionBenchFixture, Metadata_DefaultInitialization) {
    CoreMetadata meta;
    EXPECT_EQ(meta.stage, Stage::READ);
    EXPECT_EQ(meta.stage_status, StageStatus::QUEUED);
    EXPECT_EQ(meta.overall_status, OverallStatus::IN_PROGRESS);
    EXPECT_EQ(meta.channels, ChannelType::RGB);
}

TEST_F(VisionBenchFixture, Metadata_PreprocessMetaAssignment) {
    CoreMetadata meta;
    meta.preprocess_meta = std::make_shared<PreprocessMeta>();
    meta.preprocess_meta->width = 224;
    meta.preprocess_meta->height = 224;
    meta.preprocess_meta->channels = 3;
    EXPECT_EQ(meta.preprocess_meta->width, 224);
    EXPECT_EQ(meta.preprocess_meta->channels, 3);
}

TEST_F(VisionBenchFixture, Metadata_StageTransition) {
    CoreMetadata meta;
    meta.stage = Stage::PREPROCESS;
    EXPECT_EQ(meta.stage, Stage::PREPROCESS);
    meta.stage_status = StageStatus::DONE;
    EXPECT_EQ(meta.stage_status, StageStatus::DONE);
}
