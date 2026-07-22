#ifndef EMBERWATCH_FEATURES_H
#define EMBERWATCH_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#define EW_FEATURE_COUNT 6u
#define EW_FEATURE_HISTORY_SIZE 7u

typedef struct {
    float asset_temp_c[EW_FEATURE_HISTORY_SIZE];
    uint8_t write_index;
    uint8_t count;
} ew_feature_history_t;

void ew_features_reset(ew_feature_history_t *history);

/* Call every five minutes for the current model. Returns false until history is ready. */
bool ew_features_update(
    ew_feature_history_t *history,
    float asset_temp_c,
    float ambient_temp_c,
    float output[EW_FEATURE_COUNT]
);

#endif
