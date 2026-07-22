#include "emberwatch_features.h"

#include <string.h>

static float sample_ago(const ew_feature_history_t *history, uint8_t samples_ago) {
    uint8_t index = (uint8_t)((history->write_index + EW_FEATURE_HISTORY_SIZE - 1u - samples_ago) % EW_FEATURE_HISTORY_SIZE);
    return history->asset_temp_c[index];
}

void ew_features_reset(ew_feature_history_t *history) {
    memset(history, 0, sizeof(*history));
}

bool ew_features_update(
    ew_feature_history_t *history,
    float asset_temp_c,
    float ambient_temp_c,
    float output[EW_FEATURE_COUNT]
) {
    float mean = 0.0f;
    float variance = 0.0f;
    uint8_t index;
    uint8_t variance_count;

    history->asset_temp_c[history->write_index] = asset_temp_c;
    history->write_index = (uint8_t)((history->write_index + 1u) % EW_FEATURE_HISTORY_SIZE);
    if (history->count < EW_FEATURE_HISTORY_SIZE) history->count++;
    if (history->count < 4u) return false;

    output[0] = ambient_temp_c;
    output[1] = asset_temp_c - ambient_temp_c;
    /* The training source has a five-minute cadence, so its 1-minute lookup resolves to
       the previous five-minute observation. Preserve that behavior for this model. */
    output[2] = (asset_temp_c - sample_ago(history, 1u)) / 5.0f;
    output[3] = output[2];
    output[4] = (asset_temp_c - sample_ago(history, 3u)) / 15.0f;

    variance_count = history->count < 6u ? history->count : 6u;
    for (index = 0; index < variance_count; ++index) mean += sample_ago(history, index);
    mean /= (float)variance_count;
    for (index = 0; index < variance_count; ++index) {
        float difference = sample_ago(history, index) - mean;
        variance += difference * difference;
    }
    output[5] = variance / (float)(variance_count - 1u); /* pandas sample variance, ddof=1 */
    return true;
}
