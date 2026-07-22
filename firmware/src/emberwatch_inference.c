#include "emberwatch_inference.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

void ew_model_reset(ew_model_state_t *state) {
    memset(state->reservoir, 0, sizeof(state->reservoir));
}

ew_model_output_t ew_model_predict(ew_model_state_t *state, const float features[EW_MODEL_INPUT_SIZE]) {
    float normalized[EW_MODEL_INPUT_SIZE];
    float next[EW_MODEL_RESERVOIR_SIZE];
    float design[EW_MODEL_DESIGN_SIZE];
    float logits[EW_MODEL_OUTPUT_SIZE];
    ew_model_output_t output;
    size_t row;
    size_t column;

    for (column = 0; column < EW_MODEL_INPUT_SIZE; ++column) {
        float value = (features[column] - ew_model_feature_mean[column]) / ew_model_feature_std[column];
        if (value > EW_MODEL_CLIP_NORMALIZED) value = EW_MODEL_CLIP_NORMALIZED;
        if (value < -EW_MODEL_CLIP_NORMALIZED) value = -EW_MODEL_CLIP_NORMALIZED;
        normalized[column] = value;
    }

    for (row = 0; row < EW_MODEL_RESERVOIR_SIZE; ++row) {
        float activation = ew_model_input_weights[row * (EW_MODEL_INPUT_SIZE + 1u)];
        for (column = 0; column < EW_MODEL_INPUT_SIZE; ++column) {
            activation += ew_model_input_weights[row * (EW_MODEL_INPUT_SIZE + 1u) + column + 1u] * normalized[column];
        }
        for (column = 0; column < EW_MODEL_RESERVOIR_SIZE; ++column) {
            activation += ew_model_reservoir_weights[row * EW_MODEL_RESERVOIR_SIZE + column] * state->reservoir[column];
        }
        next[row] = (1.0f - EW_MODEL_LEAK_RATE) * state->reservoir[row] + EW_MODEL_LEAK_RATE * tanhf(activation);
    }
    memcpy(state->reservoir, next, sizeof(next));

    design[0] = 1.0f;
    memcpy(&design[1], normalized, sizeof(normalized));
    memcpy(&design[1 + EW_MODEL_INPUT_SIZE], state->reservoir, sizeof(state->reservoir));
    for (row = 0; row < EW_MODEL_OUTPUT_SIZE; ++row) {
        logits[row] = 0.0f;
        for (column = 0; column < EW_MODEL_DESIGN_SIZE; ++column) {
            logits[row] += ew_model_output_weights[row * EW_MODEL_DESIGN_SIZE + column] * design[column];
        }
    }

    {
        float maximum = logits[0];
        float total = 0.0f;
        for (row = 1; row < EW_MODEL_OUTPUT_SIZE; ++row) {
            if (logits[row] > maximum) maximum = logits[row];
        }
        for (row = 0; row < EW_MODEL_OUTPUT_SIZE; ++row) {
            output.probabilities[row] = expf(logits[row] - maximum);
            total += output.probabilities[row];
        }
        output.class_index = 0u;
        output.confidence = 0.0f;
        for (row = 0; row < EW_MODEL_OUTPUT_SIZE; ++row) {
            output.probabilities[row] /= total;
            if (output.probabilities[row] > output.confidence) {
                output.confidence = output.probabilities[row];
                output.class_index = (uint8_t)row;
            }
        }
    }
    output.alert_probability = output.probabilities[1] + output.probabilities[2];
    return output;
}
