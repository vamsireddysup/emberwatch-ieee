#include "emberwatch_inference_q.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

void ew_q_reset(ew_q_state_t *state) {
    memset(state->reservoir, 0, sizeof(state->reservoir));
}

ew_q_output_t ew_q_predict(ew_q_state_t *state, const float features[EW_Q_INPUT_SIZE]) {
    float normalized[EW_Q_INPUT_SIZE];
    float next[EW_Q_RESERVOIR_SIZE];
    float design[EW_Q_DESIGN_SIZE];
    float logits[EW_Q_OUTPUT_SIZE];
    ew_q_output_t output;
    size_t row;
    size_t column;

    for (column = 0; column < EW_Q_INPUT_SIZE; ++column) {
        float value = (features[column] - ew_q_feature_mean[column]) / ew_q_feature_std[column];
        if (value > EW_Q_CLIP_NORMALIZED) value = EW_Q_CLIP_NORMALIZED;
        if (value < -EW_Q_CLIP_NORMALIZED) value = -EW_Q_CLIP_NORMALIZED;
        normalized[column] = value;
    }

    /* Reservoir update. Dense int-coded input weights (bias + inputs), CSR int-coded
       recurrent weights. Each row dequantizes with its own float scale, matching
       src/quantize.py. */
    for (row = 0; row < EW_Q_RESERVOIR_SIZE; ++row) {
        const size_t input_base = row * (EW_Q_INPUT_SIZE + 1u);
        float input_acc = (float)ew_q_input_weights[input_base];
        for (column = 0; column < EW_Q_INPUT_SIZE; ++column) {
            input_acc += (float)ew_q_input_weights[input_base + column + 1u] * normalized[column];
        }
        input_acc *= ew_q_input_scales[row];

        float recurrent_acc = 0.0f;
        {
            const uint16_t start = ew_q_reservoir_row_ptr[row];
            const uint16_t end = ew_q_reservoir_row_ptr[row + 1u];
            uint16_t k;
            for (k = start; k < end; ++k) {
                recurrent_acc += (float)ew_q_reservoir_values[k] * state->reservoir[ew_q_reservoir_indices[k]];
            }
            recurrent_acc *= ew_q_reservoir_scales[row];
        }

        next[row] = (1.0f - EW_Q_LEAK_RATE) * state->reservoir[row]
                    + EW_Q_LEAK_RATE * tanhf(input_acc + recurrent_acc);
    }
    memcpy(state->reservoir, next, sizeof(next));

    design[0] = 1.0f;
    memcpy(&design[1], normalized, sizeof(normalized));
    memcpy(&design[1 + EW_Q_INPUT_SIZE], state->reservoir, sizeof(state->reservoir));
    for (row = 0; row < EW_Q_OUTPUT_SIZE; ++row) {
        const size_t base = row * EW_Q_DESIGN_SIZE;
        float accumulator = 0.0f;
        for (column = 0; column < EW_Q_DESIGN_SIZE; ++column) {
            accumulator += (float)ew_q_output_weights[base + column] * design[column];
        }
        logits[row] = accumulator * ew_q_output_scales[row];
    }

    {
        float maximum = logits[0];
        float total = 0.0f;
        for (row = 1; row < EW_Q_OUTPUT_SIZE; ++row) {
            if (logits[row] > maximum) maximum = logits[row];
        }
        for (row = 0; row < EW_Q_OUTPUT_SIZE; ++row) {
            output.probabilities[row] = expf(logits[row] - maximum);
            total += output.probabilities[row];
        }
        output.class_index = 0u;
        output.confidence = 0.0f;
        for (row = 0; row < EW_Q_OUTPUT_SIZE; ++row) {
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
