#ifndef EMBERWATCH_INFERENCE_Q_H
#define EMBERWATCH_INFERENCE_Q_H

#include <stdint.h>

#include "emberwatch_model_q.h"

/*
 * Quantized, sparse-reservoir inference. Same semantics as ew_model_predict in
 * emberwatch_inference.h, but weights are int8_t or int16_t with per-row float scales
 * and the reservoir matrix is stored CSR. Accumulation stays in float; only storage is
 * quantized. Reservoir state remains float and is the only mutable RAM the model needs.
 */

typedef struct {
    float reservoir[EW_Q_RESERVOIR_SIZE];
} ew_q_state_t;

typedef struct {
    uint8_t class_index;
    float confidence;
    float alert_probability;
    float probabilities[EW_Q_OUTPUT_SIZE];
} ew_q_output_t;

void ew_q_reset(ew_q_state_t *state);
ew_q_output_t ew_q_predict(ew_q_state_t *state, const float features[EW_Q_INPUT_SIZE]);

#endif
