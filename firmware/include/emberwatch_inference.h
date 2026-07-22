#ifndef EMBERWATCH_INFERENCE_H
#define EMBERWATCH_INFERENCE_H

#include <stdint.h>

#include "emberwatch_model.h"

typedef struct {
    float reservoir[EW_MODEL_RESERVOIR_SIZE];
} ew_model_state_t;

typedef struct {
    uint8_t class_index;
    float confidence;
    float alert_probability;
    float probabilities[EW_MODEL_OUTPUT_SIZE];
} ew_model_output_t;

void ew_model_reset(ew_model_state_t *state);
ew_model_output_t ew_model_predict(ew_model_state_t *state, const float features[EW_MODEL_INPUT_SIZE]);

#endif
