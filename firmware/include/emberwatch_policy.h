#ifndef EMBERWATCH_POLICY_H
#define EMBERWATCH_POLICY_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint8_t state;
    uint8_t warming_run;
    uint8_t normal_run;
    uint16_t samples_since_transmit;
} ew_alert_policy_t;

typedef struct {
    uint8_t state;
    bool transmit;
} ew_alert_decision_t;

void ew_alert_policy_reset(ew_alert_policy_t *policy, uint16_t heartbeat_samples);
ew_alert_decision_t ew_alert_policy_step(
    ew_alert_policy_t *policy,
    float warming_probability,
    float anomaly_probability,
    float threshold,
    uint8_t warming_confirmations,
    uint8_t clear_confirmations,
    uint16_t heartbeat_samples
);

#endif
