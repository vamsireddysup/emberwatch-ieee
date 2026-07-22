#include "emberwatch_policy.h"

#include "emberwatch_protocol.h"

void ew_alert_policy_reset(ew_alert_policy_t *policy, uint16_t heartbeat_samples) {
    policy->state = EW_STATE_NORMAL;
    policy->warming_run = 0u;
    policy->normal_run = 0u;
    policy->samples_since_transmit = heartbeat_samples;
}

ew_alert_decision_t ew_alert_policy_step(
    ew_alert_policy_t *policy,
    float warming_probability,
    float anomaly_probability,
    float threshold,
    uint8_t warming_confirmations,
    uint8_t clear_confirmations,
    uint16_t heartbeat_samples
) {
    uint8_t previous = policy->state;
    float alert_probability = warming_probability + anomaly_probability;
    if (anomaly_probability >= threshold) {
        policy->state = EW_STATE_ANOMALY;
        policy->warming_run = 0u;
        policy->normal_run = 0u;
    } else if (alert_probability >= threshold) {
        if (policy->warming_run < 255u) policy->warming_run++;
        policy->normal_run = 0u;
        if (policy->warming_run >= warming_confirmations && policy->state == EW_STATE_NORMAL) {
            policy->state = EW_STATE_WARMING;
        }
    } else {
        if (policy->normal_run < 255u) policy->normal_run++;
        policy->warming_run = 0u;
        if (policy->normal_run >= clear_confirmations) policy->state = EW_STATE_NORMAL;
    }

    {
        bool changed = policy->state != previous;
        bool heartbeat = policy->samples_since_transmit >= heartbeat_samples;
        ew_alert_decision_t decision = {policy->state, changed || heartbeat};
        if (decision.transmit) {
            policy->samples_since_transmit = 0u;
        } else if (policy->samples_since_transmit < UINT16_MAX) {
            policy->samples_since_transmit++;
        }
        return decision;
    }
}
