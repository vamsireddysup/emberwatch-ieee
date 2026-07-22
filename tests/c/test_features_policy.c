#include "emberwatch_features.h"
#include "emberwatch_policy.h"
#include "emberwatch_protocol.h"

#include <assert.h>
#include <math.h>

int main(void) {
    ew_feature_history_t history;
    float features[EW_FEATURE_COUNT];
    ew_alert_policy_t policy;
    ew_alert_decision_t decision;

    ew_features_reset(&history);
    assert(!ew_features_update(&history, 20.0f, 10.0f, features));
    assert(!ew_features_update(&history, 21.0f, 10.0f, features));
    assert(!ew_features_update(&history, 22.0f, 10.0f, features));
    assert(ew_features_update(&history, 23.0f, 10.0f, features));
    assert(fabsf(features[0] - 10.0f) < 1e-6f);
    assert(fabsf(features[1] - 13.0f) < 1e-6f);
    assert(fabsf(features[2] - 0.2f) < 1e-6f);
    assert(fabsf(features[4] - 0.2f) < 1e-6f);
    assert(fabsf(features[5] - 1.6666666f) < 1e-5f);

    ew_alert_policy_reset(&policy, 72u);
    decision = ew_alert_policy_step(&policy, 0.1f, 0.0f, 0.7f, 2u, 3u, 72u);
    assert(decision.transmit && decision.state == EW_STATE_NORMAL);
    decision = ew_alert_policy_step(&policy, 0.8f, 0.0f, 0.7f, 2u, 3u, 72u);
    assert(!decision.transmit && decision.state == EW_STATE_NORMAL);
    decision = ew_alert_policy_step(&policy, 0.8f, 0.0f, 0.7f, 2u, 3u, 72u);
    assert(decision.transmit && decision.state == EW_STATE_WARMING);
    decision = ew_alert_policy_step(&policy, 0.0f, 0.9f, 0.7f, 2u, 3u, 72u);
    assert(decision.transmit && decision.state == EW_STATE_ANOMALY);
    return 0;
}
