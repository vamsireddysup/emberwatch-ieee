#include "emberwatch_protocol.h"

#include <assert.h>
#include <string.h>

int main(void) {
    uint8_t packet[EW_PACKET_SIZE];
    ew_telemetry_t input = {
        .flags = EW_FLAG_ALERT | EW_FLAG_MODEL_VALID,
        .device_id = 42,
        .sequence = 65530,
        .uptime_s = 123456,
        .asset_temp_c_x100 = 6342,
        .ambient_temp_c_x100 = -525,
        .confidence_u8 = 231,
        .state = EW_STATE_ANOMALY,
        .battery_mv = 3190,
    };
    ew_telemetry_t output;
    ew_encode_telemetry(&input, packet);
    assert(ew_decode_telemetry(packet, &output));
    assert(output.flags == input.flags);
    assert(output.device_id == input.device_id);
    assert(output.sequence == input.sequence);
    assert(output.uptime_s == input.uptime_s);
    assert(output.asset_temp_c_x100 == input.asset_temp_c_x100);
    assert(output.ambient_temp_c_x100 == input.ambient_temp_c_x100);
    assert(output.confidence_u8 == input.confidence_u8);
    assert(output.state == input.state);
    assert(output.battery_mv == input.battery_mv);
    packet[12] ^= 1u;
    assert(!ew_decode_telemetry(packet, &output));
    return 0;
}
