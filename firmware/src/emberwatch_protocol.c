#include "emberwatch_protocol.h"

static void write_u16(uint8_t *output, uint16_t value) {
    output[0] = (uint8_t)(value & 0xFFu);
    output[1] = (uint8_t)(value >> 8);
}

static void write_u32(uint8_t *output, uint32_t value) {
    output[0] = (uint8_t)(value & 0xFFu);
    output[1] = (uint8_t)((value >> 8) & 0xFFu);
    output[2] = (uint8_t)((value >> 16) & 0xFFu);
    output[3] = (uint8_t)(value >> 24);
}

static uint16_t read_u16(const uint8_t *input) {
    return (uint16_t)input[0] | ((uint16_t)input[1] << 8);
}

static uint32_t read_u32(const uint8_t *input) {
    return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
           ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

uint16_t ew_crc16_ccitt(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFFu;
    size_t index;
    for (index = 0; index < length; ++index) {
        uint8_t bit;
        crc ^= (uint16_t)data[index] << 8;
        for (bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

void ew_encode_telemetry(const ew_telemetry_t *telemetry, uint8_t output[EW_PACKET_SIZE]) {
    uint16_t crc;
    output[0] = EW_PACKET_MAGIC;
    output[1] = EW_PROTOCOL_VERSION;
    output[2] = 1u;
    output[3] = telemetry->flags;
    write_u16(&output[4], telemetry->device_id);
    write_u16(&output[6], telemetry->sequence);
    write_u32(&output[8], telemetry->uptime_s);
    write_u16(&output[12], (uint16_t)telemetry->asset_temp_c_x100);
    write_u16(&output[14], (uint16_t)telemetry->ambient_temp_c_x100);
    output[16] = telemetry->confidence_u8;
    output[17] = telemetry->state;
    write_u16(&output[18], telemetry->battery_mv);
    crc = ew_crc16_ccitt(output, 20u);
    write_u16(&output[20], crc);
}

bool ew_decode_telemetry(const uint8_t input[EW_PACKET_SIZE], ew_telemetry_t *telemetry) {
    if (input[0] != EW_PACKET_MAGIC || input[1] != EW_PROTOCOL_VERSION || input[2] != 1u) {
        return false;
    }
    if (read_u16(&input[20]) != ew_crc16_ccitt(input, 20u)) {
        return false;
    }
    telemetry->flags = input[3];
    telemetry->device_id = read_u16(&input[4]);
    telemetry->sequence = read_u16(&input[6]);
    telemetry->uptime_s = read_u32(&input[8]);
    telemetry->asset_temp_c_x100 = (int16_t)read_u16(&input[12]);
    telemetry->ambient_temp_c_x100 = (int16_t)read_u16(&input[14]);
    telemetry->confidence_u8 = input[16];
    telemetry->state = input[17];
    telemetry->battery_mv = read_u16(&input[18]);
    return true;
}
