#ifndef EMBERWATCH_PROTOCOL_H
#define EMBERWATCH_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EW_PACKET_SIZE 22u
#define EW_PACKET_MAGIC 0xE7u
#define EW_PROTOCOL_VERSION 1u

enum {
    EW_STATE_NORMAL = 0,
    EW_STATE_WARMING = 1,
    EW_STATE_ANOMALY = 2,
    EW_STATE_SENSOR_FAULT = 3
};

enum {
    EW_FLAG_ALERT = 1u << 0,
    EW_FLAG_BATTERY_LOW = 1u << 1,
    EW_FLAG_SENSOR_FAULT = 1u << 2,
    EW_FLAG_MODEL_VALID = 1u << 3
};

typedef struct {
    uint8_t flags;
    uint16_t device_id;
    uint16_t sequence;
    uint32_t uptime_s;
    int16_t asset_temp_c_x100;
    int16_t ambient_temp_c_x100;
    uint8_t confidence_u8;
    uint8_t state;
    uint16_t battery_mv;
} ew_telemetry_t;

uint16_t ew_crc16_ccitt(const uint8_t *data, size_t length);
void ew_encode_telemetry(const ew_telemetry_t *telemetry, uint8_t output[EW_PACKET_SIZE]);
bool ew_decode_telemetry(const uint8_t input[EW_PACKET_SIZE], ew_telemetry_t *telemetry);

#endif
