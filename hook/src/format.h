// .spot log layout, version 2 (version 1 had no SpotTarget::half_height). Little-endian, packed.
// Mirrored by analyze/spotter_analyze/format.py; change both together and bump SPOT_VERSION.
//
// File  = SpotHeader, then one SpotFrame per rendered frame,
//         each followed by SpotFrame::n_targets SpotTarget records.
//
// Angles are UE conventions in degrees: yaw positive = turning right, pitch positive = up.
// Positions are UE world units (X forward, Y right, Z up at yaw 0).
#pragma once
#include <cstdint>

#pragma pack(push, 1)

constexpr char     SPOT_MAGIC[4] = {'S', 'P', 'O', 'T'};
constexpr uint16_t SPOT_VERSION  = 2;
constexpr uint32_t SPOT_NO_SHOTS = 0xFFFFFFFFu; // shot counter not available this frame
constexpr uint8_t  SPOT_BTN_FIRE = 1;

struct SpotHeader {
    char     magic[4];
    uint16_t version;
    uint16_t header_size;   // sizeof(SpotHeader)
    uint64_t qpc_freq;      // QueryPerformanceFrequency, for reference
    int64_t  start_wall_us; // Unix time in microseconds (UTC) at t_us == 0
    char     game[40];      // game version string, NUL padded
};
static_assert(sizeof(SpotHeader) == 64);

struct SpotFrame {
    uint64_t t_us;          // microseconds since start, from QPC
    float    yaw, pitch;    // camera rotation
    float    fov;           // horizontal FOV as the engine reports it
    float    cam[3];        // camera location
    uint32_t shots;         // cumulative shots fired, or SPOT_NO_SHOTS
    uint8_t  buttons;       // SPOT_BTN_* bits
    uint8_t  n_targets;
    uint16_t reserved;
};
static_assert(sizeof(SpotFrame) == 40);

struct SpotTarget {
    uint32_t id;            // stable for the life of the target actor
    float    pos[3];        // centre of the hittable volume
    float    radius;        // world units
    float    half_height;   // world units; equals radius for a sphere, larger for tall bots
    float    health;
};
static_assert(sizeof(SpotTarget) == 28);

#pragma pack(pop)
