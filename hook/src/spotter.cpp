// spotter.dll: read-only telemetry sampler for KovaaK's.
//
// One background thread polls the game's memory at 500 Hz and appends camera rotation,
// live sensitivity, shot counter and live target positions to runs/spotter-<time>.spot (see format.h).
// It never writes to game memory, hooks nothing, and does not hide itself.
#include <windows.h>
#include <share.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>

#include "format.h"
#include "offsets.h"

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTargets = 64;
constexpr int kPollMs = 2;

HMODULE g_self = nullptr;
uintptr_t g_base = 0;
std::atomic<bool> g_stop{false};
FILE* g_log = nullptr;

void logf(const char* fmt, ...) {
    if (!g_log) return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    fprintf(g_log, "%02d:%02d:%02d.%03d  ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
}

template <typename T>
inline T rd(uintptr_t addr) { return *reinterpret_cast<const T*>(addr); }

struct Vec3 { float x, y, z; };
struct FStringRaw { wchar_t* data; int32_t num; int32_t max; };

// ---- "is this object an AMetaCharacter?" cached per UClass pointer ----
struct ClassEntry { uintptr_t cls; bool is_char; };
ClassEntry g_classes[512];
int g_class_count = 0;

float g_seen_radius = -1.f, g_seen_scale = -1.f;

struct NewClass { bool set; wchar_t name[96]; float health, bb_radius, bb_height, capsule, scale; uint8_t bb_type; };

// Raw (may fault): name of a UObject into buf. Uses the engine's own FName::AppendString
// with a caller-owned buffer large enough that the engine never reallocates it.
void object_name(uintptr_t obj, wchar_t* buf, int cap) {
    buf[0] = 0;
    FStringRaw s{buf, 0, cap};
    using AppendFn = void (*)(const void* fname, FStringRaw* out);
    reinterpret_cast<AppendFn>(g_base + off::AppendString)(
        reinterpret_cast<const void*>(obj + off::UObject_Name), &s);
    if (s.data != buf) { buf[0] = 0; return; }   // engine reallocated: treat as unknown
    buf[(s.num > 0 && s.num < cap) ? s.num - (buf[s.num - 1] == 0 ? 1 : 0) : 0] = 0;
}

bool is_meta_character(uintptr_t obj, wchar_t* leaf_name_out /* 96, may be null */) {
    const uintptr_t cls = rd<uintptr_t>(obj + off::UObject_Class);
    if (!cls) return false;
    for (int i = 0; i < g_class_count; ++i)
        if (g_classes[i].cls == cls) return g_classes[i].is_char;
    bool found = false;
    wchar_t name[128];
    uintptr_t c = cls;
    for (int depth = 0; c && depth < 32; ++depth) {
        object_name(c, name, 128);
        if (depth == 0 && leaf_name_out) wcsncpy_s(leaf_name_out, 96, name, _TRUNCATE);
        if (wcscmp(name, L"MetaCharacter") == 0) { found = true; break; }
        c = rd<uintptr_t>(c + off::UStruct_SuperStruct);
    }
    if (g_class_count < 512) g_classes[g_class_count++] = {cls, found};
    return found;
}

// ---- stable ids: a target keeps its id while it stays alive at the same address ----
struct Live { uintptr_t ptr; uint32_t id; };
Live g_live[kMaxTargets];
int g_live_count = 0;
uint32_t g_next_id = 1;

struct Sample {
    SpotFrame frame;
    SpotTarget targets[kMaxTargets];
    uintptr_t ptrs[kMaxTargets];
    uintptr_t world;
    int actors;
    NewClass fresh;
};

// Raw read of one sample. Every pointer chase here can fault if the game is mid-teardown,
// so the caller wraps this in SEH and drops the sample on a fault.
bool sample_raw(Sample& s) {
    const uintptr_t world = rd<uintptr_t>(g_base + off::GWorld);
    if (!world) return false;
    if (world != s.world) { g_class_count = 0; s.world = world; }   // classes can unload with the level
    const uintptr_t gi = rd<uintptr_t>(world + off::UWorld_OwningGameInstance);
    if (!gi) return false;
    const uintptr_t players = rd<uintptr_t>(gi + off::UGameInstance_LocalPlayers);
    if (!players || rd<int32_t>(gi + off::UGameInstance_LocalPlayers + 8) < 1) return false;
    const uintptr_t player = rd<uintptr_t>(players);
    const uintptr_t pc = player ? rd<uintptr_t>(player + off::UPlayer_PlayerController) : 0;
    if (!pc) return false;
    const uintptr_t pcm = rd<uintptr_t>(pc + off::APlayerController_PlayerCameraManager);
    const uintptr_t pawn = rd<uintptr_t>(pc + off::APlayerController_AcknowledgedPawn);
    if (!pcm || !pawn || !is_meta_character(pawn, nullptr)) return false;   // menus, map editor

    const uintptr_t pov = pcm + off::APlayerCameraManager_POV;
    const Vec3 loc = rd<Vec3>(pov + off::POV_Location);
    const Vec3 rot = rd<Vec3>(pov + off::POV_Rotation);   // FRotator: pitch, yaw, roll
    s.frame.cam[0] = loc.x; s.frame.cam[1] = loc.y; s.frame.cam[2] = loc.z;
    s.frame.pitch = rot.x; s.frame.yaw = rot.y;
    s.frame.fov = rd<float>(pov + off::POV_FOV);
    s.frame.input_yaw_scale = rd<float>(pc + off::APlayerController_InputYawScale);
    s.frame.user_yaw_sens = rd<float>(pc + off::AMetaPlayerController_UserYawSensitivity);

    uint32_t shots = 0;
    bool held = false;
    const uintptr_t weapons = rd<uintptr_t>(pawn + off::AMetaCharacter_CachedWeapons);
    int32_t nweapons = rd<int32_t>(pawn + off::AMetaCharacter_CachedWeapons + 8);
    if (!weapons || nweapons < 0 || nweapons > 16) nweapons = 0;
    for (int i = 0; i < nweapons; ++i) {
        const uintptr_t w = rd<uintptr_t>(weapons + 8ull * i);
        if (!w) continue;
        shots += static_cast<uint32_t>(rd<int32_t>(w + off::AWeaponParentActor_ShotsFiredThisSession));
        held |= rd<uint8_t>(w + off::AWeaponParentActor_bShootHeld) != 0;
    }
    s.frame.shots = nweapons ? shots : SPOT_NO_SHOTS;
    s.frame.buttons = held ? SPOT_BTN_FIRE : 0;

    const uintptr_t level = rd<uintptr_t>(world + off::UWorld_PersistentLevel);
    if (!level) return false;
    const uintptr_t actors = rd<uintptr_t>(level + off::ULevel_Actors);
    const int32_t nactors = rd<int32_t>(level + off::ULevel_Actors + 8);
    if (!actors || nactors < 0 || nactors > 50000) return false;
    s.actors = nactors;

    int n = 0;
    for (int i = 0; i < nactors && n < kMaxTargets; ++i) {
        const uintptr_t a = rd<uintptr_t>(actors + 8ull * i);
        if (!a || a == pawn) continue;
        const int known = g_class_count;
        wchar_t leaf[96];
        if (!is_meta_character(a, leaf)) continue;
        const float health = rd<float>(a + off::AMetaCharacter_Health);
        const uintptr_t root = rd<uintptr_t>(a + off::AActor_RootComponent);
        if (!root) continue;
        const float scale = rd<float>(root + off::USceneComponent_RelativeScale3D);
        const uintptr_t bb = a + off::AMetaCharacter_BoundingBox;
        const float bb_radius = rd<float>(bb + off::BB_Radius);
        (void)known;
        if (!s.fresh.set && (bb_radius != g_seen_radius || scale != g_seen_scale)) {   // new hitbox: report it
            g_seen_radius = bb_radius; g_seen_scale = scale;
            s.fresh.set = true;
            wcsncpy_s(s.fresh.name, leaf, _TRUNCATE);
            s.fresh.health = health; s.fresh.bb_radius = bb_radius; s.fresh.scale = scale;
            s.fresh.bb_height = rd<float>(bb + off::BB_Height);
            s.fresh.bb_type = rd<uint8_t>(bb + off::BB_Type);
            const uintptr_t capsule = rd<uintptr_t>(a + off::ACharacter_CapsuleComponent);
            s.fresh.capsule = capsule ? rd<float>(capsule + off::UCapsuleComponent_Radius) : -1.f;
        }
        if (!(health > 0.f)) continue;                     // dead, waiting to respawn
        const Vec3 p = rd<Vec3>(root + off::USceneComponent_RelativeLocation);
        s.ptrs[n] = a;
        // Spheroid (type 2) is a ball; cylinders and cuboids are as tall as the profile says.
        const float half_height = rd<uint8_t>(bb + off::BB_Type) == 2
            ? bb_radius : (std::max)(bb_radius, rd<float>(bb + off::BB_Height) * 0.5f);
        s.targets[n] = {0, {p.x, p.y, p.z}, bb_radius * scale, half_height * scale, health};
        ++n;
    }
    s.frame.n_targets = static_cast<uint8_t>(n);
    return true;
}

bool sample_guarded(Sample& s) {
    __try {
        return sample_raw(s);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

void assign_ids(Sample& s) {
    Live next[kMaxTargets];
    const int n = s.frame.n_targets;
    for (int i = 0; i < n; ++i) {
        uint32_t id = 0;
        for (int j = 0; j < g_live_count; ++j)
            if (g_live[j].ptr == s.ptrs[i]) { id = g_live[j].id; break; }
        if (!id) id = g_next_id++;
        s.targets[i].id = id;
        next[i] = {s.ptrs[i], id};
    }
    memcpy(g_live, next, sizeof(Live) * n);
    g_live_count = n;
}

fs::path runs_dir() {
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(g_self, self, MAX_PATH);
    const fs::path dll_dir = fs::path(self).parent_path();
    std::error_code ec;
    for (fs::path p = dll_dir; p.has_relative_path(); p = p.parent_path()) {
        if (fs::is_directory(p / L"runs", ec)) return p / L"runs";
        if (p == p.parent_path()) break;
    }
    return dll_dir;
}

bool same_build() {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(g_base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(g_base + dos->e_lfanew);
    logf("game exe: TimeDateStamp=%08X SizeOfImage=%08X", nt->FileHeader.TimeDateStamp,
         nt->OptionalHeader.SizeOfImage);
    return nt->FileHeader.TimeDateStamp == off::kTimeDateStamp &&
           nt->OptionalHeader.SizeOfImage == off::kSizeOfImage;
}

DWORD WINAPI run(LPVOID) {
    g_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    const fs::path dir = runs_dir();
    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t stamp[64];
    swprintf_s(stamp, L"spotter-%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour,
               st.wMinute, st.wSecond);
    g_log = _wfsopen((dir / (std::wstring(stamp) + L".log")).c_str(), L"w", _SH_DENYNO);   // readable while the game runs
    logf("spotter loaded, pid %lu", GetCurrentProcessId());
    if (!same_build()) {
        logf("this is not the game build the offsets were dumped from; not sampling. "
             "Re-dump the SDK and update hook/src/offsets.h.");
        if (g_log) { fclose(g_log); g_log = nullptr; }
        FreeLibraryAndExitThread(g_self, 0);
    }

    FILE* out = _wfsopen((dir / (std::wstring(stamp) + L".spot")).c_str(), L"wb", _SH_DENYNO);
    if (!out) { logf("cannot create the .spot file in %ls", dir.c_str()); return 0; }
    setvbuf(out, nullptr, _IOFBF, 1 << 20);

    LARGE_INTEGER freq, t0;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    const int64_t unix_us = ((static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) / 10
                            - 11644473600000000LL;
    SpotHeader h{};
    memcpy(h.magic, SPOT_MAGIC, 4);
    h.version = SPOT_VERSION;
    h.header_size = sizeof(h);
    h.qpc_freq = static_cast<uint64_t>(freq.QuadPart);
    h.start_wall_us = unix_us;
    sprintf_s(h.game, "kovaaks ts=%08X", off::kTimeDateStamp);
    fwrite(&h, sizeof(h), 1, out);

    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS);
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -10000LL * kPollMs;
        SetWaitableTimer(timer, &due, kPollMs, nullptr, nullptr, FALSE);
    } else {
        logf("no high-resolution timer; falling back to Sleep(1)");
    }

    static Sample s{}, last{};
    uint64_t written = 0, faults = 0, last_write_us = 0, last_flush_us = 0, last_beat_us = 0;
    bool was_active = false;
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (timer) WaitForSingleObject(timer, 50); else Sleep(1);
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        const uint64_t t_us = static_cast<uint64_t>((now.QuadPart - t0.QuadPart) * 1000000 / freq.QuadPart);

        s.frame = SpotFrame{};
        s.fresh.set = false;
        const bool ok = sample_guarded(s);
        if (!ok) ++faults;
        if (ok != was_active) {
            logf(ok ? "in a scenario: sampling" : "no player/world: paused");
            was_active = ok;
        }
        if (s.fresh.set)
            logf("target class %ls: health=%.1f bb_type=%u bb_radius=%.2f bb_height=%.2f capsule_radius=%.2f scale=%.3f",
                 s.fresh.name, s.fresh.health, s.fresh.bb_type, s.fresh.bb_radius, s.fresh.bb_height,
                 s.fresh.capsule, s.fresh.scale);
        if (ok) {
            assign_ids(s);
            s.frame.t_us = t_us;
            const int n = s.frame.n_targets;
            // Polling is faster than the game renders, so skip samples identical to the last
            // one written (they would read as zero-velocity stutter), but never go quiet
            // for more than 10 ms.
            SpotFrame a = s.frame, b = last.frame;
            a.t_us = b.t_us = 0;
            const bool same = memcmp(&a, &b, sizeof(a)) == 0 &&
                              memcmp(s.targets, last.targets, sizeof(SpotTarget) * n) == 0;
            if (!same || t_us - last_write_us >= 10000) {
                fwrite(&s.frame, sizeof(SpotFrame), 1, out);
                if (n) fwrite(s.targets, sizeof(SpotTarget), n, out);
                last.frame = s.frame;
                memcpy(last.targets, s.targets, sizeof(SpotTarget) * n);
                last_write_us = t_us;
                ++written;
            }
        }
        if (t_us - last_flush_us >= 250000) { fflush(out); last_flush_us = t_us; }
        if (t_us - last_beat_us >= 10000000) {
            logf("frames=%llu skipped_reads=%llu actors=%d targets=%u shots=%u yaw=%.2f pitch=%.2f fov=%.1f "
                 "yaw_scale=%.6f user_sens=%.6f",
                 written, faults, s.actors, s.frame.n_targets, s.frame.shots, s.frame.yaw, s.frame.pitch,
                 s.frame.fov, s.frame.input_yaw_scale, s.frame.user_yaw_sens);
            last_beat_us = t_us;
        }
    }
    fclose(out);
    return 0;
}

}  // namespace

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = module;
        DisableThreadLibraryCalls(module);
        if (HANDLE t = CreateThread(nullptr, 0, run, nullptr, 0, nullptr)) CloseHandle(t);
    } else if (reason == DLL_PROCESS_DETACH) {
        g_stop = true;
    }
    return TRUE;
}
