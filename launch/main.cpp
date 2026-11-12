// spotter-launch: load spotter.dll into a running KovaaK's.
//
//   spotter-launch [--dll <path>] [--wait <seconds>] [--require-offline]
//
// Plain LoadLibrary injection; nothing here hides the DLL from the game or its anti-cheat.
// --require-offline refuses to inject unless Steam's most recent login is in offline mode.
#include <windows.h>
#include <tlhelp32.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

static const wchar_t* kGameExe = L"FPSAimTrainer-Win64-Shipping.exe";

static DWORD find_process(const wchar_t* exe) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32W pe{sizeof(pe)};
    DWORD pid = 0;
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
        if (_wcsicmp(pe.szExeFile, exe) == 0) { pid = pe.th32ProcessID; break; }
    }
    CloseHandle(snap);
    return pid;
}

static bool module_loaded(DWORD pid, const std::wstring& name) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
    if (snap == INVALID_HANDLE_VALUE) return false;
    MODULEENTRY32W me{sizeof(me)};
    bool found = false;
    for (BOOL ok = Module32FirstW(snap, &me); ok; ok = Module32NextW(snap, &me)) {
        if (_wcsicmp(me.szModule, name.c_str()) == 0) { found = true; break; }
    }
    CloseHandle(snap);
    return found;
}

// True only if Steam's most recently used account has "WantsOfflineMode" "1" in loginusers.vdf.
// Anything we cannot read counts as "not offline".
static bool steam_is_offline(std::wstring& why) {
    wchar_t path[MAX_PATH];
    DWORD size = sizeof(path);
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"SteamPath", RRF_RT_REG_SZ,
                     nullptr, path, &size) != ERROR_SUCCESS) {
        why = L"could not find Steam in the registry";
        return false;
    }
    std::ifstream in(fs::path(path) / "config" / "loginusers.vdf");
    if (!in) { why = L"could not read Steam's loginusers.vdf"; return false; }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    // Value of "key" "value" inside one account block, or "" if absent.
    auto value = [](const std::string& block, const char* key) -> std::string {
        size_t k = block.find(std::string("\"") + key + "\"");
        if (k == std::string::npos) return "";
        size_t q1 = block.find('"', k + std::strlen(key) + 2);
        size_t q2 = q1 == std::string::npos ? q1 : block.find('"', q1 + 1);
        return q2 == std::string::npos ? "" : block.substr(q1 + 1, q2 - q1 - 1);
    };
    // Each account is a { ... } block one level below the root "users" block. The current
    // account is the one flagged MostRecent (older Steam) or with the newest Timestamp.
    bool found = false, offline = false;
    long long newest = -1;
    size_t pos = text.find('{');
    while (pos != std::string::npos) {
        size_t open = text.find('{', pos + 1);
        if (open == std::string::npos) break;
        size_t close = text.find('}', open);
        if (close == std::string::npos) break;
        const std::string block = text.substr(open, close - open);
        const std::string stamp = value(block, "Timestamp");
        long long when = value(block, "MostRecent") == "1" ? LLONG_MAX : std::atoll(stamp.c_str());
        if (when > newest) {
            newest = when;
            found = true;
            offline = value(block, "WantsOfflineMode") == "1";
        }
        pos = close;
    }
    if (!found) { why = L"no Steam account found in loginusers.vdf"; return false; }
    if (!offline) why = L"Steam's current account is not in offline mode";
    return offline;
}

static bool inject(DWORD pid, const std::wstring& dll, std::wstring& why) {
    HANDLE proc = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION |
                              PROCESS_VM_WRITE | PROCESS_VM_READ, FALSE, pid);
    if (!proc) { why = L"OpenProcess failed (" + std::to_wstring(GetLastError()) + L")"; return false; }
    const SIZE_T bytes = (dll.size() + 1) * sizeof(wchar_t);
    bool ok = false;
    void* mem = VirtualAllocEx(proc, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!mem) {
        why = L"VirtualAllocEx failed";
    } else if (!WriteProcessMemory(proc, mem, dll.c_str(), bytes, nullptr)) {
        why = L"WriteProcessMemory failed";
    } else {
        // kernel32 is mapped at the same address in every process of a session.
        auto load = reinterpret_cast<LPTHREAD_START_ROUTINE>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"));
        HANDLE thread = CreateRemoteThread(proc, nullptr, 0, load, mem, 0, nullptr);
        if (!thread) {
            why = L"CreateRemoteThread failed (" + std::to_wstring(GetLastError()) + L")";
        } else {
            WaitForSingleObject(thread, 15000);
            DWORD code = 0;
            GetExitCodeThread(thread, &code);
            CloseHandle(thread);
            ok = code != 0 && code != STILL_ACTIVE;
            if (!ok) why = L"LoadLibraryW returned null inside the game (missing dependency or DllMain refused)";
        }
    }
    if (mem) VirtualFreeEx(proc, mem, 0, MEM_RELEASE);
    CloseHandle(proc);
    return ok;
}

int wmain(int argc, wchar_t** argv) {
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    fs::path dll = fs::path(self).parent_path() / L"spotter.dll";
    int wait_s = 0;
    bool require_offline = false;

    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--dll" && i + 1 < argc) dll = argv[++i];
        else if (a == L"--wait" && i + 1 < argc) wait_s = _wtoi(argv[++i]);
        else if (a == L"--require-offline") require_offline = true;
        else {
            std::wcerr << L"usage: spotter-launch [--dll <path>] [--wait <seconds>] [--require-offline]\n";
            return 2;
        }
    }

    std::error_code ec;
    dll = fs::absolute(dll, ec);
    if (!fs::is_regular_file(dll, ec)) {
        std::wcerr << L"DLL not found: " << dll.wstring() << L"\n";
        return 1;
    }
    std::wstring why;
    if (require_offline && !steam_is_offline(why)) {
        std::wcerr << L"refusing to inject: " << why << L" (--require-offline)\n";
        return 3;
    }

    DWORD pid = find_process(kGameExe);
    for (int waited = 0; !pid && waited < wait_s; ++waited) {
        if (waited == 0) std::wcout << L"waiting for " << kGameExe << L"...\n";
        Sleep(1000);
        pid = find_process(kGameExe);
    }
    if (!pid) {
        std::wcerr << kGameExe << L" is not running. Start KovaaK's first, or pass --wait <seconds>.\n";
        return 1;
    }
    if (module_loaded(pid, dll.filename().wstring())) {
        std::wcout << dll.filename().wstring() << L" is already loaded in pid " << pid << L"\n";
        return 0;
    }
    if (!inject(pid, dll.wstring(), why)) {
        std::wcerr << L"injection failed: " << why << L"\n";
        return 1;
    }
    std::wcout << L"loaded " << dll.filename().wstring() << L" into pid " << pid << L"\n";
    return 0;
}
