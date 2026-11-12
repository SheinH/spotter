// Offsets for one specific KovaaK's build, taken from a Dumper-7 SDK dump (UE 4.26.2 fork).
// The DLL checks the exe fingerprint below and refuses to sample on any other build:
// after a game patch, re-dump the SDK and update this file.
#pragma once
#include <cstdint>

namespace off {

// PE header of FPSAimTrainer-Win64-Shipping.exe, game version 3.9.13 (2026-11-02)
constexpr uint32_t kTimeDateStamp = 0x6AE8C5C1;
constexpr uint32_t kSizeOfImage   = 0x05811000;

// Globals (RVA from the exe base)
constexpr uintptr_t GWorld       = 0x052B1AC8;
constexpr uintptr_t AppendString = 0x015C9460;   // void FName::AppendString(FString&) const

// CoreUObject
constexpr uintptr_t UObject_Class      = 0x10;
constexpr uintptr_t UObject_Name       = 0x18;
constexpr uintptr_t UStruct_SuperStruct = 0x40;

// Engine
constexpr uintptr_t UWorld_PersistentLevel     = 0x30;
constexpr uintptr_t UWorld_OwningGameInstance  = 0x180;
constexpr uintptr_t UGameInstance_LocalPlayers = 0x38;    // TArray<ULocalPlayer*>
constexpr uintptr_t UPlayer_PlayerController   = 0x30;
constexpr uintptr_t APlayerController_AcknowledgedPawn    = 0x2A0;
constexpr uintptr_t APlayerController_PlayerCameraManager = 0x2B8;
constexpr uintptr_t APlayerCameraManager_POV   = 0x1AE0 + 0x10;  // CameraCachePrivate.POV
constexpr uintptr_t POV_Location = 0x00, POV_Rotation = 0x0C, POV_FOV = 0x18;
constexpr uintptr_t ULevel_Actors              = 0x98;    // TArray<AActor*>
constexpr uintptr_t AActor_RootComponent       = 0x130;
constexpr uintptr_t USceneComponent_RelativeLocation = 0x11C;
constexpr uintptr_t USceneComponent_RelativeScale3D  = 0x134;
constexpr uintptr_t ACharacter_CapsuleComponent = 0x290;
constexpr uintptr_t UCapsuleComponent_Radius    = 0x464;

// GameSkillsTrainer
constexpr uintptr_t AMetaCharacter_CachedWeapons = 0x940;  // TArray<AWeaponParentActor*>
constexpr uintptr_t AMetaCharacter_Health        = 0xE58;
constexpr uintptr_t AMetaCharacter_BoundingBox   = 0x610 + 0xE0;  // mCharacterProfileNative.CharBoundingBox
constexpr uintptr_t BB_Type = 0x0, BB_Height = 0x4, BB_Radius = 0x8;
constexpr uintptr_t AWeaponParentActor_ShotsFiredThisSession = 0x290;
constexpr uintptr_t AWeaponParentActor_bShootHeld            = 0x380;

}  // namespace off
