// Activate Unity 6's Meta Quest *build profile* (not just plain Android).
//
// Unity 6.1+ ships a derived "Meta Quest" classic platform on top of Android
// with its own player/quality overrides (Vulkan, IL2CPP, ARM64, MultiView,
// Quest-tuned quality level). It is auto-registered by the Editor when the
// Android module is installed, and it lives alongside the plain Android
// profile in `BuildProfileContext.classicPlatformProfiles`.
//
// IMPORTANT BACKGROUND
// --------------------
// 1. The public `BuildProfile.SetActiveBuildProfile(p)` REJECTS classic
//    profiles outright:
//      "[BuildProfile] Classic Platforms cannot be set as the active build
//       profile."
//    See `BuildProfileContext.activeProfile` setter in the Unity reference
//    source — it logs that exact warning if you try.
//
// 2. The supported way to switch a *classic* platform (which Meta Quest is —
//    a derived platform of Android) is the internal native binding
//    `EditorUserBuildSettings.SwitchActiveBuildTargetGuid(BuildProfile)`,
//    wrapped publicly in 6000.2 by
//    `BuildProfileModuleUtil.SwitchLegacyActiveFromBuildProfile(p)`. Both
//    are reached here via reflection because the wrapper status flips
//    between Unity versions.
//
// 3. The Meta Quest derived-platform GUID is hardcoded in
//    `BuildTargetDiscovery.bindings.cs` as
//    "80657fe557de4d17822398b3a01b8c9e". We hardcode the same constant
//    here so we don't depend on display-name strings (localised) or on
//    `m_BuildSubtarget == 6` heuristics.

using System;
using System.Collections;
using System.Reflection;
using UnityEditor;
using UnityEditor.Build.Profile;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        // Stable GUID Unity uses internally for the Meta Quest derived
        // platform (see Editor/Mono/BuildTargetDiscovery.bindings.cs in
        // the Unity reference source).
        const string k_MetaQuestPlatformGuid = "80657fe557de4d17822398b3a01b8c9e";

        // Cached reflective handles. Populated lazily by
        // ResolveBuildProfileApi() and reused.
        static bool _bpResolved;
        static Type _bpContextType;
        static object _bpContextInstance;
        static MethodInfo _bpGetForClassicPlatformByGuid;
        static MethodInfo _bpSwitchActiveByProfile;          // EditorUserBuildSettings.SwitchActiveBuildTargetGuid(BuildProfile)
        static MethodInfo _bpModuleUtilSwitchLegacy;         // BuildProfileModuleUtil.SwitchLegacyActiveFromBuildProfile(BuildProfile) — public wrapper in 6000.2
        static PropertyInfo _bpActivePlatformGuidProp;       // EditorUserBuildSettings.activePlatformGuid (internal)
        static MethodInfo _bpGuidEmptyMethod;                // UnityEditor.GUID.Empty()
        static Type _bpGuidType;
        static ConstructorInfo _bpGuidStringCtor;

        static bool ResolveBuildProfileApi()
        {
            if (_bpResolved) return _bpContextInstance != null;
            _bpResolved = true;

            try
            {
                var bpAsm = typeof(BuildProfile).Assembly;
                var eubsAsm = typeof(EditorUserBuildSettings).Assembly;

                // UnityEditor.GUID lives in the editor assembly.
                _bpGuidType = eubsAsm.GetType("UnityEditor.GUID")
                              ?? bpAsm.GetType("UnityEditor.GUID");
                _bpGuidStringCtor = _bpGuidType?.GetConstructor(new[] { typeof(string) });

                _bpContextType = bpAsm.GetType("UnityEditor.Build.Profile.BuildProfileContext");
                if (_bpContextType == null)
                {
                    Debug.LogWarning("[RoomScan Setup] BuildProfileContext type not found.");
                    return false;
                }

                var instanceProp = _bpContextType.GetProperty("instance",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
                _bpContextInstance = instanceProp?.GetValue(null);
                if (_bpContextInstance == null)
                {
                    Debug.LogWarning("[RoomScan Setup] BuildProfileContext.instance not accessible.");
                    return false;
                }

                // GetForClassicPlatform(GUID) — instance, internal.
                if (_bpGuidType != null)
                {
                    _bpGetForClassicPlatformByGuid = _bpContextType.GetMethod(
                        "GetForClassicPlatform",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                        null, new[] { _bpGuidType }, null);
                }

                // EditorUserBuildSettings.SwitchActiveBuildTargetGuid(BuildProfile)
                // — internal static. This is the one the editor itself
                // calls from BuildProfile.OnValidate / BuildProfileWindow.
                _bpSwitchActiveByProfile = typeof(EditorUserBuildSettings).GetMethod(
                    "SwitchActiveBuildTargetGuid",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                    null, new[] { typeof(BuildProfile) }, null);

                // BuildProfileModuleUtil.SwitchLegacyActiveFromBuildProfile(BuildProfile)
                // is the friendlier wrapper exposed in 6000.2; in newer
                // builds it may move or change visibility.
                var moduleUtilType = bpAsm.GetType("UnityEditor.Build.Profile.BuildProfileModuleUtil");
                _bpModuleUtilSwitchLegacy = moduleUtilType?.GetMethod(
                    "SwitchLegacyActiveFromBuildProfile",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                    null, new[] { typeof(BuildProfile) }, null);

                _bpActivePlatformGuidProp = typeof(EditorUserBuildSettings).GetProperty(
                    "activePlatformGuid",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

                _bpGuidEmptyMethod = _bpGuidType?.GetMethod(
                    "Empty",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);

                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] BuildProfile API resolve failed: {ex.Message}");
                _bpContextInstance = null;
                return false;
            }
        }

        // Returns a reflected UnityEditor.GUID for the Meta Quest derived
        // platform. Returns null if the GUID type isn't available.
        static object MetaQuestGuid()
        {
            if (_bpGuidStringCtor == null) return null;
            try { return _bpGuidStringCtor.Invoke(new object[] { k_MetaQuestPlatformGuid }); }
            catch { return null; }
        }

        /// <summary>
        /// Returns the auto-generated classic Meta Quest BuildProfile if
        /// Unity has registered one (Unity 6.1+ with Android module), or
        /// null otherwise.
        /// </summary>
        static BuildProfile FindMetaQuestClassicProfile()
        {
            if (!ResolveBuildProfileApi()) return null;

            try
            {
                var guid = MetaQuestGuid();
                if (guid != null && _bpGetForClassicPlatformByGuid != null)
                {
                    var profile = _bpGetForClassicPlatformByGuid.Invoke(_bpContextInstance, new[] { guid }) as BuildProfile;
                    if (profile != null) return profile;
                }

                // Last-ditch fallback: scan classicPlatformProfiles by
                // platformGuid (covers obscure setups where the GUID-keyed
                // dictionary isn't populated yet).
                var classicsProp = _bpContextType.GetProperty("classicPlatformProfiles",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (classicsProp?.GetValue(_bpContextInstance) is IEnumerable classics)
                {
                    var guidStrProp = typeof(BuildProfile).GetProperty("platformGuid",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    foreach (var item in classics)
                    {
                        if (item is not BuildProfile p) continue;
                        var g = guidStrProp?.GetValue(p)?.ToString();
                        if (string.Equals(g, k_MetaQuestPlatformGuid, StringComparison.OrdinalIgnoreCase))
                            return p;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] Meta Quest profile lookup failed: {ex.Message}");
            }
            return null;
        }

        /// <summary>
        /// True if the Meta Quest classic platform is currently the active
        /// platform/profile selection in the Build Profiles window.
        /// </summary>
        static bool IsActiveProfileMetaQuest()
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android) return false;
            if (!ResolveBuildProfileApi() || _bpActivePlatformGuidProp == null) return false;

            try
            {
                var activeGuid = _bpActivePlatformGuidProp.GetValue(null);
                return string.Equals(activeGuid?.ToString(), k_MetaQuestPlatformGuid, StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Activates the classic Meta Quest build profile via the internal
        /// classic-platform switching API. Returns true on success (a
        /// domain reload will follow). If no Meta Quest profile is
        /// registered or the API isn't accessible, returns false so
        /// callers can fall back to plain Android.
        /// </summary>
        static bool TryActivateMetaQuestProfile()
        {
            if (!ResolveBuildProfileApi()) return false;

            var profile = FindMetaQuestClassicProfile();
            if (profile == null)
            {
                // Profile dictionary is populated lazily on first access to
                // the BuildProfileContext UI. Touch a few public APIs to
                // trigger lazy creation, then retry once.
                try { _ = BuildProfile.GetActiveBuildProfile(); } catch { /* ignore */ }
                profile = FindMetaQuestClassicProfile();
            }
            if (profile == null) return false;

            // Prefer the public BuildProfileModuleUtil wrapper when it's
            // available — it's what Unity's own UI calls.
            try
            {
                if (_bpModuleUtilSwitchLegacy != null)
                {
                    _bpModuleUtilSwitchLegacy.Invoke(null, new object[] { profile });
                    Debug.Log("[RoomScan Setup] Activated Meta Quest classic build profile " +
                              "(via BuildProfileModuleUtil.SwitchLegacyActiveFromBuildProfile).");
                    return true;
                }

                if (_bpSwitchActiveByProfile != null)
                {
                    var ok = _bpSwitchActiveByProfile.Invoke(null, new object[] { profile });
                    bool success = ok is bool b ? b : true;
                    if (success)
                    {
                        Debug.Log("[RoomScan Setup] Activated Meta Quest classic build profile " +
                                  "(via EditorUserBuildSettings.SwitchActiveBuildTargetGuid).");
                        return true;
                    }
                    Debug.LogWarning("[RoomScan Setup] SwitchActiveBuildTargetGuid returned false.");
                    return false;
                }

                Debug.LogWarning("[RoomScan Setup] No internal API available to activate the " +
                                 "Meta Quest classic profile (Unity " + Application.unityVersion + ").");
                return false;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[RoomScan Setup] Meta Quest profile activation failed: " +
                                 (ex.InnerException?.Message ?? ex.Message));
                return false;
            }
        }
    }
}
