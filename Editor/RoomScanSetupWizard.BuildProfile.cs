// Activate Unity 6's Meta Quest *build profile* (not just plain Android),
// so the Build Profiles window reflects the same Meta Quest selection a
// user would pick by hand.
//
// Unity 6.1+ ships a derived "Meta Quest" classic platform on top of
// Android with its own player/quality overrides (Vulkan, IL2CPP, ARM64,
// MultiView, Quest-tuned quality level). The classic profile for it is
// auto-created by Unity in Library/BuildProfiles/ with
// `m_BuildSubtarget = 6` and reachable through
// `BuildProfileContext.classicPlatformProfiles`.
//
// IMPORTANT: `BuildProfile.SetActiveBuildProfile()` REJECTS classic
// platforms (Unity logs "Classic Platforms cannot be set as the active
// build profile."). Classic profiles must be activated through
// `BuildProfileContext.TrySelectInstalledClassicPlatformByGUID(...)`,
// which is internal. We drive both via reflection and gracefully fall
// back to plain Android if anything is unavailable.

using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.Build.Profile;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        // Cache reflected types/methods — the BuildProfile API surface is
        // chunky and most of it is internal. ResolveBuildProfileApi() is
        // called lazily and is a no-op once cached.
        static Type _bpContextType;
        static object _bpContextInstance;
        static PropertyInfo _bpClassicProfilesProp;
        static MethodInfo _bpDisplayNameMethod;
        static MethodInfo _bpIsClassicMethod;
        static MethodInfo _bpTrySelectClassicByGuidMethod;
        static MethodInfo _bpIsActiveBuildProfileOrPlatformMethod;
        static PropertyInfo _bpActiveProfileProp;
        static PropertyInfo _bpProfilePlatformGuidProp;
        static PropertyInfo _bpProfilePlatformIdProp;

        static bool ResolveBuildProfileApi()
        {
            if (_bpContextType != null) return true;

            try
            {
                var asm = typeof(BuildProfile).Assembly;
                _bpContextType = asm.GetType("UnityEditor.Build.Profile.BuildProfileContext");
                if (_bpContextType == null) return false;

                var instanceProp = _bpContextType.GetProperty("instance",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
                _bpContextInstance = instanceProp?.GetValue(null);
                if (_bpContextInstance == null) return false;

                _bpClassicProfilesProp = _bpContextType.GetProperty("classicPlatformProfiles",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                _bpDisplayNameMethod = _bpContextType.GetMethod("GetClassicPlatformDisplayName",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

                _bpIsClassicMethod = _bpContextType.GetMethod("IsClassicPlatformProfile",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

                _bpTrySelectClassicByGuidMethod = _bpContextType.GetMethod(
                    "TrySelectInstalledClassicPlatformByGUID",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                _bpIsActiveBuildProfileOrPlatformMethod = _bpContextType.GetMethod(
                    "IsActiveBuildProfileOrPlatform",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                    ?? _bpContextType.GetMethod("IsActiveBuildProfileOrPlatform",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                _bpActiveProfileProp = _bpContextType.GetProperty("activeProfile",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                _bpProfilePlatformGuidProp = typeof(BuildProfile).GetProperty("platformGuid",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                _bpProfilePlatformIdProp = typeof(BuildProfile).GetProperty("platformId",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] BuildProfile API resolve failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Returns the auto-generated classic Meta Quest BuildProfile if
        /// Unity has registered one (Unity 6.1+ with Android module), or
        /// null otherwise. Looks up by display name to avoid hard-coding
        /// the platform GUID, with a subtarget-based fallback.
        /// </summary>
        static BuildProfile FindMetaQuestClassicProfile()
        {
            if (!ResolveBuildProfileApi()) return null;

            try
            {
                if (_bpClassicProfilesProp?.GetValue(_bpContextInstance) is not IEnumerable classics)
                    return null;

                foreach (var item in classics)
                {
                    if (item is not BuildProfile profile) continue;

                    string name = null;
                    if (_bpDisplayNameMethod != null)
                    {
                        try { name = _bpDisplayNameMethod.Invoke(null, new object[] { profile }) as string; }
                        catch { /* ignore — fall through to subtarget probe */ }
                    }

                    if (!string.IsNullOrEmpty(name) &&
                        name.IndexOf("Meta Quest", StringComparison.OrdinalIgnoreCase) >= 0)
                        return profile;
                }

                // Display-name lookup failed (older API surface). Fall back
                // to picking the Android classic profile whose subtarget
                // is the Meta Quest one.
                foreach (var item in classics)
                {
                    if (item is BuildProfile p && IsMetaQuestProfileBySubtarget(p))
                        return p;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] Meta Quest profile lookup failed: {ex.Message}");
            }
            return null;
        }

        /// <summary>
        /// True if the given build profile is the Meta Quest derivative of
        /// Android (subtarget 6 in the auto-generated classic profile).
        /// </summary>
        static bool IsMetaQuestProfileBySubtarget(BuildProfile profile)
        {
            try
            {
                var so = new SerializedObject(profile);
                var bt = so.FindProperty("m_BuildTarget");
                if (bt == null || bt.intValue != (int)BuildTarget.Android) return false;

                var pbp = so.FindProperty("m_PlatformBuildProfile");
                if (pbp == null) return false;

                var sub = pbp.FindPropertyRelative("data.m_BuildSubtarget")
                          ?? pbp.FindPropertyRelative("m_BuildSubtarget");
                // 6 = MetaQuest in UnityEditor.Android.AndroidPlatformBuildSettings.
                return sub != null && sub.intValue == 6;
            }
            catch { return false; }
        }

        /// <summary>
        /// True if the Meta Quest classic platform is currently the active
        /// platform/profile selection in the Build Profiles window.
        /// </summary>
        static bool IsActiveProfileMetaQuest()
        {
            // Build target must at minimum be Android — quick reject.
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android) return false;

            if (!ResolveBuildProfileApi()) return false;

            try
            {
                var profile = FindMetaQuestClassicProfile();
                if (profile == null) return false;

                if (_bpIsActiveBuildProfileOrPlatformMethod != null)
                {
                    object result = null;
                    if (_bpIsActiveBuildProfileOrPlatformMethod.IsStatic)
                        result = _bpIsActiveBuildProfileOrPlatformMethod.Invoke(null, new object[] { profile });
                    else
                        result = _bpIsActiveBuildProfileOrPlatformMethod.Invoke(_bpContextInstance, new object[] { profile });

                    if (result is bool b) return b;
                }

                // Fallback 1: BuildProfileContext.activeProfile points at
                // the currently selected profile (classic or asset).
                if (_bpActiveProfileProp != null)
                {
                    var active = _bpActiveProfileProp.GetValue(_bpContextInstance) as BuildProfile;
                    if (active != null && active == profile) return true;
                }

                // Fallback 2: public API — only returns asset-based active
                // profiles, but covers the rare path where Unity surfaces
                // classic platforms through GetActiveBuildProfile too.
                var publicActive = BuildProfile.GetActiveBuildProfile();
                if (publicActive == profile) return true;
            }
            catch { /* ignore */ }

            return false;
        }

        /// <summary>
        /// Activates the classic Meta Quest build profile via the internal
        /// classic-platform selection API. Returns true on success (a
        /// domain reload will follow). If no Meta Quest profile is
        /// registered or the API isn't accessible, returns false so
        /// callers can fall back.
        /// </summary>
        static bool TryActivateMetaQuestProfile()
        {
            if (!ResolveBuildProfileApi()) return false;

            var profile = FindMetaQuestClassicProfile();
            if (profile == null) return false;

            // Pull the platform GUID off the profile — required for
            // TrySelectInstalledClassicPlatformByGUID(...).
            object platformGuidObj = null;
            try
            {
                platformGuidObj = _bpProfilePlatformGuidProp?.GetValue(profile);
                if (platformGuidObj == null)
                    platformGuidObj = _bpProfilePlatformIdProp?.GetValue(profile);
            }
            catch { /* ignore — fall through to SerializedObject path */ }

            // Last-ditch: read the GUID hex string out of the asset.
            string platformGuidString = null;
            if (platformGuidObj == null)
            {
                try
                {
                    var so = new SerializedObject(profile);
                    var pid = so.FindProperty("m_PlatformId");
                    if (pid != null)
                    {
                        if (pid.propertyType == SerializedPropertyType.Hash128)
                            platformGuidString = pid.hash128Value.ToString();
                        else if (pid.propertyType == SerializedPropertyType.String)
                            platformGuidString = pid.stringValue;
                    }
                }
                catch { /* ignore */ }
            }

            if (_bpTrySelectClassicByGuidMethod == null)
            {
                Debug.LogWarning("[RoomScan Setup] BuildProfileContext.TrySelectInstalledClassicPlatformByGUID not found.");
                return false;
            }

            // Coerce our GUID into the parameter type the method expects.
            var paramType = _bpTrySelectClassicByGuidMethod.GetParameters().FirstOrDefault()?.ParameterType;
            object arg = CoerceGuidArg(platformGuidObj, platformGuidString, paramType);
            if (arg == null)
            {
                Debug.LogWarning("[RoomScan Setup] Could not derive Meta Quest platform GUID for activation.");
                return false;
            }

            try
            {
                var result = _bpTrySelectClassicByGuidMethod.Invoke(_bpContextInstance, new[] { arg });
                bool ok = result is bool b ? b : true;
                if (ok)
                {
                    Debug.Log("[RoomScan Setup] Activated Meta Quest classic build profile.");
                    return true;
                }
                Debug.LogWarning("[RoomScan Setup] TrySelectInstalledClassicPlatformByGUID returned false.");
                return false;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] Meta Quest profile activation failed: " +
                                 (ex.InnerException?.Message ?? ex.Message));
                return false;
            }
        }

        // BuildProfile's platformGuid is a UnityEditor.GUID; m_PlatformId
        // serialises as Hash128. The selection method takes whichever the
        // build of Unity exposes — convert as needed.
        static object CoerceGuidArg(object guidObj, string guidString, Type paramType)
        {
            if (paramType == null) return guidObj ?? guidString;

            if (guidObj != null && paramType.IsInstanceOfType(guidObj)) return guidObj;
            if (paramType == typeof(string)) return guidObj?.ToString() ?? guidString;

            // Try constructing the parameter type from a string (works for
            // both UnityEditor.GUID and Hash128 — both have string ctors).
            try
            {
                var src = guidObj?.ToString() ?? guidString;
                if (string.IsNullOrEmpty(src)) return null;

                var ctor = paramType.GetConstructor(new[] { typeof(string) });
                if (ctor != null) return ctor.Invoke(new object[] { src });

                // UnityEditor.GUID has TryParse(string, out GUID).
                var tryParse = paramType.GetMethod("TryParse",
                    BindingFlags.Public | BindingFlags.Static,
                    null, new[] { typeof(string), paramType.MakeByRefType() }, null);
                if (tryParse != null)
                {
                    var args = new object[] { src, Activator.CreateInstance(paramType) };
                    var ok = (bool)tryParse.Invoke(null, args);
                    if (ok) return args[1];
                }
            }
            catch { /* ignore — caller handles null */ }

            return null;
        }
    }
}
