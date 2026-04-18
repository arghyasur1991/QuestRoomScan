// Activate Unity 6's Meta Quest *build profile* (not just plain Android),
// so the Build Profiles window reflects the same Meta Quest selection a
// user would pick by hand.
//
// Unity 6.1+ ships a derived "Meta Quest" platform on top of Android with
// its own player/quality overrides (Vulkan, IL2CPP, ARM64, MultiView,
// Quest-tuned quality level). The classic profile for it is auto-created
// by Unity in Library/BuildProfiles/ with `m_BuildSubtarget = 6` and
// reachable through `BuildProfileContext.classicPlatformProfiles`.
//
// `BuildProfile.SetActiveBuildProfile(profile)` is public; the lookup
// helpers around classic profiles are internal so we drive them via
// reflection. Falls back cleanly to plain Android if anything is missing
// (older Unity, Meta Quest module not installed, etc.).

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
        /// <summary>
        /// Returns the auto-generated classic Meta Quest BuildProfile if
        /// Unity has registered one (Unity 6.1+ with Android module), or
        /// null otherwise. Looks up by display name to avoid hard-coding
        /// the platform GUID.
        /// </summary>
        static BuildProfile FindMetaQuestClassicProfile()
        {
            try
            {
                var asm = typeof(BuildProfile).Assembly;
                var contextType = asm.GetType("UnityEditor.Build.Profile.BuildProfileContext");
                if (contextType == null) return null;

                // ScriptableSingleton<BuildProfileContext>.instance — exposed
                // both as a static property and via the base singleton helper.
                var instanceProp = contextType.GetProperty("instance",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
                var instance = instanceProp?.GetValue(null);
                if (instance == null) return null;

                var classicsProp = contextType.GetProperty("classicPlatformProfiles",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (classicsProp?.GetValue(instance) is not IEnumerable classics) return null;

                var displayMethod = contextType.GetMethod("GetClassicPlatformDisplayName",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

                foreach (var item in classics)
                {
                    if (item is not BuildProfile profile) continue;

                    string name = null;
                    if (displayMethod != null)
                    {
                        try { name = displayMethod.Invoke(null, new object[] { profile }) as string; }
                        catch { /* ignore — fall through to subtarget probe */ }
                    }

                    if (!string.IsNullOrEmpty(name) &&
                        name.IndexOf("Meta Quest", StringComparison.OrdinalIgnoreCase) >= 0)
                        return profile;
                }

                // Display-name lookup failed (older API surface). Fall back
                // to picking the Android classic profile whose subtarget
                // is *not* the default plain-Android one.
                foreach (var item in classics)
                {
                    if (item is not BuildProfile profile) continue;
                    if (IsMetaQuestProfileBySubtarget(profile)) return profile;
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
        /// True if the *active* build profile is the Meta Quest derivative.
        /// </summary>
        static bool IsActiveProfileMetaQuest()
        {
            try
            {
                var active = BuildProfile.GetActiveBuildProfile();
                return active != null && IsMetaQuestProfileBySubtarget(active);
            }
            catch { return false; }
        }

        /// <summary>
        /// Activates the classic Meta Quest build profile. Returns true on
        /// success (a domain reload will follow). If no Meta Quest profile
        /// is registered, returns false so callers can fall back.
        /// </summary>
        static bool TryActivateMetaQuestProfile()
        {
            var profile = FindMetaQuestClassicProfile();
            if (profile == null) return false;

            try
            {
                BuildProfile.SetActiveBuildProfile(profile);
                Debug.Log("[RoomScan Setup] Activated Meta Quest build profile.");
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[RoomScan Setup] SetActiveBuildProfile(Meta Quest) failed: {ex.Message}");
                return false;
            }
        }
    }
}
