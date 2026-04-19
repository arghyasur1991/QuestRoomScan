// VR Project Bootstrap orchestrator.
//
// Audits and fixes project-level VR config that the rest of the
// RoomScanSetupWizard does not handle (PlayerSettings, XR Plug-in Management
// loaders, OpenXR features and interaction profiles, OVRProjectConfig,
// OVRRuntimeSettings, and a final sweep through Meta's own
// OVRProjectSetup.FixAllAsync).
//
// Scope is intentionally generic: nothing here writes per-game identity
// (productName, companyName, applicationIdentifier). Those remain
// flagged-but-not-fixed.
//
// All fixes are idempotent: rerunning Audit + Fix All is safe.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

namespace Genesis.RoomScan.Editor
{
    internal enum CheckSeverity { Outstanding, Recommended }
    internal enum CheckResult { Ok, Failing, NotFixable }

    internal sealed class VRCheck
    {
        public string Id;
        public string Label;
        public CheckSeverity Severity;
        public BuildTargetGroup Group;
        public Func<string> CurrentValue;
        public string TargetValue;
        public Func<bool> IsOk;
        public Action Fix;
    }

    internal static class VRProjectBootstrap
    {
        // OpenXR feature ids — kept here so we don't take a hard reference on
        // internal types just to read constants. Mismatch with the SDK is
        // caught at audit-time by FeatureHelpers returning null.
        const string FID_META_XR             = "com.meta.openxr.feature.metaxr";
        const string FID_META_FOVEATION      = "com.meta.openxr.feature.foveation";
        const string FID_OCULUS_TOUCH        = "com.unity.openxr.feature.input.oculustouch";
        const string FID_QUEST_TOUCH_PLUS    = "com.unity.openxr.feature.input.metaquestplus";
        const string FID_QUEST_TOUCH_PRO     = "com.unity.openxr.feature.input.metaquestpro";
        const string FID_AR_CAMERA           = "com.unity.openxr.feature.arfoundation-meta-camera";
        const string FID_AR_OCCLUSION        = "com.unity.openxr.feature.arfoundation-meta-occlusion";
        const string FID_AR_SESSION          = "com.unity.openxr.feature.arfoundation-meta-session";

        const string OPENXR_LOADER_TYPE = "UnityEngine.XR.OpenXR.OpenXRLoader";

        // Android applicationIdentifier rules (per Android Studio docs):
        //   * at least two dot-separated segments
        //   * each segment starts with a letter
        //   * all characters are [a-zA-Z0-9_]
        // We deliberately accept uppercase letters — Pascal-cased segments
        // such as `com.MyCompany.MyGame` are perfectly legal app-ids even
        // though all-lowercase is more idiomatic on Android.
        static readonly Regex APPID_RE =
            new(@"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$", RegexOptions.Compiled);

        public static IReadOnlyList<VRCheck> AllChecks { get; } = BuildChecks();

        // -- Public API ---------------------------------------------------

        /// <summary>
        /// Re-evaluates every check. Cheap (just inspectors) so the wizard's
        /// 0.8s heartbeat can call this directly.
        /// </summary>
        public static void Audit()
        {
            // OpenXR settings asset is created lazily by RefreshFeatures, so
            // poke both groups before any check tries to read them. Without
            // this, brand-new projects return null from
            // OpenXRSettings.GetSettingsForBuildTargetGroup.
            FeatureHelpers.RefreshFeatures(BuildTargetGroup.Android);
            FeatureHelpers.RefreshFeatures(BuildTargetGroup.Standalone);
        }

        /// <summary>
        /// Runs Fix() on each check whose severity is at-or-above
        /// <paramref name="includeUpTo"/>. "Outstanding" runs only the
        /// hard-required ones; "Recommended" runs both tiers.
        /// </summary>
        public static async Task FixAllAsync(CheckSeverity includeUpTo)
        {
            if (Application.isPlaying)
            {
                Debug.LogError("[VR Bootstrap] Cannot fix while Play mode is active.");
                return;
            }

            Audit();

            // Build-target switch must happen first AND alone — it triggers a
            // domain reload that wipes the rest of this loop's continuation.
            // Bail out after switching so the user re-clicks Fix All and the
            // remaining checks evaluate against the new target.
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
            {
                Debug.Log("[VR Bootstrap] Switching active build target \u2192 Android. " +
                          "Click Fix All again after the reload completes to apply the rest.");
                EditorUserBuildSettings.SwitchActiveBuildTarget(
                    BuildTargetGroup.Android, BuildTarget.Android);
                return;
            }

            int fixed_ = 0, skipped = 0;
            foreach (var check in AllChecks)
            {
                if (check.Severity == CheckSeverity.Recommended &&
                    includeUpTo == CheckSeverity.Outstanding)
                    continue;

                if (check.Fix == null) { skipped++; continue; }

                if (check.IsOk()) continue;

                try
                {
                    check.Fix();
                    fixed_++;
                    Debug.Log($"[VR Bootstrap] Fixed: {check.Id} ({check.Label})");
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[VR Bootstrap] Fix failed for {check.Id}: {ex.Message}\n{ex.StackTrace}");
                }
            }

            // Meta's own catch-all sweep — best handled last so its tasks see
            // the loaders / features we just enabled.
            if (includeUpTo == CheckSeverity.Recommended)
            {
                try
                {
                    await OVRProjectSetup.FixAllAsync(BuildTargetGroup.Android);
                    Debug.Log("[VR Bootstrap] Meta XR Project Setup Tool: ran FixAllAsync(Android).");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[VR Bootstrap] OVRProjectSetup.FixAllAsync raised {ex.GetType().Name}: {ex.Message}");
                }
            }

            AssetDatabase.SaveAssets();
            Debug.Log($"[VR Bootstrap] Done. fixed={fixed_}, skipped(no-fix)={skipped}.");
        }

        // -- Check registry ----------------------------------------------

        static List<VRCheck> BuildChecks()
        {
            var list = new List<VRCheck>
            {
                // === Outstanding ===
                // Active build target must be Android — the entire OpenXR
                // feature configuration is meaningless until you switch.
                // Switching triggers a domain reload that aborts FixAllAsync
                // mid-loop, so the orchestrator special-cases this one and
                // returns early after the switch (user re-clicks Fix All).
                new VRCheck {
                    Id = "android.buildtarget.active",
                    Label = "Active build target = Android (Quest)",
                    Severity = CheckSeverity.Outstanding,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => EditorUserBuildSettings.activeBuildTarget.ToString(),
                    TargetValue = "Android",
                    IsOk = () => EditorUserBuildSettings.activeBuildTarget == BuildTarget.Android,
                    Fix = () => {
                        EditorUserBuildSettings.SwitchActiveBuildTarget(
                            BuildTargetGroup.Android, BuildTarget.Android);
                    },
                },

                MakeXRLoaderCheck(BuildTargetGroup.Android, CheckSeverity.Outstanding,
                    id: "android.xr.loader.openxr",
                    label: "Android: OpenXR loader assigned"),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Android, CheckSeverity.Outstanding,
                    id: "android.openxr.metaxr",
                    label: "Android: Meta XR feature enabled",
                    featureId: FID_META_XR),

                new VRCheck {
                    Id = "android.openxr.touchprofile",
                    Label = "Android: at least one Touch controller profile enabled",
                    Severity = CheckSeverity.Outstanding,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => DescribeTouchProfiles(BuildTargetGroup.Android),
                    TargetValue = ">= 1 of (Oculus Touch, Meta Quest Touch Plus, Meta Quest Touch Pro)",
                    IsOk = () => AnyFeatureEnabled(BuildTargetGroup.Android,
                                  FID_OCULUS_TOUCH, FID_QUEST_TOUCH_PLUS, FID_QUEST_TOUCH_PRO),
                    Fix = () => {
                        SetFeatureEnabled(BuildTargetGroup.Android, FID_OCULUS_TOUCH, true);
                        SetFeatureEnabled(BuildTargetGroup.Android, FID_QUEST_TOUCH_PLUS, true);
                        SetFeatureEnabled(BuildTargetGroup.Android, FID_QUEST_TOUCH_PRO, true);
                    },
                },

                MakeScriptingBackendCheck(),
                MakeArm64Check(),
                MakeMinSdkCheck(),
                MakeTargetSdkCheck(),
                MakeVulkanOnlyCheck(),
                MakeStereoRenderingCheck(),
                MakeTextureCompressionAstcCheck(),

                new VRCheck {
                    Id = "android.player.appid.set",
                    Label = "Android: applicationIdentifier is a valid per-game id (manual)",
                    Severity = CheckSeverity.Outstanding,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => PlayerSettings.applicationIdentifier ?? "(null)",
                    TargetValue = "valid Android app-id (com.<Company>.<Product>) and != com.DefaultCompany.unityProject",
                    IsOk = () => {
                        var id = PlayerSettings.applicationIdentifier ?? "";
                        return APPID_RE.IsMatch(id) && id != "com.DefaultCompany.unityProject";
                    },
                    Fix = null,
                },

                // === Recommended ===
                MakeXRLoaderCheck(BuildTargetGroup.Standalone, CheckSeverity.Recommended,
                    id: "standalone.xr.loader.openxr",
                    label: "Standalone: OpenXR loader assigned (Quest Link / PCVR dev loop)"),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Standalone, CheckSeverity.Recommended,
                    id: "standalone.openxr.touch",
                    label: "Standalone: Oculus Touch profile enabled",
                    featureId: FID_OCULUS_TOUCH),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Android, CheckSeverity.Recommended,
                    id: "android.openxr.foveation",
                    label: "Android: Meta XR Foveation enabled",
                    featureId: FID_META_FOVEATION),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Android, CheckSeverity.Recommended,
                    id: "android.openxr.passthrough",
                    label: "Android: Meta Quest Camera (Passthrough) feature enabled",
                    featureId: FID_AR_CAMERA),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Android, CheckSeverity.Recommended,
                    id: "android.openxr.session",
                    label: "Android: Meta Quest Session feature enabled",
                    featureId: FID_AR_SESSION),

                MakeOpenXRFeatureCheck(BuildTargetGroup.Android, CheckSeverity.Recommended,
                    id: "android.openxr.occlusion",
                    label: "Android: Meta Quest Occlusion feature enabled",
                    featureId: FID_AR_OCCLUSION),

                new VRCheck {
                    Id = "ovr.projectconfig.quest3",
                    Label = "OVRProjectConfig: target devices include Quest3 (and Quest3S)",
                    Severity = CheckSeverity.Recommended,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => {
                        var c = OVRProjectConfig.CachedProjectConfig;
                        return c == null ? "(no config)" :
                            string.Join(",", c.targetDeviceTypes.Select(t => t.ToString()));
                    },
                    TargetValue = "include Quest3, Quest3S",
                    IsOk = () => {
                        var c = OVRProjectConfig.CachedProjectConfig;
                        return c != null
                            && c.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest3)
                            && c.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest3S);
                    },
                    Fix = () => {
                        var c = OVRProjectConfig.CachedProjectConfig;
                        if (c == null) return;
                        if (!c.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest3))
                            c.targetDeviceTypes.Add(OVRProjectConfig.DeviceType.Quest3);
                        if (!c.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest3S))
                            c.targetDeviceTypes.Add(OVRProjectConfig.DeviceType.Quest3S);
                        OVRProjectConfig.CommitProjectConfig(c);
                    },
                },

                MakeOvrConfigEnumCheck(
                    id: "ovr.projectconfig.handtracking",
                    label: "OVRProjectConfig: hand tracking enabled (controllers + hands)",
                    read: c => c.handTrackingSupport.ToString(),
                    isOk: c => c.handTrackingSupport != OVRProjectConfig.HandTrackingSupport.ControllersOnly,
                    fix:  c => c.handTrackingSupport = OVRProjectConfig.HandTrackingSupport.ControllersAndHands,
                    target: ">= ControllersAndHands"),

                MakeOvrConfigEnumCheck(
                    id: "ovr.projectconfig.anchors",
                    label: "OVRProjectConfig: spatial anchor support enabled",
                    read: c => c.anchorSupport.ToString(),
                    isOk: c => c.anchorSupport != OVRProjectConfig.AnchorSupport.Disabled,
                    fix:  c => c.anchorSupport = OVRProjectConfig.AnchorSupport.Enabled,
                    target: "Enabled"),

                MakeOvrConfigEnumCheck(
                    id: "ovr.projectconfig.scene",
                    label: "OVRProjectConfig: scene support",
                    read: c => c.sceneSupport.ToString(),
                    isOk: c => c.sceneSupport >= OVRProjectConfig.FeatureSupport.Supported,
                    fix:  c => c.sceneSupport = OVRProjectConfig.FeatureSupport.Supported,
                    target: ">= Supported"),

                MakeOvrConfigEnumCheck(
                    id: "ovr.projectconfig.passthrough",
                    label: "OVRProjectConfig: insight passthrough support",
                    read: c => c.insightPassthroughSupport.ToString(),
                    isOk: c => c.insightPassthroughSupport >= OVRProjectConfig.FeatureSupport.Supported,
                    fix:  c => c.insightPassthroughSupport = OVRProjectConfig.FeatureSupport.Supported,
                    target: ">= Supported"),

                new VRCheck {
                    Id = "meta.runtime.settings.preloaded",
                    Label = "Meta XR runtime settings asset exists",
                    Severity = CheckSeverity.Recommended,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => OVRRuntimeSettings.GetRuntimeSettings() != null
                        ? "exists" : "missing",
                    TargetValue = "exists",
                    IsOk = () => OVRRuntimeSettings.GetRuntimeSettings() != null,
                    Fix = () => {
                        // GetRuntimeSettings auto-creates Resources/OculusRuntimeSettings.asset
                        // on the editor side via LoadAsset's create-fallback path.
                        var s = OVRRuntimeSettings.GetRuntimeSettings();
                        if (s != null) EditorUtility.SetDirty(s);
                    },
                },

                new VRCheck {
                    Id = "meta.setuptool.fixall",
                    Label = "Meta XR Project Setup Tool: run Fix All Outstanding (catch-all)",
                    Severity = CheckSeverity.Recommended,
                    Group = BuildTargetGroup.Android,
                    CurrentValue = () => "(synthetic — always shown)",
                    TargetValue = "FixAllAsync(Android) ran in this session",
                    IsOk = () => false, // synthetic catch-all; intentionally always re-runnable
                    Fix = null,         // invoked by FixAllAsync orchestrator after per-check loop
                },
            };

            return list;
        }

        // -- Region: PlayerSettings checks --------------------------------
        #region PlayerSettings

        static VRCheck MakeScriptingBackendCheck() => new()
        {
            Id = "android.player.scripting.il2cpp",
            Label = "Android: scripting backend = IL2CPP",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => PlayerSettings.GetScriptingBackend(NamedBuildTarget.Android).ToString(),
            TargetValue = "IL2CPP",
            IsOk = () => PlayerSettings.GetScriptingBackend(NamedBuildTarget.Android)
                         == ScriptingImplementation.IL2CPP,
            Fix = () => PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android,
                         ScriptingImplementation.IL2CPP),
        };

        static VRCheck MakeArm64Check() => new()
        {
            Id = "android.player.arch.arm64",
            Label = "Android: target architectures = ARM64 only",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => PlayerSettings.Android.targetArchitectures.ToString(),
            TargetValue = "ARM64",
            IsOk = () => PlayerSettings.Android.targetArchitectures == AndroidArchitecture.ARM64,
            Fix = () => PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64,
        };

        static VRCheck MakeMinSdkCheck() => new()
        {
            Id = "android.player.minsdk.29",
            Label = "Android: min SDK >= 29 (Quest 3 floor)",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => ((int)PlayerSettings.Android.minSdkVersion).ToString(),
            TargetValue = ">= 29",
            IsOk = () => (int)PlayerSettings.Android.minSdkVersion >= 29,
            Fix = () => PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29,
        };

        static VRCheck MakeTargetSdkCheck() => new()
        {
            Id = "android.player.targetsdk.32plus",
            Label = "Android: target SDK >= 32 (Meta-recommended)",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => {
                var t = PlayerSettings.Android.targetSdkVersion;
                return t == AndroidSdkVersions.AndroidApiLevelAuto ? "Auto" : ((int)t).ToString();
            },
            TargetValue = ">= 32 (or Auto, which Unity now resolves to >= 33)",
            IsOk = () => {
                var t = PlayerSettings.Android.targetSdkVersion;
                return t == AndroidSdkVersions.AndroidApiLevelAuto || (int)t >= 32;
            },
            Fix = () => PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto,
        };

        static VRCheck MakeVulkanOnlyCheck() => new()
        {
            Id = "android.player.gfxapi.vulkan",
            Label = "Android: graphics API = Vulkan only (auto APIs disabled)",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => {
                var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
                var prefix = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android) ? "Auto: " : "";
                return prefix + (apis.Length == 0 ? "(none)" : string.Join(",", apis));
            },
            TargetValue = "[Vulkan] only, automatic=false",
            IsOk = () => {
                if (PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android)) return false;
                var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
                return apis.Length == 1 && apis[0] == GraphicsDeviceType.Vulkan;
            },
            Fix = () => {
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android,
                    new[] { GraphicsDeviceType.Vulkan });
            },
        };

        static VRCheck MakeStereoRenderingCheck() => new()
        {
            Id = "android.player.stereo.multiview",
            Label = "Android: stereo rendering = Single Pass Instanced (multi-view)",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => PlayerSettings.stereoRenderingPath.ToString(),
            TargetValue = "Instancing (Single Pass Instanced)",
            IsOk = () => PlayerSettings.stereoRenderingPath == StereoRenderingPath.Instancing,
            Fix = () => PlayerSettings.stereoRenderingPath = StereoRenderingPath.Instancing,
        };

        // Quest GPUs (Adreno) only have hardware decoders for ASTC; ETC2 falls
        // back to software decode which inflates VRAM and hurts perf. Meta's
        // own docs and the Build Profiles "Meta Quest" preset both ship with
        // ASTC as the only Android texture compression format.
        static VRCheck MakeTextureCompressionAstcCheck() => new()
        {
            Id = "android.player.texcompression.astc",
            Label = "Android: texture compression = ASTC only",
            Severity = CheckSeverity.Outstanding,
            Group = BuildTargetGroup.Android,
            CurrentValue = () =>
            {
                var fmts = PlayerSettings.Android.textureCompressionFormats;
                return fmts == null || fmts.Length == 0
                    ? "(none)"
                    : string.Join(",", fmts);
            },
            TargetValue = "[ASTC] only",
            IsOk = () =>
            {
                var fmts = PlayerSettings.Android.textureCompressionFormats;
                return fmts != null && fmts.Length == 1
                       && fmts[0] == TextureCompressionFormat.ASTC;
            },
            Fix = () => PlayerSettings.Android.textureCompressionFormats =
                        new[] { TextureCompressionFormat.ASTC },
        };

        #endregion

        // -- Region: XR Plug-in Management loaders ------------------------
        #region XRLoaders

        static VRCheck MakeXRLoaderCheck(
            BuildTargetGroup group, CheckSeverity sev, string id, string label) => new()
        {
            Id = id,
            Label = label,
            Severity = sev,
            Group = group,
            CurrentValue = () => DescribeLoaders(group),
            TargetValue = "OpenXRLoader",
            IsOk = () => GetXRManager(group)?.activeLoaders?.Any(l => l != null
                && l.GetType().FullName == OPENXR_LOADER_TYPE) == true,
            Fix = () => EnsureOpenXRLoader(group),
        };

        static string DescribeLoaders(BuildTargetGroup group)
        {
            var mgr = GetXRManager(group);
            if (mgr == null) return "(no XR settings)";
            var loaders = mgr.activeLoaders;
            if (loaders == null || loaders.Count == 0) return "(none)";
            return string.Join(",", loaders.Where(l => l != null).Select(l => l.GetType().Name));
        }

        static XRManagerSettings GetXRManager(BuildTargetGroup group)
        {
            var perTarget = GetOrCreateXRPerBuildTarget();
            if (perTarget == null) return null;
            var settings = perTarget.SettingsForBuildTarget(group);
            return settings == null ? null : settings.AssignedSettings;
        }

        // XRGeneralSettingsPerBuildTarget.GetOrCreate is internal — reflect.
        static XRGeneralSettingsPerBuildTarget _cachedPerTarget;
        static XRGeneralSettingsPerBuildTarget GetOrCreateXRPerBuildTarget()
        {
            // If something already created the asset (e.g. user opened the XR
            // Plug-in Management settings page) we'll find it via the existing
            // EditorBuildSettings config object first.
            if (EditorBuildSettings.TryGetConfigObject<XRGeneralSettingsPerBuildTarget>(
                    XRGeneralSettings.k_SettingsKey, out var existing) && existing != null)
            {
                _cachedPerTarget = existing;
                return existing;
            }

            if (_cachedPerTarget != null) return _cachedPerTarget;

            // Reflective access to the internal GetOrCreate so we don't need
            // to vendor a copy of its asset-creation logic.
            var t = typeof(XRGeneralSettingsPerBuildTarget);
            var m = t.GetMethod("GetOrCreate", BindingFlags.Static | BindingFlags.NonPublic);
            if (m == null)
            {
                Debug.LogError("[VR Bootstrap] XRGeneralSettingsPerBuildTarget.GetOrCreate not found via reflection — XR Plug-in Management package may have changed.");
                return null;
            }
            _cachedPerTarget = (XRGeneralSettingsPerBuildTarget)m.Invoke(null, null);
            return _cachedPerTarget;
        }

        static void EnsureOpenXRLoader(BuildTargetGroup group)
        {
            var perTarget = GetOrCreateXRPerBuildTarget();
            if (perTarget == null) return;

            if (!perTarget.HasManagerSettingsForBuildTarget(group))
                perTarget.CreateDefaultManagerSettingsForBuildTarget(group);

            var mgr = perTarget.SettingsForBuildTarget(group)?.AssignedSettings;
            if (mgr == null)
            {
                Debug.LogError($"[VR Bootstrap] Could not create XRManagerSettings for {group}.");
                return;
            }

            if (!XRPackageMetadataStore.AssignLoader(mgr, OPENXR_LOADER_TYPE, group))
            {
                Debug.LogWarning($"[VR Bootstrap] AssignLoader returned false for OpenXR on {group} (already present?)");
            }
        }

        #endregion

        // -- Region: OpenXR features --------------------------------------
        #region OpenXRFeatures

        static VRCheck MakeOpenXRFeatureCheck(
            BuildTargetGroup group, CheckSeverity sev,
            string id, string label, string featureId) => new()
        {
            Id = id,
            Label = label,
            Severity = sev,
            Group = group,
            CurrentValue = () => DescribeFeature(group, featureId),
            TargetValue = "enabled",
            IsOk = () => IsFeatureEnabled(group, featureId),
            Fix = () => SetFeatureEnabled(group, featureId, true),
        };

        static string DescribeFeature(BuildTargetGroup group, string featureId)
        {
            var f = FeatureHelpers.GetFeatureWithIdForBuildTarget(group, featureId);
            if (f == null) return "(feature not found)";
            return f.enabled ? "enabled" : "disabled";
        }

        static bool IsFeatureEnabled(BuildTargetGroup group, string featureId)
        {
            var f = FeatureHelpers.GetFeatureWithIdForBuildTarget(group, featureId);
            return f != null && f.enabled;
        }

        static bool AnyFeatureEnabled(BuildTargetGroup group, params string[] featureIds)
        {
            foreach (var fid in featureIds)
                if (IsFeatureEnabled(group, fid)) return true;
            return false;
        }

        static void SetFeatureEnabled(BuildTargetGroup group, string featureId, bool enabled)
        {
            // Make sure the settings asset and the feature instance exist
            // (RefreshFeatures materializes both when missing).
            FeatureHelpers.RefreshFeatures(group);
            var f = FeatureHelpers.GetFeatureWithIdForBuildTarget(group, featureId);
            if (f == null)
            {
                Debug.LogWarning($"[VR Bootstrap] OpenXR feature '{featureId}' not registered for {group} — package providing it may not be installed.");
                return;
            }
            if (f.enabled == enabled) return;
            f.enabled = enabled;
            EditorUtility.SetDirty(f);
        }

        static string DescribeTouchProfiles(BuildTargetGroup group)
        {
            var enabled = new List<string>();
            if (IsFeatureEnabled(group, FID_OCULUS_TOUCH))     enabled.Add("OculusTouch");
            if (IsFeatureEnabled(group, FID_QUEST_TOUCH_PLUS)) enabled.Add("MetaQuestTouchPlus");
            if (IsFeatureEnabled(group, FID_QUEST_TOUCH_PRO))  enabled.Add("MetaQuestTouchPro");
            return enabled.Count == 0 ? "(none)" : string.Join(",", enabled);
        }

        #endregion

        // -- Region: OVRProjectConfig ------------------------------------
        #region OVRProjectConfig

        static VRCheck MakeOvrConfigEnumCheck(
            string id, string label,
            Func<OVRProjectConfig, string> read,
            Func<OVRProjectConfig, bool> isOk,
            Action<OVRProjectConfig> fix,
            string target) => new()
        {
            Id = id,
            Label = label,
            Severity = CheckSeverity.Recommended,
            Group = BuildTargetGroup.Android,
            CurrentValue = () => {
                var c = OVRProjectConfig.CachedProjectConfig;
                return c == null ? "(no config)" : read(c);
            },
            TargetValue = target,
            IsOk = () => {
                var c = OVRProjectConfig.CachedProjectConfig;
                return c != null && isOk(c);
            },
            Fix = () => {
                var c = OVRProjectConfig.CachedProjectConfig;
                if (c == null) return;
                fix(c);
                OVRProjectConfig.CommitProjectConfig(c);
            },
        };

        #endregion
    }
}
