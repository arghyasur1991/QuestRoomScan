// Wizard partial: VR PROJECT BOOTSTRAP section.
//
// Purely presentation layer — all logic lives in VRProjectBootstrap.cs so it
// can also be invoked from CI / menu items / other tooling later.

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        // Cached audit snapshot. Refreshed on the wizard's heartbeat so the
        // counts in the section header stay live as the user fixes things via
        // other paths (Project Settings, Meta tool, etc.).
        readonly List<VRCheck> _vrOutstanding = new();
        readonly List<VRCheck> _vrRecommended = new();
        readonly List<VRCheck> _vrOk          = new();

        bool _vrFoldOutstanding = true;
        bool _vrFoldRecommended = false;
        bool _vrFoldOk          = false;

        bool _vrFixInProgress;

        // Materialize OpenXR settings assets exactly once per wizard session
        // so the on-tick recategorization is cheap (no AssetDatabase writes).
        static bool _vrAuditedOnce;

        partial void RefreshVRProject()
        {
            if (!_vrAuditedOnce)
            {
                _vrAuditedOnce = true;
                VRProjectBootstrap.Audit();
            }

            _vrOutstanding.Clear();
            _vrRecommended.Clear();
            _vrOk.Clear();

            foreach (var check in VRProjectBootstrap.AllChecks)
            {
                bool ok;
                try { ok = check.IsOk(); }
                catch (System.Exception ex)
                {
                    Debug.LogWarning($"[VR Bootstrap] check '{check.Id}' threw {ex.GetType().Name}: {ex.Message}");
                    ok = false;
                }

                if (ok)                                      _vrOk.Add(check);
                else if (check.Severity == CheckSeverity.Outstanding) _vrOutstanding.Add(check);
                else                                         _vrRecommended.Add(check);
            }
        }

        partial void DrawVRProjectSection()
        {
            BeginSection("VR PROJECT BOOTSTRAP");

            EditorGUILayout.HelpBox(
                "Audits and fixes project-level VR config (XR Plug-in Management, OpenXR " +
                "features, PlayerSettings, OVRProjectConfig, Meta XR Project Setup Tool).\n" +
                "Generic to every Quest project — does not write per-game identity.",
                MessageType.Info);

            // Top-of-section status line
            int total = _vrOutstanding.Count + _vrRecommended.Count + _vrOk.Count;
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(12);
            var prev = GUI.color;
            GUI.color = _vrOutstanding.Count > 0 ? COL_MISS
                      : _vrRecommended.Count > 0 ? COL_WARN
                      : COL_OK;
            GUILayout.Label(
                $"{_vrOutstanding.Count} outstanding   {_vrRecommended.Count} recommended   {_vrOk.Count} ok   ({total} total)",
                EditorStyles.boldLabel);
            GUI.color = prev;
            EditorGUILayout.EndHorizontal();

            // Buttons
            GUILayout.Space(4);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(12);

            using (new EditorGUI.DisabledScope(_vrFixInProgress))
            {
                if (GUILayout.Button("Audit Project", GUILayout.Height(22)))
                {
                    VRProjectBootstrap.Audit();
                    RefreshVRProject();
                }

                using (new EditorGUI.DisabledScope(_vrOutstanding.Count == 0))
                {
                    if (GUILayout.Button("Fix All Outstanding", GUILayout.Height(22)))
                        _ = RunFixAsync(CheckSeverity.Outstanding);
                }

                if (GUILayout.Button("Fix All", GUILayout.Height(22)))
                    _ = RunFixAsync(CheckSeverity.Recommended);
            }

            GUILayout.Space(12);
            EditorGUILayout.EndHorizontal();

            if (_vrFixInProgress)
            {
                EditorGUILayout.HelpBox("Fixing in progress (including Meta XR Project Setup Tool sweep)…",
                    MessageType.Info);
            }

            // Foldouts
            GUILayout.Space(6);
            DrawVRFold(ref _vrFoldOutstanding, "Outstanding", _vrOutstanding, COL_MISS);
            DrawVRFold(ref _vrFoldRecommended, "Recommended", _vrRecommended, COL_WARN);
            DrawVRFold(ref _vrFoldOk,          "OK",          _vrOk,          COL_OK);

            EndSection();
        }

        async System.Threading.Tasks.Task RunFixAsync(CheckSeverity sev)
        {
            _vrFixInProgress = true;
            Repaint();
            try
            {
                await VRProjectBootstrap.FixAllAsync(sev);
            }
            finally
            {
                _vrFixInProgress = false;
                RefreshVRProject();
                Repaint();
            }
        }

        void DrawVRFold(ref bool open, string title, List<VRCheck> items, Color tint)
        {
            string header = $"{title} ({items.Count})";
            open = EditorGUILayout.Foldout(open, header, true, EditorStyles.foldoutHeader);
            if (!open || items.Count == 0) return;

            EditorGUI.indentLevel++;
            foreach (var c in items)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Space(20);

                var prev = GUI.color;
                GUI.color = tint;
                bool ok = items == _vrOk;
                GUILayout.Label(ok ? "\u2713" : (c.Severity == CheckSeverity.Outstanding ? "\u2717" : "!"),
                    EditorStyles.boldLabel, GUILayout.Width(16));
                GUI.color = prev;

                GUILayout.Label(new GUIContent(c.Label, BuildTooltip(c)),
                    GUILayout.ExpandWidth(true));

                if (!ok)
                {
                    var fixLabel = c.Fix == null ? "(manual)" : "current → target";
                    GUILayout.Label(fixLabel, EditorStyles.miniLabel, GUILayout.Width(110));
                }

                EditorGUILayout.EndHorizontal();

                if (!ok)
                {
                    EditorGUILayout.BeginHorizontal();
                    GUILayout.Space(40);
                    string current = SafeRead(c.CurrentValue);
                    EditorGUILayout.LabelField($"current: {current}", EditorStyles.miniLabel);
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    GUILayout.Space(40);
                    EditorGUILayout.LabelField($"target:  {c.TargetValue}", EditorStyles.miniLabel);
                    EditorGUILayout.EndHorizontal();
                }
            }
            EditorGUI.indentLevel--;
            GUILayout.Space(2);
        }

        static string BuildTooltip(VRCheck c)
        {
            return $"{c.Id}\nseverity: {c.Severity}\ngroup: {c.Group}\ntarget: {c.TargetValue}";
        }

        static string SafeRead(System.Func<string> reader)
        {
            try { return reader?.Invoke() ?? "(null)"; }
            catch (System.Exception ex) { return $"<error: {ex.GetType().Name}>"; }
        }
    }
}
