#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Sequential model loading queue. ORT can crash when loading multiple
    /// InferenceSessions concurrently — this ensures one-at-a-time loading
    /// on a background thread. Modeled on LiveTalk's ModelUtils.Initialize loop.
    /// </summary>
    internal static class OrtLoadQueue
    {
        private static readonly Queue<(Task task, string name)> _queue = new();
        private static bool _running;

        internal static void Enqueue(Task coldTask, string modelName)
        {
            lock (_queue)
            {
                _queue.Enqueue((coldTask, modelName));
                if (!_running)
                {
                    _running = true;
                    Task.Run(ProcessLoop);
                }
            }
        }

        private static async Task ProcessLoop()
        {
            while (true)
            {
                (Task task, string name) item;
                lock (_queue)
                {
                    if (_queue.Count == 0)
                    {
                        _running = false;
                        return;
                    }
                    item = _queue.Dequeue();
                }

                try
                {
                    Logger.Info($"[OrtLoadQueue] Loading {item.name}...");
                    item.task.Start();
                    await item.task;
                    Logger.Info($"[OrtLoadQueue] Loaded {item.name}");
                }
                catch (Exception e)
                {
                    Logger.Error($"[OrtLoadQueue] Failed to load {item.name}: {e.Message}");
                }
            }
        }
    }
}
#endif
