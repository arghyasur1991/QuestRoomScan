using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Generates deterministic 256x256 procedural textures at runtime.
    /// Used as fallbacks when <see cref="ThemePack"/> texture fields are null.
    /// </summary>
    public static class ProceduralTextures
    {
        private const int Size = 256;

        public static Texture2D GenerateGrid(Color bg, Color line, int cellSize = 32)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                bool edge = x % cellSize == 0 || y % cellSize == 0;
                px[y * Size + x] = edge ? line : bg;
            }
            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        public static Texture2D GenerateCheckerboard(Color a, Color b, int cellSize = 32)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                bool check = ((x / cellSize) + (y / cellSize)) % 2 == 0;
                px[y * Size + x] = check ? a : b;
            }
            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        /// <summary>
        /// Horizontal/vertical circuit-like traces with deterministic pseudo-random
        /// branching.  Good for sci-fi wall patterns.
        /// </summary>
        public static Texture2D GenerateCircuitLines(Color bg, Color line)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            for (int i = 0; i < px.Length; i++) px[i] = bg;

            var rng = new System.Random(42);
            int traceCount = 18;
            for (int t = 0; t < traceCount; t++)
            {
                bool horizontal = rng.NextDouble() > 0.5;
                int pos = rng.Next(Size);
                int start = rng.Next(Size / 4);
                int end = start + Size / 4 + rng.Next(Size / 2);
                if (end > Size) end = Size;

                for (int i = start; i < end; i++)
                {
                    int x = horizontal ? i : pos;
                    int y = horizontal ? pos : i;
                    if (x >= 0 && x < Size && y >= 0 && y < Size)
                    {
                        px[y * Size + x] = line;
                        // thicken trace
                        if (horizontal && y + 1 < Size) px[(y + 1) * Size + x] = line;
                        if (!horizontal && x + 1 < Size) px[y * Size + x + 1] = line;
                    }

                    // branch
                    if (rng.NextDouble() < 0.03)
                    {
                        int branchLen = 8 + rng.Next(20);
                        int dir = rng.NextDouble() > 0.5 ? 1 : -1;
                        for (int b = 0; b < branchLen; b++)
                        {
                            int bx = horizontal ? x : x + dir * b;
                            int by = horizontal ? y + dir * b : y;
                            if (bx >= 0 && bx < Size && by >= 0 && by < Size)
                                px[by * Size + bx] = line;
                        }
                    }
                }

                // node dots at endpoints
                DrawDot(px, start, pos, horizontal, line, 2);
                DrawDot(px, end - 1, pos, horizontal, line, 2);
            }

            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        /// <summary>
        /// Perlin-noise thresholded vein pattern.  Good for organic/horror themes.
        /// </summary>
        public static Texture2D GenerateOrganicVeins(Color bg, Color vein)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            float scale = 6f;
            float offsetX = 137.5f, offsetY = 293.7f;

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float n1 = Mathf.PerlinNoise(x / (float)Size * scale + offsetX,
                                              y / (float)Size * scale + offsetY);
                float n2 = Mathf.PerlinNoise(x / (float)Size * scale * 2.5f + 50f,
                                              y / (float)Size * scale * 2.5f + 80f);
                float combined = n1 * 0.6f + n2 * 0.4f;
                float edge = Mathf.Abs(combined - 0.45f);
                float veinStrength = 1f - Mathf.Clamp01(edge * 12f);

                px[y * Size + x] = Color.Lerp(bg, vein, veinStrength);
            }
            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        /// <summary>
        /// Perlin noise glow map: bright areas on a dark background.
        /// </summary>
        public static Texture2D GenerateNoiseEmissive(Color bg, Color peak)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            float scale = 5f;
            float ox = 47.3f, oy = 83.1f;

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float n = Mathf.PerlinNoise(x / (float)Size * scale + ox,
                                             y / (float)Size * scale + oy);
                float glow = Mathf.Clamp01((n - 0.55f) * 4f);
                px[y * Size + x] = Color.Lerp(bg, peak, glow);
            }
            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        /// <summary>
        /// Water stains / mold damage pattern. Irregular dark patches on a clean
        /// background — good for horror/decay themes.
        /// </summary>
        public static Texture2D GenerateWaterStains(Color clean, Color stain)
        {
            var tex = Create();
            var px = new Color[Size * Size];
            float scale1 = 3.5f, scale2 = 8f, scale3 = 15f;
            float ox1 = 211.7f, oy1 = 167.3f;
            float ox2 = 53.1f, oy2 = 97.8f;
            float ox3 = 301.4f, oy3 = 412.9f;

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = x / (float)Size;
                float v = y / (float)Size;

                // Large blotchy shapes
                float n1 = Mathf.PerlinNoise(u * scale1 + ox1, v * scale1 + oy1);
                // Medium detail
                float n2 = Mathf.PerlinNoise(u * scale2 + ox2, v * scale2 + oy2);
                // Fine grain
                float n3 = Mathf.PerlinNoise(u * scale3 + ox3, v * scale3 + oy3);

                float combined = n1 * 0.5f + n2 * 0.3f + n3 * 0.2f;

                // Threshold into irregular patches with soft edges
                float stainMask = Mathf.Clamp01((combined - 0.38f) * 4f);
                // Add drip streaks (vertical bias)
                float drip = Mathf.PerlinNoise(u * 20f + 77f, v * 3f + 33f);
                drip = Mathf.Clamp01((drip - 0.55f) * 6f) * 0.4f;

                float totalMask = Mathf.Clamp01(stainMask + drip);
                px[y * Size + x] = Color.Lerp(clean, stain, totalMask);
            }

            tex.SetPixels(px);
            tex.Apply(true);
            tex.wrapMode = TextureWrapMode.Repeat;
            return tex;
        }

        private static Texture2D Create()
        {
            return new Texture2D(Size, Size, TextureFormat.RGBA32, true)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Repeat
            };
        }

        private static void DrawDot(Color[] px, int coord, int fixedCoord,
            bool horizontal, Color col, int radius)
        {
            int cx = horizontal ? coord : fixedCoord;
            int cy = horizontal ? fixedCoord : coord;
            for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
            {
                int x = cx + dx, y = cy + dy;
                if (x >= 0 && x < Size && y >= 0 && y < Size)
                    px[y * Size + x] = col;
            }
        }
    }
}
