// GSplat shared helpers — ported from msplat (Apache 2.0)
// Metal → HLSL translation of core Gaussian splatting primitives.

#ifndef GSPLAT_HELPERS_HLSL
#define GSPLAT_HELPERS_HLSL

#define GS_BLOCK_X 16
#define GS_BLOCK_Y 16
#define GS_BLOCK_SIZE (GS_BLOCK_X * GS_BLOCK_Y)
#define GS_CHANNELS 3

#define GS_MAX_TILE_ELEMS 1024
#define GS_SORT_TG_SIZE 512

static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
static const float SH_C3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

// SSIM constants
#define SSIM_WIN 11
#define SSIM_HALF_WIN 5
#define SSIM_C1 0.0001f
#define SSIM_C2 0.0009f

uint NumSHBases(uint degree)
{
    if (degree == 0) return 1;
    if (degree == 1) return 4;
    if (degree == 2) return 9;
    if (degree == 3) return 16;
    return 25;
}

float Ndc2Pix(float x, float W, float cx)
{
    return 0.5f * W * x + cx - 0.5f;
}

void GetBBox(float2 center, float2 dims, int3 imgSize,
             out uint2 bbMin, out uint2 bbMax)
{
    bbMin.x = (uint)min(max(0, (int)(center.x - dims.x)), imgSize.x);
    bbMax.x = (uint)min(max(0, (int)(center.x + dims.x + 1)), imgSize.x);
    bbMin.y = (uint)min(max(0, (int)(center.y - dims.y)), imgSize.y);
    bbMax.y = (uint)min(max(0, (int)(center.y + dims.y + 1)), imgSize.y);
}

void GetTileBBox(float2 pixCenter, float2 pixRadius, int3 tileBounds,
                 out uint2 tileMin, out uint2 tileMax)
{
    float2 tc = float2(pixCenter.x / (float)GS_BLOCK_X,
                       pixCenter.y / (float)GS_BLOCK_Y);
    float2 tr = float2(pixRadius.x / (float)GS_BLOCK_X,
                       pixRadius.y / (float)GS_BLOCK_Y);
    GetBBox(tc, tr, tileBounds, tileMin, tileMax);
}

float3 Transform4x3(float4x4 mat, float3 p)
{
    return float3(
        mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3],
        mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3],
        mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3]
    );
}

float4 Transform4x4(float4x4 mat, float3 p)
{
    return float4(
        mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3],
        mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3],
        mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3],
        mat[3][0]*p.x + mat[3][1]*p.y + mat[3][2]*p.z + mat[3][3]
    );
}

float3x3 QuatToRotmat(float4 q)
{
    float s = rsqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    float w = q.x * s;
    float x = q.y * s;
    float y = q.z * s;
    float z = q.w * s;

    return float3x3(
        1.f - 2.f*(y*y + z*z), 2.f*(x*y + w*z),       2.f*(x*z - w*y),
        2.f*(x*y - w*z),       1.f - 2.f*(x*x + z*z), 2.f*(y*z + w*x),
        2.f*(x*z + w*y),       2.f*(y*z - w*x),        1.f - 2.f*(x*x + y*y)
    );
}

// Build 3D covariance from scale + quaternion: cov = R*S*S^T*R^T
// Stores upper triangle (6 floats)
void ScaleRotToCov3D(float3 scale, float globScale, float4 quat, out float cov3d[6])
{
    float3x3 R = QuatToRotmat(quat);
    float3x3 S = float3x3(
        globScale * scale.x, 0, 0,
        0, globScale * scale.y, 0,
        0, 0, globScale * scale.z
    );
    float3x3 M = mul(R, S);
    float3x3 T = mul(M, transpose(M));

    cov3d[0] = T[0][0];
    cov3d[1] = T[0][1];
    cov3d[2] = T[0][2];
    cov3d[3] = T[1][1];
    cov3d[4] = T[1][2];
    cov3d[5] = T[2][2];
}

// Project 3D covariance to 2D via EWA splatting
float3 ProjectCov3DEWA(float cov3d[6], float4x4 viewmat,
                       float fx, float fy, float tanFovX, float tanFovY,
                       inout float3 pView)
{
    float limX = 1.3f * tanFovX;
    float limY = 1.3f * tanFovY;
    pView.x = pView.z * min(limX, max(-limX, pView.x / pView.z));
    pView.y = pView.z * min(limY, max(-limY, pView.y / pView.z));

    float rz = 1.f / pView.z;
    float rz2 = rz * rz;

    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * pView.x * rz2;
    float j21 = -fy * pView.y * rz2;

    float3 mr0 = float3(viewmat[0][0], viewmat[0][1], viewmat[0][2]);
    float3 mr1 = float3(viewmat[1][0], viewmat[1][1], viewmat[1][2]);
    float3 mr2 = float3(viewmat[2][0], viewmat[2][1], viewmat[2][2]);

    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;

    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

bool ComputeCov2DBounds(float3 cov2d, out float3 conic, out float radius)
{
    conic = float3(0, 0, 0);
    radius = 0;
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float invDet = 1.f / det;

    conic.x = cov2d.z * invDet;
    conic.y = -cov2d.y * invDet;
    conic.z = cov2d.x * invDet;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = sqrt(max(0.1f, b * b - det));
    radius = ceil(3.f * sqrt(b + disc));
    return true;
}

float2 ProjectPix(float4x4 projmat, float3 p, uint2 imgSize, float2 pp)
{
    float4 pH = Transform4x4(projmat, p);
    float rw = 1.f / (pH.w + 1e-6f);
    float3 pProj = float3(pH.x * rw, pH.y * rw, pH.z * rw);
    return float2(Ndc2Pix(pProj.x, (float)imgSize.x, pp.x),
                  Ndc2Pix(pProj.y, (float)imgSize.y, pp.y));
}

// Evaluate SH coefficients to color given view direction
void SHCoeffsToColor(uint degreesToUse, float3 viewdir,
                     StructuredBuffer<float> dcCoeffs, uint dcOffset,
                     StructuredBuffer<float> restCoeffs, uint restOffset,
                     out float3 color)
{
    color = float3(
        SH_C0 * dcCoeffs[dcOffset + 0],
        SH_C0 * dcCoeffs[dcOffset + 1],
        SH_C0 * dcCoeffs[dcOffset + 2]
    );
    if (degreesToUse < 1) return;

    float x = viewdir.x, y = viewdir.y, z = viewdir.z;
    float xx = x*x, xy = x*y, xz = x*z, yy = y*y, yz = y*z, zz = z*z;

    [unroll]
    for (int c = 0; c < 3; c++)
    {
        color[c] += SH_C1 * (-y * restCoeffs[restOffset + 0*3 + c] +
                               z * restCoeffs[restOffset + 1*3 + c] -
                               x * restCoeffs[restOffset + 2*3 + c]);
    }
    if (degreesToUse < 2) return;

    [unroll]
    for (int c2 = 0; c2 < 3; c2++)
    {
        color[c2] +=
            SH_C2[0] * xy          * restCoeffs[restOffset + 3*3 + c2] +
            SH_C2[1] * yz          * restCoeffs[restOffset + 4*3 + c2] +
            SH_C2[2] * (2.f*zz - xx - yy) * restCoeffs[restOffset + 5*3 + c2] +
            SH_C2[3] * xz          * restCoeffs[restOffset + 6*3 + c2] +
            SH_C2[4] * (xx - yy)   * restCoeffs[restOffset + 7*3 + c2];
    }
    if (degreesToUse < 3) return;

    [unroll]
    for (int c3 = 0; c3 < 3; c3++)
    {
        color[c3] +=
            SH_C3[0] * y*(3.f*xx - yy)          * restCoeffs[restOffset + 8*3  + c3] +
            SH_C3[1] * xy*z                      * restCoeffs[restOffset + 9*3  + c3] +
            SH_C3[2] * y*(4.f*zz - xx - yy)     * restCoeffs[restOffset + 10*3 + c3] +
            SH_C3[3] * z*(2.f*zz - 3.f*xx - 3.f*yy) * restCoeffs[restOffset + 11*3 + c3] +
            SH_C3[4] * x*(4.f*zz - xx - yy)     * restCoeffs[restOffset + 12*3 + c3] +
            SH_C3[5] * z*(xx - yy)               * restCoeffs[restOffset + 13*3 + c3] +
            SH_C3[6] * x*(xx - 3.f*yy)          * restCoeffs[restOffset + 14*3 + c3];
    }
}

// Overload that reads from RWStructuredBuffer (for backward pass / init)
void SHCoeffsToColorRW(uint degreesToUse, float3 viewdir,
                       RWStructuredBuffer<float> dcCoeffs, uint dcOffset,
                       RWStructuredBuffer<float> restCoeffs, uint restOffset,
                       out float3 color)
{
    color = float3(
        SH_C0 * dcCoeffs[dcOffset + 0],
        SH_C0 * dcCoeffs[dcOffset + 1],
        SH_C0 * dcCoeffs[dcOffset + 2]
    );
    if (degreesToUse < 1) return;

    float x = viewdir.x, y = viewdir.y, z = viewdir.z;
    float xx = x*x, xy = x*y, xz = x*z, yy = y*y, yz = y*z, zz = z*z;

    [unroll]
    for (int c = 0; c < 3; c++)
    {
        color[c] += SH_C1 * (-y * restCoeffs[restOffset + 0*3 + c] +
                               z * restCoeffs[restOffset + 1*3 + c] -
                               x * restCoeffs[restOffset + 2*3 + c]);
    }
    if (degreesToUse < 2) return;

    [unroll]
    for (int c2 = 0; c2 < 3; c2++)
    {
        color[c2] +=
            SH_C2[0] * xy          * restCoeffs[restOffset + 3*3 + c2] +
            SH_C2[1] * yz          * restCoeffs[restOffset + 4*3 + c2] +
            SH_C2[2] * (2.f*zz - xx - yy) * restCoeffs[restOffset + 5*3 + c2] +
            SH_C2[3] * xz          * restCoeffs[restOffset + 6*3 + c2] +
            SH_C2[4] * (xx - yy)   * restCoeffs[restOffset + 7*3 + c2];
    }
    if (degreesToUse < 3) return;

    [unroll]
    for (int c3 = 0; c3 < 3; c3++)
    {
        color[c3] +=
            SH_C3[0] * y*(3.f*xx - yy)          * restCoeffs[restOffset + 8*3  + c3] +
            SH_C3[1] * xy*z                      * restCoeffs[restOffset + 9*3  + c3] +
            SH_C3[2] * y*(4.f*zz - xx - yy)     * restCoeffs[restOffset + 10*3 + c3] +
            SH_C3[3] * z*(2.f*zz - 3.f*xx - 3.f*yy) * restCoeffs[restOffset + 11*3 + c3] +
            SH_C3[4] * x*(4.f*zz - xx - yy)     * restCoeffs[restOffset + 12*3 + c3] +
            SH_C3[5] * z*(xx - yy)               * restCoeffs[restOffset + 13*3 + c3] +
            SH_C3[6] * x*(xx - 3.f*yy)          * restCoeffs[restOffset + 14*3 + c3];
    }
}

// Backward VJPs used by Phase 2 (forward-declared here, defined in backward shaders)

// Inverse of cov2d → conic VJP
void Cov2DToConicVJP(float3 conic, float3 vConic, out float vCov2d[3])
{
    float det = conic.x * conic.z - conic.y * conic.y;
    float detSq = det * det;
    vCov2d[0] = (-conic.z * conic.z * vConic.x +
                  conic.y * conic.z * vConic.y +
                  (det - conic.x * conic.z) * vConic.z) / detSq;
    vCov2d[1] = ( 2.f * conic.y * conic.z * vConic.x -
                  (conic.x * conic.z + det) * vConic.y +
                  2.f * conic.x * conic.y * vConic.z) / detSq;
    vCov2d[2] = ((det - conic.x * conic.z) * vConic.x +
                  conic.x * conic.y * vConic.y -
                  conic.x * conic.x * vConic.z) / detSq;
}

// Project pixel VJP
float3 ProjectPixVJP(float4x4 projmat, float3 p, uint2 imgSize, float2 vXY)
{
    float4 pH = Transform4x4(projmat, p);
    float rw = 1.f / (pH.w + 1e-6f);
    float2 vNdc = float2(0.5f * (float)imgSize.x * vXY.x,
                         0.5f * (float)imgSize.y * vXY.y);
    float3 vP;
    vP.x = vNdc.x * rw * projmat[0][0] + vNdc.y * rw * projmat[1][0];
    vP.y = vNdc.x * rw * projmat[0][1] + vNdc.y * rw * projmat[1][1];
    vP.z = vNdc.x * rw * projmat[0][2] + vNdc.y * rw * projmat[1][2];
    float vW = -(vNdc.x * pH.x + vNdc.y * pH.y) * rw * rw;
    vP.x += vW * projmat[3][0];
    vP.y += vW * projmat[3][1];
    vP.z += vW * projmat[3][2];
    return vP;
}

#endif // GSPLAT_HELPERS_HLSL
