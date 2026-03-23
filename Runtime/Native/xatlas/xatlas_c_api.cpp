#include "xatlas.h"
#include "../meshoptimizer/meshoptimizer.h"
#include <cstring>

#if defined(_WIN32)
#define XATLAS_EXPORT __declspec(dllexport)
#else
#define XATLAS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

XATLAS_EXPORT xatlas::Atlas* xatlas_create() {
    return xatlas::Create();
}

XATLAS_EXPORT void xatlas_destroy(xatlas::Atlas* atlas) {
    if (atlas) xatlas::Destroy(atlas);
}

XATLAS_EXPORT int xatlas_add_mesh(
    xatlas::Atlas* atlas,
    const float* positions, int positionStride,
    const float* normals, int normalStride,
    int vertexCount,
    const int* indices, int indexCount)
{
    xatlas::MeshDecl decl;
    decl.vertexPositionData = positions;
    decl.vertexPositionStride = positionStride;
    decl.vertexNormalData = normals;
    decl.vertexNormalStride = normalStride;
    decl.vertexCount = vertexCount;
    decl.indexData = indices;
    decl.indexCount = indexCount;
    decl.indexFormat = xatlas::IndexFormat::UInt32;
    return (int)xatlas::AddMesh(atlas, decl);
}

XATLAS_EXPORT void xatlas_generate(xatlas::Atlas* atlas, int maxResolution) {
    xatlas::ChartOptions chartOpts;
    xatlas::PackOptions packOpts;
    packOpts.resolution = maxResolution > 0 ? maxResolution : 0;
    packOpts.padding = 2;
    packOpts.bilinear = true;
    packOpts.bruteForce = false;
    xatlas::Generate(atlas, chartOpts, packOpts);
}

XATLAS_EXPORT void xatlas_get_atlas_dims(const xatlas::Atlas* atlas, int* width, int* height) {
    if (!atlas) { *width = 0; *height = 0; return; }
    *width = (int)atlas->width;
    *height = (int)atlas->height;
}

XATLAS_EXPORT int xatlas_get_vertex_count(const xatlas::Atlas* atlas, int meshIndex) {
    if (!atlas || meshIndex < 0 || (uint32_t)meshIndex >= atlas->meshCount) return 0;
    return (int)atlas->meshes[meshIndex].vertexCount;
}

XATLAS_EXPORT int xatlas_get_index_count(const xatlas::Atlas* atlas, int meshIndex) {
    if (!atlas || meshIndex < 0 || (uint32_t)meshIndex >= atlas->meshCount) return 0;
    return (int)atlas->meshes[meshIndex].indexCount;
}

XATLAS_EXPORT void xatlas_get_vertices(
    const xatlas::Atlas* atlas, int meshIndex,
    float* uvs, int* xrefs, int maxVerts)
{
    if (!atlas || meshIndex < 0 || (uint32_t)meshIndex >= atlas->meshCount) return;
    const xatlas::Mesh& mesh = atlas->meshes[meshIndex];
    int count = (int)mesh.vertexCount < maxVerts ? (int)mesh.vertexCount : maxVerts;
    for (int i = 0; i < count; i++) {
        uvs[i * 2] = mesh.vertexArray[i].uv[0];
        uvs[i * 2 + 1] = mesh.vertexArray[i].uv[1];
        xrefs[i] = (int)mesh.vertexArray[i].xref;
    }
}

XATLAS_EXPORT void xatlas_get_indices(
    const xatlas::Atlas* atlas, int meshIndex,
    int* outIndices, int maxIndices)
{
    if (!atlas || meshIndex < 0 || (uint32_t)meshIndex >= atlas->meshCount) return;
    const xatlas::Mesh& mesh = atlas->meshes[meshIndex];
    int count = (int)mesh.indexCount < maxIndices ? (int)mesh.indexCount : maxIndices;
    for (int i = 0; i < count; i++)
        outIndices[i] = (int)mesh.indexArray[i];
}

XATLAS_EXPORT void xatlas_generate_opts(
    xatlas::Atlas* atlas,
    float maxChartArea, float maxBoundaryLength,
    float normalDeviationWeight, float roundnessWeight,
    float straightnessWeight, float normalSeamWeight,
    float textureSeamWeight, float maxCost,
    unsigned int maxIterations,
    unsigned int maxChartSize, unsigned int padding,
    float texelsPerUnit, unsigned int resolution,
    int bilinear, int blockAlign, int bruteForce,
    int rotateChartsToAxis, int rotateCharts)
{
    xatlas::ChartOptions co;
    co.maxChartArea          = maxChartArea;
    co.maxBoundaryLength     = maxBoundaryLength;
    co.normalDeviationWeight = normalDeviationWeight;
    co.roundnessWeight       = roundnessWeight;
    co.straightnessWeight    = straightnessWeight;
    co.normalSeamWeight      = normalSeamWeight;
    co.textureSeamWeight     = textureSeamWeight;
    co.maxCost               = maxCost;
    co.maxIterations         = maxIterations;

    xatlas::PackOptions po;
    po.maxChartSize        = maxChartSize;
    po.padding             = padding;
    po.texelsPerUnit       = texelsPerUnit;
    po.resolution          = resolution;
    po.bilinear            = bilinear != 0;
    po.blockAlign          = blockAlign != 0;
    po.bruteForce          = bruteForce != 0;
    po.rotateChartsToAxis  = rotateChartsToAxis != 0;
    po.rotateCharts        = rotateCharts != 0;

    xatlas::Generate(atlas, co, po);
}

XATLAS_EXPORT int meshopt_simplify_mesh(
    const float* positions, int vertexCount, int positionStride,
    const unsigned int* indices, int indexCount,
    int targetIndexCount, float targetError,
    unsigned int* outIndices, float* outError)
{
    return (int)meshopt_simplify(
        outIndices,
        indices, (size_t)indexCount,
        positions, (size_t)vertexCount, (size_t)positionStride,
        (size_t)targetIndexCount, targetError,
        0, outError);
}

} // extern "C"
