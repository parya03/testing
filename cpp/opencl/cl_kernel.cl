// __kernel void vadd(
//     __global const float *a,
//     __global const float *b,
//     __global float *c)
// {
//     int gid = get_global_id(0);
//     c[gid] = a[gid] + b[gid];
// }

__kernel void matrixVectorMul(__global int* matrixA,
    __global int* vectorB,
    __global int* resultVector,
    int width_A)
{
    int tx = get_global_id(0);

    int value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectorB[k];
    }

    resultVector[tx] = value;
}