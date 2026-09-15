#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void adam_step_kernel_half_9975a192_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float16> var_g,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::float32> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::float16> var_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::float16* var_6;
        wp::float32 var_7;
        wp::float16 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 1.0;
        wp::float32 var_15;
        wp::float16* var_16;
        wp::float32 var_17;
        wp::float16 var_18;
        wp::float32 var_19;
        wp::float16* var_20;
        wp::float32 var_21;
        wp::float16 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32* var_25;
        const wp::float32 var_26 = 1.0;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32* var_33;
        const wp::float32 var_34 = 1.0;
        const wp::float32 var_35 = 1.0;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float16* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float16 var_46;
        wp::float16 var_47;
        wp::float16 var_48;
        //---------
        // forward
        // def adam_step_kernel_half(                                                             <L 50>
        // i = wp.tid()                                                                           <L 61>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * float(g[i])                                      <L 62>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::float(var_8);
        var_9 = wp::mul(var_5, var_7);
        var_10 = wp::add(var_2, var_9);
        wp::array_store(var_m, var_0, var_10);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * float(g[i]) * float(g[i])                        <L 63>
        var_11 = wp::address(var_v, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_beta2, var_13);
        var_15 = wp::sub(var_14, var_beta2);
        var_16 = wp::address(var_g, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::float(var_18);
        var_19 = wp::mul(var_15, var_17);
        var_20 = wp::address(var_g, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::float(var_22);
        var_23 = wp::mul(var_19, var_21);
        var_24 = wp::add(var_12, var_23);
        wp::array_store(var_v, var_0, var_24);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 64>
        var_25 = wp::address(var_m, var_0);
        var_28 = wp::add(var_t, var_27);
        var_29 = wp::pow(var_beta1, var_28);
        var_30 = wp::sub(var_26, var_29);
        var_32 = wp::load(var_25);
        var_31 = wp::div(var_32, var_30);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 65>
        var_33 = wp::address(var_v, var_0);
        var_36 = wp::add(var_t, var_35);
        var_37 = wp::pow(var_beta2, var_36);
        var_38 = wp::sub(var_34, var_37);
        var_40 = wp::load(var_33);
        var_39 = wp::div(var_40, var_38);
        // params[i] = params[i] - wp.float16(lr * mhat / (wp.sqrt(vhat) + eps))                  <L 66>
        var_41 = wp::address(var_params, var_0);
        var_42 = wp::mul(var_lr, var_31);
        var_43 = wp::sqrt(var_39);
        var_44 = wp::add(var_43, var_eps);
        var_45 = wp::div(var_42, var_44);
        var_46 = wp::float16(var_45);
        var_48 = wp::load(var_41);
        var_47 = wp::sub(var_48, var_46);
        wp::array_store(var_params, var_0, var_47);
    }
}



extern "C" __global__ void adam_step_kernel_half_9975a192_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float16> var_g,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::float32> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::float16> var_params,
    wp::array_t<wp::float16> adj_g,
    wp::array_t<wp::float32> adj_m,
    wp::array_t<wp::float32> adj_v,
    wp::float32 adj_lr,
    wp::float32 adj_beta1,
    wp::float32 adj_beta2,
    wp::float32 adj_t,
    wp::float32 adj_eps,
    wp::array_t<wp::float16> adj_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::float16* var_6;
        wp::float32 var_7;
        wp::float16 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 1.0;
        wp::float32 var_15;
        wp::float16* var_16;
        wp::float32 var_17;
        wp::float16 var_18;
        wp::float32 var_19;
        wp::float16* var_20;
        wp::float32 var_21;
        wp::float16 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32* var_25;
        const wp::float32 var_26 = 1.0;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32* var_33;
        const wp::float32 var_34 = 1.0;
        const wp::float32 var_35 = 1.0;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float16* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float16 var_46;
        wp::float16 var_47;
        wp::float16 var_48;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float16 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float16 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float16 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float16 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float16 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float16 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float16 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float16 adj_46 = {};
        wp::float16 adj_47 = {};
        wp::float16 adj_48 = {};
        //---------
        // forward
        // def adam_step_kernel_half(                                                             <L 50>
        // i = wp.tid()                                                                           <L 61>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * float(g[i])                                      <L 62>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::float(var_8);
        var_9 = wp::mul(var_5, var_7);
        var_10 = wp::add(var_2, var_9);
        // wp::array_store(var_m, var_0, var_10);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * float(g[i]) * float(g[i])                        <L 63>
        var_11 = wp::address(var_v, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_beta2, var_13);
        var_15 = wp::sub(var_14, var_beta2);
        var_16 = wp::address(var_g, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::float(var_18);
        var_19 = wp::mul(var_15, var_17);
        var_20 = wp::address(var_g, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::float(var_22);
        var_23 = wp::mul(var_19, var_21);
        var_24 = wp::add(var_12, var_23);
        // wp::array_store(var_v, var_0, var_24);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 64>
        var_25 = wp::address(var_m, var_0);
        var_28 = wp::add(var_t, var_27);
        var_29 = wp::pow(var_beta1, var_28);
        var_30 = wp::sub(var_26, var_29);
        var_32 = wp::load(var_25);
        var_31 = wp::div(var_32, var_30);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 65>
        var_33 = wp::address(var_v, var_0);
        var_36 = wp::add(var_t, var_35);
        var_37 = wp::pow(var_beta2, var_36);
        var_38 = wp::sub(var_34, var_37);
        var_40 = wp::load(var_33);
        var_39 = wp::div(var_40, var_38);
        // params[i] = params[i] - wp.float16(lr * mhat / (wp.sqrt(vhat) + eps))                  <L 66>
        var_41 = wp::address(var_params, var_0);
        var_42 = wp::mul(var_lr, var_31);
        var_43 = wp::sqrt(var_39);
        var_44 = wp::add(var_43, var_eps);
        var_45 = wp::div(var_42, var_44);
        var_46 = wp::float16(var_45);
        var_48 = wp::load(var_41);
        var_47 = wp::sub(var_48, var_46);
        // wp::array_store(var_params, var_0, var_47);
        //---------
        // reverse
        wp::adj_array_store(var_params, var_0, var_47, adj_params, adj_0, adj_47);
        wp::adj_sub(var_48, var_46, adj_41, adj_46, adj_47);
        wp::adj_float16(var_45, adj_45, adj_46);
        wp::adj_div(var_42, var_44, var_45, adj_42, adj_44, adj_45);
        wp::adj_add(var_43, var_eps, adj_43, adj_eps, adj_44);
        wp::adj_sqrt(var_39, var_43, adj_39, adj_43);
        wp::adj_mul(var_lr, var_31, adj_lr, adj_31, adj_42);
        wp::adj_address(var_params, var_0, adj_params, adj_0, adj_41);
        // adj: params[i] = params[i] - wp.float16(lr * mhat / (wp.sqrt(vhat) + eps))             <L 66>
        wp::adj_div(var_40, var_38, var_39, adj_33, adj_38, adj_39);
        wp::adj_sub(var_34, var_37, adj_34, adj_37, adj_38);
        wp::adj_pow(var_beta2, var_36, var_37, adj_beta2, adj_36, adj_37);
        wp::adj_add(var_t, var_35, adj_t, adj_35, adj_36);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_33);
        // adj: vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                    <L 65>
        wp::adj_div(var_32, var_30, var_31, adj_25, adj_30, adj_31);
        wp::adj_sub(var_26, var_29, adj_26, adj_29, adj_30);
        wp::adj_pow(var_beta1, var_28, var_29, adj_beta1, adj_28, adj_29);
        wp::adj_add(var_t, var_27, adj_t, adj_27, adj_28);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_25);
        // adj: mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                    <L 64>
        wp::adj_array_store(var_v, var_0, var_24, adj_v, adj_0, adj_24);
        wp::adj_add(var_12, var_23, adj_12, adj_23, adj_24);
        wp::adj_mul(var_19, var_21, adj_19, adj_21, adj_23);
        wp::adj_float(var_22, adj_20, adj_21);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_20);
        wp::adj_mul(var_15, var_17, adj_15, adj_17, adj_19);
        wp::adj_float(var_18, adj_16, adj_17);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_16);
        wp::adj_sub(var_14, var_beta2, adj_14, adj_beta2, adj_15);
        wp::adj_mul(var_beta2, var_13, adj_beta2, adj_11, adj_12);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_11);
        // adj: v[i] = beta2 * v[i] + (1.0 - beta2) * float(g[i]) * float(g[i])                   <L 63>
        wp::adj_array_store(var_m, var_0, var_10, adj_m, adj_0, adj_10);
        wp::adj_add(var_2, var_9, adj_2, adj_9, adj_10);
        wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_9);
        wp::adj_float(var_8, adj_6, adj_7);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_6);
        wp::adj_sub(var_4, var_beta1, adj_4, adj_beta1, adj_5);
        wp::adj_mul(var_beta1, var_3, adj_beta1, adj_1, adj_2);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_1);
        // adj: m[i] = beta1 * m[i] + (1.0 - beta1) * float(g[i])                                 <L 62>
        // adj: i = wp.tid()                                                                      <L 61>
        // adj: def adam_step_kernel_half(                                                        <L 50>
        continue;
    }
}



extern "C" __global__ void adam_step_kernel_vec3_ccff9369_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_g,
    wp::array_t<wp::vec_t<3, wp::float32>> var_m,
    wp::array_t<wp::vec_t<3, wp::float32>> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::vec_t<3, wp::float32>> var_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::float32 var_13 = 1.0;
        wp::float32 var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        const wp::float32 var_23 = 1.0;
        const wp::float32 var_24 = 1.0;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        const wp::float32 var_31 = 1.0;
        const wp::float32 var_32 = 1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        //---------
        // forward
        // def adam_step_kernel_vec3(                                                             <L 8>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                             <L 20>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_5, var_8);
        var_9 = wp::add(var_2, var_7);
        wp::array_store(var_m, var_0, var_9);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g[i], g[i])                            <L 21>
        var_10 = wp::address(var_v, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_beta2, var_12);
        var_14 = wp::sub(var_13, var_beta2);
        var_15 = wp::address(var_g, var_0);
        var_16 = wp::address(var_g, var_0);
        var_18 = wp::load(var_15);
        var_19 = wp::load(var_16);
        var_17 = wp::cw_mul(var_18, var_19);
        var_20 = wp::mul(var_14, var_17);
        var_21 = wp::add(var_11, var_20);
        wp::array_store(var_v, var_0, var_21);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 22>
        var_22 = wp::address(var_m, var_0);
        var_25 = wp::add(var_t, var_24);
        var_26 = wp::pow(var_beta1, var_25);
        var_27 = wp::sub(var_23, var_26);
        var_29 = wp::load(var_22);
        var_28 = wp::div(var_29, var_27);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 23>
        var_30 = wp::address(var_v, var_0);
        var_33 = wp::add(var_t, var_32);
        var_34 = wp::pow(var_beta2, var_33);
        var_35 = wp::sub(var_31, var_34);
        var_37 = wp::load(var_30);
        var_36 = wp::div(var_37, var_35);
        // sqrt_vhat = wp.vec3(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]))              <L 24>
        var_39 = wp::extract(var_36, var_38);
        var_40 = wp::sqrt(var_39);
        var_42 = wp::extract(var_36, var_41);
        var_43 = wp::sqrt(var_42);
        var_45 = wp::extract(var_36, var_44);
        var_46 = wp::sqrt(var_45);
        var_47 = wp::vec_t<3, wp::float32>(var_40, var_43, var_46);
        // eps_vec3 = wp.vec3(eps, eps, eps)                                                      <L 25>
        var_48 = wp::vec_t<3, wp::float32>(var_eps, var_eps, var_eps);
        // params[i] = params[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec3))                   <L 26>
        var_49 = wp::address(var_params, var_0);
        var_50 = wp::add(var_47, var_48);
        var_51 = wp::cw_div(var_28, var_50);
        var_52 = wp::mul(var_lr, var_51);
        var_54 = wp::load(var_49);
        var_53 = wp::sub(var_54, var_52);
        wp::array_store(var_params, var_0, var_53);
    }
}



extern "C" __global__ void adam_step_kernel_vec3_ccff9369_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_g,
    wp::array_t<wp::vec_t<3, wp::float32>> var_m,
    wp::array_t<wp::vec_t<3, wp::float32>> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::vec_t<3, wp::float32>> var_params,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_g,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_m,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_v,
    wp::float32 adj_lr,
    wp::float32 adj_beta1,
    wp::float32 adj_beta2,
    wp::float32 adj_t,
    wp::float32 adj_eps,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::float32 var_13 = 1.0;
        wp::float32 var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        const wp::float32 var_23 = 1.0;
        const wp::float32 var_24 = 1.0;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        const wp::float32 var_31 = 1.0;
        const wp::float32 var_32 = 1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        //---------
        // forward
        // def adam_step_kernel_vec3(                                                             <L 8>
        // i = wp.tid()                                                                           <L 19>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                             <L 20>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_5, var_8);
        var_9 = wp::add(var_2, var_7);
        // wp::array_store(var_m, var_0, var_9);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g[i], g[i])                            <L 21>
        var_10 = wp::address(var_v, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_beta2, var_12);
        var_14 = wp::sub(var_13, var_beta2);
        var_15 = wp::address(var_g, var_0);
        var_16 = wp::address(var_g, var_0);
        var_18 = wp::load(var_15);
        var_19 = wp::load(var_16);
        var_17 = wp::cw_mul(var_18, var_19);
        var_20 = wp::mul(var_14, var_17);
        var_21 = wp::add(var_11, var_20);
        // wp::array_store(var_v, var_0, var_21);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 22>
        var_22 = wp::address(var_m, var_0);
        var_25 = wp::add(var_t, var_24);
        var_26 = wp::pow(var_beta1, var_25);
        var_27 = wp::sub(var_23, var_26);
        var_29 = wp::load(var_22);
        var_28 = wp::div(var_29, var_27);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 23>
        var_30 = wp::address(var_v, var_0);
        var_33 = wp::add(var_t, var_32);
        var_34 = wp::pow(var_beta2, var_33);
        var_35 = wp::sub(var_31, var_34);
        var_37 = wp::load(var_30);
        var_36 = wp::div(var_37, var_35);
        // sqrt_vhat = wp.vec3(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]))              <L 24>
        var_39 = wp::extract(var_36, var_38);
        var_40 = wp::sqrt(var_39);
        var_42 = wp::extract(var_36, var_41);
        var_43 = wp::sqrt(var_42);
        var_45 = wp::extract(var_36, var_44);
        var_46 = wp::sqrt(var_45);
        var_47 = wp::vec_t<3, wp::float32>(var_40, var_43, var_46);
        // eps_vec3 = wp.vec3(eps, eps, eps)                                                      <L 25>
        var_48 = wp::vec_t<3, wp::float32>(var_eps, var_eps, var_eps);
        // params[i] = params[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec3))                   <L 26>
        var_49 = wp::address(var_params, var_0);
        var_50 = wp::add(var_47, var_48);
        var_51 = wp::cw_div(var_28, var_50);
        var_52 = wp::mul(var_lr, var_51);
        var_54 = wp::load(var_49);
        var_53 = wp::sub(var_54, var_52);
        // wp::array_store(var_params, var_0, var_53);
        //---------
        // reverse
        wp::adj_array_store(var_params, var_0, var_53, adj_params, adj_0, adj_53);
        wp::adj_sub(var_54, var_52, adj_49, adj_52, adj_53);
        wp::adj_mul(var_lr, var_51, adj_lr, adj_51, adj_52);
        wp::adj_cw_div(var_28, var_50, var_51, adj_28, adj_50, adj_51);
        wp::adj_add(var_47, var_48, adj_47, adj_48, adj_50);
        wp::adj_address(var_params, var_0, adj_params, adj_0, adj_49);
        // adj: params[i] = params[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec3))              <L 26>
        wp::adj_vec_t(var_eps, var_eps, var_eps, adj_eps, adj_eps, adj_eps, adj_48);
        // adj: eps_vec3 = wp.vec3(eps, eps, eps)                                                 <L 25>
        wp::adj_vec_t(var_40, var_43, var_46, adj_40, adj_43, adj_46, adj_47);
        wp::adj_sqrt(var_45, var_46, adj_45, adj_46);
        wp::adj_extract(var_36, var_44, adj_36, adj_44, adj_45);
        wp::adj_sqrt(var_42, var_43, adj_42, adj_43);
        wp::adj_extract(var_36, var_41, adj_36, adj_41, adj_42);
        wp::adj_sqrt(var_39, var_40, adj_39, adj_40);
        wp::adj_extract(var_36, var_38, adj_36, adj_38, adj_39);
        // adj: sqrt_vhat = wp.vec3(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]))         <L 24>
        wp::adj_div(var_37, var_35, adj_30, adj_35, adj_36);
        wp::adj_sub(var_31, var_34, adj_31, adj_34, adj_35);
        wp::adj_pow(var_beta2, var_33, var_34, adj_beta2, adj_33, adj_34);
        wp::adj_add(var_t, var_32, adj_t, adj_32, adj_33);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_30);
        // adj: vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                    <L 23>
        wp::adj_div(var_29, var_27, adj_22, adj_27, adj_28);
        wp::adj_sub(var_23, var_26, adj_23, adj_26, adj_27);
        wp::adj_pow(var_beta1, var_25, var_26, adj_beta1, adj_25, adj_26);
        wp::adj_add(var_t, var_24, adj_t, adj_24, adj_25);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_22);
        // adj: mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                    <L 22>
        wp::adj_array_store(var_v, var_0, var_21, adj_v, adj_0, adj_21);
        wp::adj_add(var_11, var_20, adj_11, adj_20, adj_21);
        wp::adj_mul(var_14, var_17, adj_14, adj_17, adj_20);
        wp::adj_cw_mul(var_18, var_19, adj_15, adj_16, adj_17);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_16);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_15);
        wp::adj_sub(var_13, var_beta2, adj_13, adj_beta2, adj_14);
        wp::adj_mul(var_beta2, var_12, adj_beta2, adj_10, adj_11);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_10);
        // adj: v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g[i], g[i])                       <L 21>
        wp::adj_array_store(var_m, var_0, var_9, adj_m, adj_0, adj_9);
        wp::adj_add(var_2, var_7, adj_2, adj_7, adj_9);
        wp::adj_mul(var_5, var_8, adj_5, adj_6, adj_7);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_6);
        wp::adj_sub(var_4, var_beta1, adj_4, adj_beta1, adj_5);
        wp::adj_mul(var_beta1, var_3, adj_beta1, adj_1, adj_2);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_1);
        // adj: m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                        <L 20>
        // adj: i = wp.tid()                                                                      <L 19>
        // adj: def adam_step_kernel_vec3(                                                        <L 8>
        continue;
    }
}



extern "C" __global__ void adam_step_kernel_float_f0550852_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_g,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::float32> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::float32> var_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 1.0;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        const wp::float32 var_23 = 1.0;
        const wp::float32 var_24 = 1.0;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32* var_30;
        const wp::float32 var_31 = 1.0;
        const wp::float32 var_32 = 1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        //---------
        // forward
        // def adam_step_kernel_float(                                                            <L 30>
        // i = wp.tid()                                                                           <L 41>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                             <L 42>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_5, var_8);
        var_9 = wp::add(var_2, var_7);
        wp::array_store(var_m, var_0, var_9);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]                                      <L 43>
        var_10 = wp::address(var_v, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_beta2, var_12);
        var_14 = wp::sub(var_13, var_beta2);
        var_15 = wp::address(var_g, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::mul(var_14, var_17);
        var_18 = wp::address(var_g, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::mul(var_16, var_20);
        var_21 = wp::add(var_11, var_19);
        wp::array_store(var_v, var_0, var_21);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 44>
        var_22 = wp::address(var_m, var_0);
        var_25 = wp::add(var_t, var_24);
        var_26 = wp::pow(var_beta1, var_25);
        var_27 = wp::sub(var_23, var_26);
        var_29 = wp::load(var_22);
        var_28 = wp::div(var_29, var_27);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 45>
        var_30 = wp::address(var_v, var_0);
        var_33 = wp::add(var_t, var_32);
        var_34 = wp::pow(var_beta2, var_33);
        var_35 = wp::sub(var_31, var_34);
        var_37 = wp::load(var_30);
        var_36 = wp::div(var_37, var_35);
        // params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)                              <L 46>
        var_38 = wp::address(var_params, var_0);
        var_39 = wp::mul(var_lr, var_28);
        var_40 = wp::sqrt(var_36);
        var_41 = wp::add(var_40, var_eps);
        var_42 = wp::div(var_39, var_41);
        var_44 = wp::load(var_38);
        var_43 = wp::sub(var_44, var_42);
        wp::array_store(var_params, var_0, var_43);
    }
}



extern "C" __global__ void adam_step_kernel_float_f0550852_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_g,
    wp::array_t<wp::float32> var_m,
    wp::array_t<wp::float32> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::float32> var_params,
    wp::array_t<wp::float32> adj_g,
    wp::array_t<wp::float32> adj_m,
    wp::array_t<wp::float32> adj_v,
    wp::float32 adj_lr,
    wp::float32 adj_beta1,
    wp::float32 adj_beta2,
    wp::float32 adj_t,
    wp::float32 adj_eps,
    wp::array_t<wp::float32> adj_params)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        const wp::float32 var_4 = 1.0;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 1.0;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        const wp::float32 var_23 = 1.0;
        const wp::float32 var_24 = 1.0;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32* var_30;
        const wp::float32 var_31 = 1.0;
        const wp::float32 var_32 = 1.0;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        //---------
        // forward
        // def adam_step_kernel_float(                                                            <L 30>
        // i = wp.tid()                                                                           <L 41>
        var_0 = builtin_tid1d();
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                             <L 42>
        var_1 = wp::address(var_m, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::mul(var_beta1, var_3);
        var_5 = wp::sub(var_4, var_beta1);
        var_6 = wp::address(var_g, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_5, var_8);
        var_9 = wp::add(var_2, var_7);
        // wp::array_store(var_m, var_0, var_9);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]                                      <L 43>
        var_10 = wp::address(var_v, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_beta2, var_12);
        var_14 = wp::sub(var_13, var_beta2);
        var_15 = wp::address(var_g, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::mul(var_14, var_17);
        var_18 = wp::address(var_g, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::mul(var_16, var_20);
        var_21 = wp::add(var_11, var_19);
        // wp::array_store(var_v, var_0, var_21);
        // mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                         <L 44>
        var_22 = wp::address(var_m, var_0);
        var_25 = wp::add(var_t, var_24);
        var_26 = wp::pow(var_beta1, var_25);
        var_27 = wp::sub(var_23, var_26);
        var_29 = wp::load(var_22);
        var_28 = wp::div(var_29, var_27);
        // vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                         <L 45>
        var_30 = wp::address(var_v, var_0);
        var_33 = wp::add(var_t, var_32);
        var_34 = wp::pow(var_beta2, var_33);
        var_35 = wp::sub(var_31, var_34);
        var_37 = wp::load(var_30);
        var_36 = wp::div(var_37, var_35);
        // params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)                              <L 46>
        var_38 = wp::address(var_params, var_0);
        var_39 = wp::mul(var_lr, var_28);
        var_40 = wp::sqrt(var_36);
        var_41 = wp::add(var_40, var_eps);
        var_42 = wp::div(var_39, var_41);
        var_44 = wp::load(var_38);
        var_43 = wp::sub(var_44, var_42);
        // wp::array_store(var_params, var_0, var_43);
        //---------
        // reverse
        wp::adj_array_store(var_params, var_0, var_43, adj_params, adj_0, adj_43);
        wp::adj_sub(var_44, var_42, adj_38, adj_42, adj_43);
        wp::adj_div(var_39, var_41, var_42, adj_39, adj_41, adj_42);
        wp::adj_add(var_40, var_eps, adj_40, adj_eps, adj_41);
        wp::adj_sqrt(var_36, var_40, adj_36, adj_40);
        wp::adj_mul(var_lr, var_28, adj_lr, adj_28, adj_39);
        wp::adj_address(var_params, var_0, adj_params, adj_0, adj_38);
        // adj: params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)                         <L 46>
        wp::adj_div(var_37, var_35, var_36, adj_30, adj_35, adj_36);
        wp::adj_sub(var_31, var_34, adj_31, adj_34, adj_35);
        wp::adj_pow(var_beta2, var_33, var_34, adj_beta2, adj_33, adj_34);
        wp::adj_add(var_t, var_32, adj_t, adj_32, adj_33);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_30);
        // adj: vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))                                    <L 45>
        wp::adj_div(var_29, var_27, var_28, adj_22, adj_27, adj_28);
        wp::adj_sub(var_23, var_26, adj_23, adj_26, adj_27);
        wp::adj_pow(var_beta1, var_25, var_26, adj_beta1, adj_25, adj_26);
        wp::adj_add(var_t, var_24, adj_t, adj_24, adj_25);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_22);
        // adj: mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))                                    <L 44>
        wp::adj_array_store(var_v, var_0, var_21, adj_v, adj_0, adj_21);
        wp::adj_add(var_11, var_19, adj_11, adj_19, adj_21);
        wp::adj_mul(var_16, var_20, adj_16, adj_18, adj_19);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_18);
        wp::adj_mul(var_14, var_17, adj_14, adj_15, adj_16);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_15);
        wp::adj_sub(var_13, var_beta2, adj_13, adj_beta2, adj_14);
        wp::adj_mul(var_beta2, var_12, adj_beta2, adj_10, adj_11);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_10);
        // adj: v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]                                 <L 43>
        wp::adj_array_store(var_m, var_0, var_9, adj_m, adj_0, adj_9);
        wp::adj_add(var_2, var_7, adj_2, adj_7, adj_9);
        wp::adj_mul(var_5, var_8, adj_5, adj_6, adj_7);
        wp::adj_address(var_g, var_0, adj_g, adj_0, adj_6);
        wp::adj_sub(var_4, var_beta1, adj_4, adj_beta1, adj_5);
        wp::adj_mul(var_beta1, var_3, adj_beta1, adj_1, adj_2);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_1);
        // adj: m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]                                        <L 42>
        // adj: i = wp.tid()                                                                      <L 41>
        // adj: def adam_step_kernel_float(                                                       <L 30>
        continue;
    }
}

