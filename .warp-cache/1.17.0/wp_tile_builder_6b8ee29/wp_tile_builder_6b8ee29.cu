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



extern "C" __global__ void unpack_group_keys_06cb105b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint64> var_packed_pairs,
    wp::array_t<wp::uint32> var_group_keys)
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
        wp::uint64* var_1;
        wp::uint64 var_2;
        wp::uint64 var_3;
        wp::uint64 var_4;
        wp::uint32 var_5;
        //---------
        // forward
        // def unpack_group_keys(                                                                 <L 110>
        // item = wp.tid()                                                                        <L 114>
        var_0 = builtin_tid1d();
        // group_keys[item] = wp.uint32(packed_pairs[item] >> wp.uint64(32))                      <L 115>
        var_1 = wp::address(var_packed_pairs, var_0);
        var_2 = 32ull;
        var_4 = wp::load(var_1);
        var_3 = wp::rshift(var_4, var_2);
        var_5 = wp::uint32(var_3);
        wp::array_store(var_group_keys, var_0, var_5);
    }
}



extern "C" __global__ void copy_pair_count_13cbd63d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_pair_prefix,
    wp::int32 var_item_count,
    wp::array_t<wp::int32> var_pair_count)
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
        const wp::int32 var_0 = 0;
        bool var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        const wp::int32 var_8 = 0;
        //---------
        // forward
        // def copy_pair_count(                                                                   <L 128>
        // if item_count > 0:                                                                     <L 133>
        var_1 = (var_item_count > var_0);
        if (var_1) {
            // pair_count[0] = pair_prefix[item_count - 1]                                        <L 134>
            var_3 = wp::sub(var_item_count, var_2);
            var_4 = wp::address(var_pair_prefix, var_3);
            var_6 = wp::load(var_4);
            wp::array_store(var_pair_count, var_5, var_6);
        }
        if (!var_1) {
            // pair_count[0] = 0                                                                  <L 136>
            wp::array_store(var_pair_count, var_8, var_7);
        }
    }
}



extern "C" __global__ void count_sorted_groups_764e2aa8_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::uint32> var_group_keys,
    wp::array_t<wp::int32> var_group_counts)
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
        wp::uint32* var_1;
        wp::int32 var_2;
        wp::uint32 var_3;
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        //---------
        // forward
        // def count_sorted_groups(                                                               <L 119>
        // item = wp.tid()                                                                        <L 123>
        var_0 = builtin_tid1d();
        // wp.atomic_add(group_counts, int(group_keys[item]), 1)                                  <L 124>
        var_1 = wp::address(var_group_keys, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::int(var_3);
        var_5 = wp::atomic_add(var_group_counts, var_2, var_4);
    }
}



extern "C" __global__ void emit_projected_tile_pairs_79bf94b3_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<4, wp::int32>> var_tile_bounds,
    wp::array_t<wp::float32> var_depths,
    wp::array_t<wp::int32> var_pair_counts,
    wp::array_t<wp::int32> var_pair_prefix,
    wp::array_t<wp::int32> var_group_ids,
    wp::array_t<wp::uint32> var_splat_ids,
    wp::int32 var_tiles_per_view,
    wp::int32 var_tiles_x,
    wp::array_t<wp::float32> var_depth_keys,
    wp::array_t<wp::uint64> var_packed_pairs)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<4, wp::int32>* var_6;
        wp::vec_t<4, wp::int32> var_7;
        wp::vec_t<4, wp::int32> var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        const wp::int32 var_17 = 3;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::range_t var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::uint64 var_36;
        wp::uint64 var_37;
        wp::uint64 var_38;
        wp::uint32* var_39;
        wp::uint64 var_40;
        wp::uint32 var_41;
        wp::uint64 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        //---------
        // forward
        // def emit_projected_tile_pairs(                                                         <L 80>
        // item = wp.tid()                                                                        <L 93>
        var_0 = builtin_tid1d();
        // count = pair_counts[item]                                                              <L 94>
        var_1 = wp::address(var_pair_counts, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if count > 0:                                                                          <L 95>
        var_5 = (var_2 > var_4);
        if (var_5) {
            // bounds = tile_bounds[item]                                                         <L 96>
            var_6 = wp::address(var_tile_bounds, var_0);
            var_8 = wp::load(var_6);
            var_7 = wp::copy(var_8);
            // output = pair_prefix[item] - count                                                 <L 97>
            var_9 = wp::address(var_pair_prefix, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::sub(var_11, var_2);
            // group_base = group_ids[item] * tiles_per_view                                      <L 98>
            var_12 = wp::address(var_group_ids, var_0);
            var_14 = wp::load(var_12);
            var_13 = wp::mul(var_14, var_tiles_per_view);
            // for tile_y in range(bounds[2], bounds[3] + 1):                                     <L 99>
            var_16 = wp::extract(var_7, var_15);
            var_18 = wp::extract(var_7, var_17);
            var_20 = wp::add(var_18, var_19);
            var_21 = wp::range(var_16, var_20);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // for tile_x in range(bounds[0], bounds[1] + 1):                                 <L 100>
                var_24 = wp::extract(var_7, var_23);
                var_26 = wp::extract(var_7, var_25);
                var_28 = wp::add(var_26, var_27);
                var_29 = wp::range(var_24, var_28);
                start_for_2:;
                    if (iter_cmp(var_29) == 0) goto end_for_2;
                    var_30 = wp::iter_next(var_29);
                    // group = group_base + tile_y * tiles_x + tile_x                             <L 101>
                    var_31 = wp::mul(var_22, var_tiles_x);
                    var_32 = wp::add(var_13, var_31);
                    var_33 = wp::add(var_32, var_30);
                    // depth_keys[output] = depths[item]                                          <L 102>
                    var_34 = wp::address(var_depths, var_0);
                    var_35 = wp::load(var_34);
                    wp::array_store(var_depth_keys, var_10, var_35);
                    // packed_pairs[output] = (                                                   <L 103>
                    // wp.uint64(group) << wp.uint64(32)                                          <L 104>
                    var_36 = wp::uint64(var_33);
                    var_37 = 32ull;
                    var_38 = wp::lshift(var_36, var_37);
                    // ) | wp.uint64(splat_ids[item])                                             <L 105>
                    var_39 = wp::address(var_splat_ids, var_0);
                    var_41 = wp::load(var_39);
                    var_40 = wp::uint64(var_41);
                    var_42 = wp::bit_or(var_38, var_40);
                    // packed_pairs[output] = (                                                   <L 103>
                    wp::array_store(var_packed_pairs, var_10, var_42);
                    // output += 1                                                                <L 106>
                    var_44 = wp::add(var_10, var_43);
                    wp::assign(var_10, var_44);
                    goto start_for_2;
                end_for_2:;
                goto start_for_0;
            end_for_0:;
        }
    }
}



extern "C" __global__ void count_projected_tile_pairs_8bdf587e_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_centres,
    wp::array_t<wp::vec_t<3, wp::float32>> var_conics,
    wp::array_t<wp::float32> var_supports,
    wp::int32 var_count,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_tile_size,
    wp::int32 var_tiles_x,
    wp::int32 var_tiles_y,
    wp::array_t<wp::vec_t<4, wp::int32>> var_tile_bounds,
    wp::array_t<wp::int32> var_pair_counts)
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
        const wp::int32 var_1 = 0;
        const wp::int32 var_2 = 0;
        const wp::int32 var_3 = -1;
        const wp::int32 var_4 = 0;
        const wp::int32 var_5 = -1;
        wp::vec_t<4, wp::int32> var_6;
        bool var_7;
        wp::vec_t<2, wp::float32>* var_8;
        wp::vec_t<2, wp::float32> var_9;
        wp::vec_t<2, wp::float32> var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::int32 var_19 = 2;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        bool var_28;
        const wp::float32 var_29 = 1e-12;
        bool var_30;
        const wp::float32 var_31 = 0.0;
        bool var_32;
        const wp::int32 var_33 = 2;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::float32 var_40 = 0.0;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.0;
        wp::float32 var_45;
        wp::float32 var_46;
        bool var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::float32 var_52;
        bool var_53;
        bool var_54;
        bool var_55;
        const wp::int32 var_56 = 0;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::int32 var_65;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::int32 var_72;
        const wp::int32 var_73 = 1;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        bool var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        const wp::int32 var_83 = 0;
        bool var_84;
        bool var_85;
        bool var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::int32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 0;
        wp::int32 var_93;
        const wp::int32 var_94 = 1;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        const wp::int32 var_98 = 0;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        const wp::int32 var_105 = 0;
        wp::int32 var_106;
        wp::int32 var_107;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        wp::int32 var_110;
        bool var_111;
        bool var_112;
        bool var_113;
        wp::vec_t<4, wp::int32> var_114;
        wp::int32 var_115;
        const wp::int32 var_116 = 1;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        //---------
        // forward
        // def count_projected_tile_pairs(                                                        <L 24>
        // splat = wp.tid()                                                                       <L 38>
        var_0 = builtin_tid1d();
        // pair_counts[splat] = 0                                                                 <L 39>
        wp::array_store(var_pair_counts, var_0, var_1);
        // tile_bounds[splat] = wp.vec4i(0, -1, 0, -1)                                            <L 40>
        var_6 = wp::vec_t<4, wp::int32>(var_2, var_3, var_4, var_5);
        wp::array_store(var_tile_bounds, var_0, var_6);
        // if splat < count:                                                                      <L 42>
        var_7 = (var_0 < var_count);
        if (var_7) {
            // centre = centres[splat]                                                            <L 43>
            var_8 = wp::address(var_centres, var_0);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // conic = conics[splat]                                                              <L 44>
            var_11 = wp::address(var_conics, var_0);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // support = supports[splat]                                                          <L 45>
            var_14 = wp::address(var_supports, var_0);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // determinant = conic[0] * conic[2] - conic[1] * conic[1]                            <L 46>
            var_18 = wp::extract(var_12, var_17);
            var_20 = wp::extract(var_12, var_19);
            var_21 = wp::mul(var_18, var_20);
            var_23 = wp::extract(var_12, var_22);
            var_25 = wp::extract(var_12, var_24);
            var_26 = wp::mul(var_23, var_25);
            var_27 = wp::sub(var_21, var_26);
            // if determinant > 1.0e-12 and support > 0.0:                                        <L 47>
            var_30 = (var_27 > var_29);
            var_28 = var_30;
            if (var_28) {
                var_32 = (var_15 > var_31);
                var_28 = var_28 && var_32;
            }
            if (var_28) {
                // cov_xx = conic[2] / determinant                                                <L 48>
                var_34 = wp::extract(var_12, var_33);
                var_35 = wp::div(var_34, var_27);
                // cov_yy = conic[0] / determinant                                                <L 49>
                var_37 = wp::extract(var_12, var_36);
                var_38 = wp::div(var_37, var_27);
                // radius_x = wp.sqrt(wp.max(support * cov_xx, 0.0))                              <L 50>
                var_39 = wp::mul(var_15, var_35);
                var_41 = wp::max(var_39, var_40);
                var_42 = wp::sqrt(var_41);
                // radius_y = wp.sqrt(wp.max(support * cov_yy, 0.0))                              <L 51>
                var_43 = wp::mul(var_15, var_38);
                var_45 = wp::max(var_43, var_44);
                var_46 = wp::sqrt(var_45);
                // if (                                                                           <L 53>
                // wp.isfinite(centre[0])                                                         <L 54>
                var_49 = wp::extract(var_9, var_48);
                var_50 = wp::isfinite(var_49);
                var_47 = var_50;
                if (var_47) {
                    // and wp.isfinite(centre[1])                                                 <L 55>
                    var_52 = wp::extract(var_9, var_51);
                    var_53 = wp::isfinite(var_52);
                    var_47 = var_47 && var_53;
                }
                if (var_47) {
                    // and wp.isfinite(radius_x)                                                  <L 56>
                    var_54 = wp::isfinite(var_42);
                    var_47 = var_47 && var_54;
                }
                if (var_47) {
                    // and wp.isfinite(radius_y)                                                  <L 57>
                    var_55 = wp::isfinite(var_46);
                    var_47 = var_47 && var_55;
                }
                if (var_47) {
                    // min_px = int(wp.floor(centre[0] - radius_x))                               <L 59>
                    var_57 = wp::extract(var_9, var_56);
                    var_58 = wp::sub(var_57, var_42);
                    var_59 = wp::floor(var_58);
                    var_60 = wp::int(var_59);
                    // max_px = int(wp.ceil(centre[0] + radius_x)) - 1                            <L 60>
                    var_62 = wp::extract(var_9, var_61);
                    var_63 = wp::add(var_62, var_42);
                    var_64 = wp::ceil(var_63);
                    var_65 = wp::int(var_64);
                    var_67 = wp::sub(var_65, var_66);
                    // min_py = int(wp.floor(centre[1] - radius_y))                               <L 61>
                    var_69 = wp::extract(var_9, var_68);
                    var_70 = wp::sub(var_69, var_46);
                    var_71 = wp::floor(var_70);
                    var_72 = wp::int(var_71);
                    // max_py = int(wp.ceil(centre[1] + radius_y)) - 1                            <L 62>
                    var_74 = wp::extract(var_9, var_73);
                    var_75 = wp::add(var_74, var_46);
                    var_76 = wp::ceil(var_75);
                    var_77 = wp::int(var_76);
                    var_79 = wp::sub(var_77, var_78);
                    // if max_px >= 0 and max_py >= 0 and min_px < width and min_py < height:       <L 64>
                    var_82 = (var_67 >= var_81);
                    var_80 = var_82;
                    if (var_80) {
                        var_84 = (var_79 >= var_83);
                        var_80 = var_80 && var_84;
                    }
                    if (var_80) {
                        var_85 = (var_60 < var_width);
                        var_80 = var_80 && var_85;
                    }
                    if (var_80) {
                        var_86 = (var_72 < var_height);
                        var_80 = var_80 && var_86;
                    }
                    if (var_80) {
                        // min_px = wp.max(min_px, 0)                                             <L 65>
                        var_88 = wp::max(var_60, var_87);
                        // max_px = wp.min(max_px, width - 1)                                     <L 66>
                        var_90 = wp::sub(var_width, var_89);
                        var_91 = wp::min(var_67, var_90);
                        // min_py = wp.max(min_py, 0)                                             <L 67>
                        var_93 = wp::max(var_72, var_92);
                        // max_py = wp.min(max_py, height - 1)                                    <L 68>
                        var_95 = wp::sub(var_height, var_94);
                        var_96 = wp::min(var_79, var_95);
                        // min_tx = wp.max(min_px // tile_size, 0)                                <L 69>
                        var_97 = wp::floordiv(var_88, var_tile_size);
                        var_99 = wp::max(var_97, var_98);
                        // max_tx = wp.min(max_px // tile_size, tiles_x - 1)                      <L 70>
                        var_100 = wp::floordiv(var_91, var_tile_size);
                        var_102 = wp::sub(var_tiles_x, var_101);
                        var_103 = wp::min(var_100, var_102);
                        // min_ty = wp.max(min_py // tile_size, 0)                                <L 71>
                        var_104 = wp::floordiv(var_93, var_tile_size);
                        var_106 = wp::max(var_104, var_105);
                        // max_ty = wp.min(max_py // tile_size, tiles_y - 1)                      <L 72>
                        var_107 = wp::floordiv(var_96, var_tile_size);
                        var_109 = wp::sub(var_tiles_y, var_108);
                        var_110 = wp::min(var_107, var_109);
                        // if min_tx <= max_tx and min_ty <= max_ty:                              <L 74>
                        var_112 = (var_99 <= var_103);
                        var_111 = var_112;
                        if (var_111) {
                            var_113 = (var_106 <= var_110);
                            var_111 = var_111 && var_113;
                        }
                        if (var_111) {
                            // tile_bounds[splat] = wp.vec4i(min_tx, max_tx, min_ty, max_ty)       <L 75>
                            var_114 = wp::vec_t<4, wp::int32>(var_99, var_103, var_106, var_110);
                            wp::array_store(var_tile_bounds, var_0, var_114);
                            // pair_counts[splat] = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)       <L 76>
                            var_115 = wp::sub(var_103, var_99);
                            var_117 = wp::add(var_115, var_116);
                            var_118 = wp::sub(var_110, var_106);
                            var_120 = wp::add(var_118, var_119);
                            var_121 = wp::mul(var_117, var_120);
                            wp::array_store(var_pair_counts, var_0, var_121);
                        }
                    }
                    var_122 = wp::where(var_80, var_88, var_60);
                    var_123 = wp::where(var_80, var_91, var_67);
                    var_124 = wp::where(var_80, var_93, var_72);
                    var_125 = wp::where(var_80, var_96, var_79);
                }
            }
        }
    }
}

