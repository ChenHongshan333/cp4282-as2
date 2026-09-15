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


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:346
static CUDA_CALLABLE wp::vec_t<3, wp::float32> projected_covariance_0(
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_z,
    wp::float32 var_focal)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-08;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::vec_t<4, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::int32 var_11 = 3;
    wp::float32 var_12;
    const wp::float32 var_13 = 1.0;
    const wp::float32 var_14 = 2.0;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 2.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 2.0;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 1.0;
    const wp::float32 var_37 = 2.0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 2.0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::float32 var_49 = 2.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 2.0;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 1.0;
    const wp::float32 var_60 = 2.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 0;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 1;
    const wp::int32 var_71 = 0;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    const wp::int32 var_74 = 0;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    const wp::int32 var_77 = 0;
    const wp::int32 var_78 = 1;
    wp::float32 var_79;
    const wp::int32 var_80 = 1;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    const wp::int32 var_83 = 2;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 0;
    const wp::int32 var_88 = 2;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.0;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    const wp::float32 var_104 = 0.0;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::vec_t<3, wp::float32> var_126;
    const wp::float32 var_127 = 2.0;
    const wp::int32 var_128 = 0;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::float32 var_132 = 2.0;
    const wp::int32 var_133 = 1;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::float32 var_137 = 2.0;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::vec_t<3, wp::float32> var_142;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    const wp::int32 var_145 = 0;
    wp::float32 var_146;
    wp::float32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::int32 var_153 = 2;
    wp::float32 var_154;
    const wp::int32 var_155 = 2;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::vec_t<3, wp::float32> var_158;
    const wp::int32 var_159 = 0;
    wp::float32 var_160;
    const wp::int32 var_161 = 0;
    wp::float32 var_162;
    wp::float32 var_163;
    const wp::int32 var_164 = 1;
    wp::float32 var_165;
    const wp::int32 var_166 = 1;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 2;
    wp::float32 var_170;
    const wp::int32 var_171 = 2;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 0;
    wp::float32 var_176;
    const wp::int32 var_177 = 0;
    wp::float32 var_178;
    wp::float32 var_179;
    const wp::int32 var_180 = 1;
    wp::float32 var_181;
    const wp::int32 var_182 = 1;
    wp::float32 var_183;
    wp::float32 var_184;
    const wp::int32 var_185 = 2;
    wp::float32 var_186;
    const wp::int32 var_187 = 2;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::vec_t<3, wp::float32> var_190;
    wp::float32 var_191;
    const wp::float32 var_192 = 0.3;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    wp::vec_t<3, wp::float32> var_197;
    //---------
    // forward
    // def projected_covariance(                                                              <L 347>
    // quaternion = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)             <L 356>
    var_0 = wp::dot(var_quaternion, var_quaternion);
    var_2 = wp::add(var_0, var_1);
    var_3 = wp::sqrt(var_2);
    var_4 = wp::div(var_quaternion, var_3);
    // qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]            <L 357>
    var_6 = wp::extract(var_4, var_5);
    var_8 = wp::extract(var_4, var_7);
    var_10 = wp::extract(var_4, var_9);
    var_12 = wp::extract(var_4, var_11);
    // r0 = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy))       <L 358>
    var_15 = wp::mul(var_10, var_10);
    var_16 = wp::mul(var_12, var_12);
    var_17 = wp::add(var_15, var_16);
    var_18 = wp::mul(var_14, var_17);
    var_19 = wp::sub(var_13, var_18);
    var_21 = wp::mul(var_8, var_10);
    var_22 = wp::mul(var_6, var_12);
    var_23 = wp::add(var_21, var_22);
    var_24 = wp::mul(var_20, var_23);
    var_26 = wp::mul(var_8, var_12);
    var_27 = wp::mul(var_6, var_10);
    var_28 = wp::sub(var_26, var_27);
    var_29 = wp::mul(var_25, var_28);
    var_30 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
    // r1 = wp.vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx))       <L 359>
    var_32 = wp::mul(var_8, var_10);
    var_33 = wp::mul(var_6, var_12);
    var_34 = wp::sub(var_32, var_33);
    var_35 = wp::mul(var_31, var_34);
    var_38 = wp::mul(var_8, var_8);
    var_39 = wp::mul(var_12, var_12);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_37, var_40);
    var_42 = wp::sub(var_36, var_41);
    var_44 = wp::mul(var_10, var_12);
    var_45 = wp::mul(var_6, var_8);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_43, var_46);
    var_48 = wp::vec_t<3, wp::float32>(var_35, var_42, var_47);
    // r2 = wp.vec3(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))       <L 360>
    var_50 = wp::mul(var_8, var_12);
    var_51 = wp::mul(var_6, var_10);
    var_52 = wp::add(var_50, var_51);
    var_53 = wp::mul(var_49, var_52);
    var_55 = wp::mul(var_10, var_12);
    var_56 = wp::mul(var_6, var_8);
    var_57 = wp::sub(var_55, var_56);
    var_58 = wp::mul(var_54, var_57);
    var_61 = wp::mul(var_8, var_8);
    var_62 = wp::mul(var_10, var_10);
    var_63 = wp::add(var_61, var_62);
    var_64 = wp::mul(var_60, var_63);
    var_65 = wp::sub(var_59, var_64);
    var_66 = wp::vec_t<3, wp::float32>(var_53, var_58, var_65);
    // c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])                                 <L 361>
    var_69 = wp::extract(var_camera, var_67, var_68);
    var_72 = wp::extract(var_camera, var_70, var_71);
    var_75 = wp::extract(var_camera, var_73, var_74);
    var_76 = wp::vec_t<3, wp::float32>(var_69, var_72, var_75);
    // c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])                                 <L 362>
    var_79 = wp::extract(var_camera, var_77, var_78);
    var_82 = wp::extract(var_camera, var_80, var_81);
    var_85 = wp::extract(var_camera, var_83, var_84);
    var_86 = wp::vec_t<3, wp::float32>(var_79, var_82, var_85);
    // c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])                                 <L 363>
    var_89 = wp::extract(var_camera, var_87, var_88);
    var_92 = wp::extract(var_camera, var_90, var_91);
    var_95 = wp::extract(var_camera, var_93, var_94);
    var_96 = wp::vec_t<3, wp::float32>(var_89, var_92, var_95);
    // j0 = wp.vec3(focal / z, 0.0, -focal * x / (z * z))                                     <L 364>
    var_97 = wp::div(var_focal, var_z);
    var_99 = wp::neg(var_focal);
    var_100 = wp::mul(var_99, var_x);
    var_101 = wp::mul(var_z, var_z);
    var_102 = wp::div(var_100, var_101);
    var_103 = wp::vec_t<3, wp::float32>(var_97, var_98, var_102);
    // j1 = wp.vec3(0.0, focal / z, -focal * y / (z * z))                                     <L 365>
    var_105 = wp::div(var_focal, var_z);
    var_106 = wp::neg(var_focal);
    var_107 = wp::mul(var_106, var_y);
    var_108 = wp::mul(var_z, var_z);
    var_109 = wp::div(var_107, var_108);
    var_110 = wp::vec_t<3, wp::float32>(var_104, var_105, var_109);
    // p0 = wp.vec3(wp.dot(j0, c0), wp.dot(j0, c1), wp.dot(j0, c2))                           <L 366>
    var_111 = wp::dot(var_103, var_76);
    var_112 = wp::dot(var_103, var_86);
    var_113 = wp::dot(var_103, var_96);
    var_114 = wp::vec_t<3, wp::float32>(var_111, var_112, var_113);
    // p1 = wp.vec3(wp.dot(j1, c0), wp.dot(j1, c1), wp.dot(j1, c2))                           <L 367>
    var_115 = wp::dot(var_110, var_76);
    var_116 = wp::dot(var_110, var_86);
    var_117 = wp::dot(var_110, var_96);
    var_118 = wp::vec_t<3, wp::float32>(var_115, var_116, var_117);
    // b0 = wp.vec3(wp.dot(p0, r0), wp.dot(p0, r1), wp.dot(p0, r2))                           <L 368>
    var_119 = wp::dot(var_114, var_30);
    var_120 = wp::dot(var_114, var_48);
    var_121 = wp::dot(var_114, var_66);
    var_122 = wp::vec_t<3, wp::float32>(var_119, var_120, var_121);
    // b1 = wp.vec3(wp.dot(p1, r0), wp.dot(p1, r1), wp.dot(p1, r2))                           <L 369>
    var_123 = wp::dot(var_118, var_30);
    var_124 = wp::dot(var_118, var_48);
    var_125 = wp::dot(var_118, var_66);
    var_126 = wp::vec_t<3, wp::float32>(var_123, var_124, var_125);
    // scales_squared = wp.vec3(wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1]), wp.exp(2.0 * log_scale[2]))       <L 370>
    var_129 = wp::extract(var_log_scale, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::exp(var_130);
    var_134 = wp::extract(var_log_scale, var_133);
    var_135 = wp::mul(var_132, var_134);
    var_136 = wp::exp(var_135);
    var_139 = wp::extract(var_log_scale, var_138);
    var_140 = wp::mul(var_137, var_139);
    var_141 = wp::exp(var_140);
    var_142 = wp::vec_t<3, wp::float32>(var_131, var_136, var_141);
    // b0_squared = wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])                      <L 371>
    var_144 = wp::extract(var_122, var_143);
    var_146 = wp::extract(var_122, var_145);
    var_147 = wp::mul(var_144, var_146);
    var_149 = wp::extract(var_122, var_148);
    var_151 = wp::extract(var_122, var_150);
    var_152 = wp::mul(var_149, var_151);
    var_154 = wp::extract(var_122, var_153);
    var_156 = wp::extract(var_122, var_155);
    var_157 = wp::mul(var_154, var_156);
    var_158 = wp::vec_t<3, wp::float32>(var_147, var_152, var_157);
    // b0_b1 = wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2])                           <L 372>
    var_160 = wp::extract(var_122, var_159);
    var_162 = wp::extract(var_126, var_161);
    var_163 = wp::mul(var_160, var_162);
    var_165 = wp::extract(var_122, var_164);
    var_167 = wp::extract(var_126, var_166);
    var_168 = wp::mul(var_165, var_167);
    var_170 = wp::extract(var_122, var_169);
    var_172 = wp::extract(var_126, var_171);
    var_173 = wp::mul(var_170, var_172);
    var_174 = wp::vec_t<3, wp::float32>(var_163, var_168, var_173);
    // b1_squared = wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])                      <L 373>
    var_176 = wp::extract(var_126, var_175);
    var_178 = wp::extract(var_126, var_177);
    var_179 = wp::mul(var_176, var_178);
    var_181 = wp::extract(var_126, var_180);
    var_183 = wp::extract(var_126, var_182);
    var_184 = wp::mul(var_181, var_183);
    var_186 = wp::extract(var_126, var_185);
    var_188 = wp::extract(var_126, var_187);
    var_189 = wp::mul(var_186, var_188);
    var_190 = wp::vec_t<3, wp::float32>(var_179, var_184, var_189);
    // a = wp.dot(scales_squared, b0_squared) + FILTER_VARIANCE                               <L 374>
    var_191 = wp::dot(var_142, var_158);
    var_193 = wp::add(var_191, var_192);
    // b = wp.dot(scales_squared, b0_b1)                                                      <L 375>
    var_194 = wp::dot(var_142, var_174);
    // c = wp.dot(scales_squared, b1_squared) + FILTER_VARIANCE                               <L 376>
    var_195 = wp::dot(var_142, var_190);
    var_196 = wp::add(var_195, var_192);
    // return wp.vec3(a, b, c)                                                                <L 377>
    var_197 = wp::vec_t<3, wp::float32>(var_193, var_194, var_196);
    return var_197;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:380
static CUDA_CALLABLE wp::vec_t<3, wp::float32> conic_at_centre_0(
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_z,
    wp::float32 var_focal)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1e-08;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // forward
    // def conic_at_centre(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44, x: float, y: float, z: float, focal: float):       <L 381>
    // covariance = projected_covariance(log_scale, quaternion, camera, x, y, z, focal)       <L 382>
    var_0 = projected_covariance_0(var_log_scale, var_quaternion, var_camera, var_x, var_y, var_z, var_focal);
    // a, b, c = covariance[0], covariance[1], covariance[2]                                  <L 383>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_0, var_5);
    // determinant = wp.max(a * c - b * b, 1.0e-8)                                            <L 384>
    var_7 = wp::mul(var_2, var_6);
    var_8 = wp::mul(var_4, var_4);
    var_9 = wp::sub(var_7, var_8);
    var_11 = wp::max(var_9, var_10);
    // return wp.vec3(c / determinant, -b / determinant, a / determinant)                     <L 385>
    var_12 = wp::div(var_6, var_11);
    var_13 = wp::neg(var_4);
    var_14 = wp::div(var_13, var_11);
    var_15 = wp::div(var_2, var_11);
    var_16 = wp::vec_t<3, wp::float32>(var_12, var_14, var_15);
    return var_16;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:388
static CUDA_CALLABLE wp::float32 alpha_at_pixel_0(
    wp::vec_t<3, wp::float32> var_mean,
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::float32 var_opacity_logit,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_px,
    wp::float32 var_py,
    wp::float32 var_width,
    wp::float32 var_height,
    wp::float32 var_focal,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 1.0;
    wp::vec_t<4, wp::float32> var_7;
    wp::vec_t<4, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.1;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.5;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 2.0;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 2;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 1.0;
    const wp::float32 var_51 = 1.0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 9.0;
    wp::float32 var_57;
    const wp::int32 var_58 = 0;
    bool var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = 2.0;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    wp::float32 var_72;
    bool var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    bool var_76;
    const wp::float32 var_77 = -0.5;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.99;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.00392156862745098;
    bool var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    //---------
    // forward
    // def alpha_at_pixel(                                                                    <L 389>
    // point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)                               <L 413>
    var_1 = wp::extract(var_mean, var_0);
    var_3 = wp::extract(var_mean, var_2);
    var_5 = wp::extract(var_mean, var_4);
    var_7 = wp::vec_t<4, wp::float32>(var_1, var_3, var_5, var_6);
    var_8 = wp::mul(var_camera, var_7);
    // z = wp.max(point[2], NEAR_PLANE)                                                       <L 414>
    var_10 = wp::extract(var_8, var_9);
    var_12 = wp::max(var_10, var_11);
    // centre_x = focal * point[0] / z + 0.5 * width                                          <L 415>
    var_14 = wp::extract(var_8, var_13);
    var_15 = wp::mul(var_focal, var_14);
    var_16 = wp::div(var_15, var_12);
    var_18 = wp::mul(var_17, var_width);
    var_19 = wp::add(var_16, var_18);
    // centre_y = focal * point[1] / z + 0.5 * height                                         <L 416>
    var_21 = wp::extract(var_8, var_20);
    var_22 = wp::mul(var_focal, var_21);
    var_23 = wp::div(var_22, var_12);
    var_25 = wp::mul(var_24, var_height);
    var_26 = wp::add(var_23, var_25);
    // conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)       <L 417>
    var_28 = wp::extract(var_8, var_27);
    var_30 = wp::extract(var_8, var_29);
    var_31 = conic_at_centre_0(var_log_scale, var_quaternion, var_camera, var_28, var_30, var_12, var_focal);
    // dx, dy = px - centre_x, py - centre_y                                                  <L 418>
    var_32 = wp::sub(var_px, var_19);
    var_33 = wp::sub(var_py, var_26);
    // q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy                 <L 419>
    var_35 = wp::extract(var_31, var_34);
    var_36 = wp::mul(var_35, var_32);
    var_37 = wp::mul(var_36, var_32);
    var_40 = wp::extract(var_31, var_39);
    var_41 = wp::mul(var_38, var_40);
    var_42 = wp::mul(var_41, var_32);
    var_43 = wp::mul(var_42, var_33);
    var_44 = wp::add(var_37, var_43);
    var_46 = wp::extract(var_31, var_45);
    var_47 = wp::mul(var_46, var_33);
    var_48 = wp::mul(var_47, var_33);
    var_49 = wp::add(var_44, var_48);
    // opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))                                         <L 420>
    var_52 = wp::neg(var_opacity_logit);
    var_53 = wp::exp(var_52);
    var_54 = wp::add(var_51, var_53);
    var_55 = wp::div(var_50, var_54);
    // support = float(SUPPORT_RADIUS_SQUARED)                                                <L 421>
    var_57 = wp::float(var_56);
    // if compact_enabled != 0:                                                               <L 422>
    var_59 = (var_compact_enabled != var_58);
    if (var_59) {
        // support = 0.0                                                                      <L 423>
        // if opacity > compact_alpha_min:                                                    <L 424>
        var_61 = (var_55 > var_compact_alpha_min);
        if (var_61) {
            // support = wp.min(                                                              <L 425>
            // float(SUPPORT_RADIUS_SQUARED),                                                 <L 426>
            var_62 = wp::float(var_56);
            // compact_beta * 2.0 * wp.log(opacity / compact_alpha_min),                      <L 427>
            var_64 = wp::mul(var_compact_beta, var_63);
            var_65 = wp::div(var_55, var_compact_alpha_min);
            var_66 = wp::log(var_65);
            var_67 = wp::mul(var_64, var_66);
            var_68 = wp::min(var_62, var_67);
        }
        var_69 = wp::where(var_61, var_68, var_60);
    }
    var_70 = wp::where(var_59, var_69, var_57);
    // alpha = float(0.0)                                                                     <L 429>
    var_72 = wp::float(var_71);
    // if support > 0.0 and q <= support:                                                     <L 430>
    var_75 = (var_70 > var_74);
    var_73 = var_75;
    if (var_73) {
        var_76 = (var_49 <= var_70);
        var_73 = var_73 && var_76;
    }
    if (var_73) {
        // candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)                               <L 431>
        var_78 = wp::mul(var_77, var_49);
        var_79 = wp::exp(var_78);
        var_80 = wp::mul(var_55, var_79);
        var_82 = wp::min(var_80, var_81);
        // if candidate >= ALPHA_CUTOFF:                                                      <L 432>
        var_84 = (var_82 >= var_83);
        if (var_84) {
            // alpha = candidate                                                              <L 433>
            var_85 = wp::copy(var_82);
        }
        var_86 = wp::where(var_84, var_85, var_72);
    }
    var_87 = wp::where(var_73, var_86, var_72);
    // return alpha                                                                           <L 434>
    return var_87;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:437
static CUDA_CALLABLE wp::vec_t<3, wp::float32> colour_at_view_0(
    wp::vec_t<3, wp::float32> var_color)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 1.0;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // forward
    // def colour_at_view(                                                                    <L 438>
    // return wp.vec3(                                                                        <L 441>
    // wp.clamp(color[0], 0.0, 1.0),                                                          <L 442>
    var_1 = wp::extract(var_color, var_0);
    var_4 = wp::clamp(var_1, var_2, var_3);
    // wp.clamp(color[1], 0.0, 1.0),                                                          <L 443>
    var_6 = wp::extract(var_color, var_5);
    var_9 = wp::clamp(var_6, var_7, var_8);
    // wp.clamp(color[2], 0.0, 1.0),                                                          <L 444>
    var_11 = wp::extract(var_color, var_10);
    var_14 = wp::clamp(var_11, var_12, var_13);
    var_15 = wp::vec_t<3, wp::float32>(var_4, var_9, var_14);
    return var_15;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:346
static CUDA_CALLABLE void adj_projected_covariance_0(
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_z,
    wp::float32 var_focal,
    wp::vec_t<3, wp::float32> & adj_log_scale,
    wp::vec_t<4, wp::float32> & adj_quaternion,
    wp::mat_t<4, 4, wp::float32> & adj_camera,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_z,
    wp::float32 & adj_focal,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-08;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::vec_t<4, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::int32 var_11 = 3;
    wp::float32 var_12;
    const wp::float32 var_13 = 1.0;
    const wp::float32 var_14 = 2.0;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 2.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 2.0;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 1.0;
    const wp::float32 var_37 = 2.0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 2.0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::float32 var_49 = 2.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 2.0;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 1.0;
    const wp::float32 var_60 = 2.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 0;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 1;
    const wp::int32 var_71 = 0;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    const wp::int32 var_74 = 0;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    const wp::int32 var_77 = 0;
    const wp::int32 var_78 = 1;
    wp::float32 var_79;
    const wp::int32 var_80 = 1;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    const wp::int32 var_83 = 2;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 0;
    const wp::int32 var_88 = 2;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.0;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    const wp::float32 var_104 = 0.0;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::vec_t<3, wp::float32> var_126;
    const wp::float32 var_127 = 2.0;
    const wp::int32 var_128 = 0;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::float32 var_132 = 2.0;
    const wp::int32 var_133 = 1;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::float32 var_137 = 2.0;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::vec_t<3, wp::float32> var_142;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    const wp::int32 var_145 = 0;
    wp::float32 var_146;
    wp::float32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::int32 var_153 = 2;
    wp::float32 var_154;
    const wp::int32 var_155 = 2;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::vec_t<3, wp::float32> var_158;
    const wp::int32 var_159 = 0;
    wp::float32 var_160;
    const wp::int32 var_161 = 0;
    wp::float32 var_162;
    wp::float32 var_163;
    const wp::int32 var_164 = 1;
    wp::float32 var_165;
    const wp::int32 var_166 = 1;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 2;
    wp::float32 var_170;
    const wp::int32 var_171 = 2;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 0;
    wp::float32 var_176;
    const wp::int32 var_177 = 0;
    wp::float32 var_178;
    wp::float32 var_179;
    const wp::int32 var_180 = 1;
    wp::float32 var_181;
    const wp::int32 var_182 = 1;
    wp::float32 var_183;
    wp::float32 var_184;
    const wp::int32 var_185 = 2;
    wp::float32 var_186;
    const wp::int32 var_187 = 2;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::vec_t<3, wp::float32> var_190;
    wp::float32 var_191;
    const wp::float32 var_192 = 0.3;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    wp::vec_t<3, wp::float32> var_197;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::vec_t<4, wp::float32> adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::int32 adj_11 = {};
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
    wp::vec_t<3, wp::float32> adj_30 = {};
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
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::vec_t<3, wp::float32> adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::vec_t<3, wp::float32> adj_66 = {};
    wp::int32 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::int32 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::float32 adj_72 = {};
    wp::int32 adj_73 = {};
    wp::int32 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::vec_t<3, wp::float32> adj_76 = {};
    wp::int32 adj_77 = {};
    wp::int32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::int32 adj_80 = {};
    wp::int32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::int32 adj_83 = {};
    wp::int32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::vec_t<3, wp::float32> adj_86 = {};
    wp::int32 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::int32 adj_90 = {};
    wp::int32 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::int32 adj_93 = {};
    wp::int32 adj_94 = {};
    wp::float32 adj_95 = {};
    wp::vec_t<3, wp::float32> adj_96 = {};
    wp::float32 adj_97 = {};
    wp::float32 adj_98 = {};
    wp::float32 adj_99 = {};
    wp::float32 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float32 adj_102 = {};
    wp::vec_t<3, wp::float32> adj_103 = {};
    wp::float32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::float32 adj_107 = {};
    wp::float32 adj_108 = {};
    wp::float32 adj_109 = {};
    wp::vec_t<3, wp::float32> adj_110 = {};
    wp::float32 adj_111 = {};
    wp::float32 adj_112 = {};
    wp::float32 adj_113 = {};
    wp::vec_t<3, wp::float32> adj_114 = {};
    wp::float32 adj_115 = {};
    wp::float32 adj_116 = {};
    wp::float32 adj_117 = {};
    wp::vec_t<3, wp::float32> adj_118 = {};
    wp::float32 adj_119 = {};
    wp::float32 adj_120 = {};
    wp::float32 adj_121 = {};
    wp::vec_t<3, wp::float32> adj_122 = {};
    wp::float32 adj_123 = {};
    wp::float32 adj_124 = {};
    wp::float32 adj_125 = {};
    wp::vec_t<3, wp::float32> adj_126 = {};
    wp::float32 adj_127 = {};
    wp::int32 adj_128 = {};
    wp::float32 adj_129 = {};
    wp::float32 adj_130 = {};
    wp::float32 adj_131 = {};
    wp::float32 adj_132 = {};
    wp::int32 adj_133 = {};
    wp::float32 adj_134 = {};
    wp::float32 adj_135 = {};
    wp::float32 adj_136 = {};
    wp::float32 adj_137 = {};
    wp::int32 adj_138 = {};
    wp::float32 adj_139 = {};
    wp::float32 adj_140 = {};
    wp::float32 adj_141 = {};
    wp::vec_t<3, wp::float32> adj_142 = {};
    wp::int32 adj_143 = {};
    wp::float32 adj_144 = {};
    wp::int32 adj_145 = {};
    wp::float32 adj_146 = {};
    wp::float32 adj_147 = {};
    wp::int32 adj_148 = {};
    wp::float32 adj_149 = {};
    wp::int32 adj_150 = {};
    wp::float32 adj_151 = {};
    wp::float32 adj_152 = {};
    wp::int32 adj_153 = {};
    wp::float32 adj_154 = {};
    wp::int32 adj_155 = {};
    wp::float32 adj_156 = {};
    wp::float32 adj_157 = {};
    wp::vec_t<3, wp::float32> adj_158 = {};
    wp::int32 adj_159 = {};
    wp::float32 adj_160 = {};
    wp::int32 adj_161 = {};
    wp::float32 adj_162 = {};
    wp::float32 adj_163 = {};
    wp::int32 adj_164 = {};
    wp::float32 adj_165 = {};
    wp::int32 adj_166 = {};
    wp::float32 adj_167 = {};
    wp::float32 adj_168 = {};
    wp::int32 adj_169 = {};
    wp::float32 adj_170 = {};
    wp::int32 adj_171 = {};
    wp::float32 adj_172 = {};
    wp::float32 adj_173 = {};
    wp::vec_t<3, wp::float32> adj_174 = {};
    wp::int32 adj_175 = {};
    wp::float32 adj_176 = {};
    wp::int32 adj_177 = {};
    wp::float32 adj_178 = {};
    wp::float32 adj_179 = {};
    wp::int32 adj_180 = {};
    wp::float32 adj_181 = {};
    wp::int32 adj_182 = {};
    wp::float32 adj_183 = {};
    wp::float32 adj_184 = {};
    wp::int32 adj_185 = {};
    wp::float32 adj_186 = {};
    wp::int32 adj_187 = {};
    wp::float32 adj_188 = {};
    wp::float32 adj_189 = {};
    wp::vec_t<3, wp::float32> adj_190 = {};
    wp::float32 adj_191 = {};
    wp::float32 adj_192 = {};
    wp::float32 adj_193 = {};
    wp::float32 adj_194 = {};
    wp::float32 adj_195 = {};
    wp::float32 adj_196 = {};
    wp::vec_t<3, wp::float32> adj_197 = {};
    //---------
    // forward
    // def projected_covariance(                                                              <L 347>
    // quaternion = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)             <L 356>
    var_0 = wp::dot(var_quaternion, var_quaternion);
    var_2 = wp::add(var_0, var_1);
    var_3 = wp::sqrt(var_2);
    var_4 = wp::div(var_quaternion, var_3);
    // qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]            <L 357>
    var_6 = wp::extract(var_4, var_5);
    var_8 = wp::extract(var_4, var_7);
    var_10 = wp::extract(var_4, var_9);
    var_12 = wp::extract(var_4, var_11);
    // r0 = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy))       <L 358>
    var_15 = wp::mul(var_10, var_10);
    var_16 = wp::mul(var_12, var_12);
    var_17 = wp::add(var_15, var_16);
    var_18 = wp::mul(var_14, var_17);
    var_19 = wp::sub(var_13, var_18);
    var_21 = wp::mul(var_8, var_10);
    var_22 = wp::mul(var_6, var_12);
    var_23 = wp::add(var_21, var_22);
    var_24 = wp::mul(var_20, var_23);
    var_26 = wp::mul(var_8, var_12);
    var_27 = wp::mul(var_6, var_10);
    var_28 = wp::sub(var_26, var_27);
    var_29 = wp::mul(var_25, var_28);
    var_30 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
    // r1 = wp.vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx))       <L 359>
    var_32 = wp::mul(var_8, var_10);
    var_33 = wp::mul(var_6, var_12);
    var_34 = wp::sub(var_32, var_33);
    var_35 = wp::mul(var_31, var_34);
    var_38 = wp::mul(var_8, var_8);
    var_39 = wp::mul(var_12, var_12);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_37, var_40);
    var_42 = wp::sub(var_36, var_41);
    var_44 = wp::mul(var_10, var_12);
    var_45 = wp::mul(var_6, var_8);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_43, var_46);
    var_48 = wp::vec_t<3, wp::float32>(var_35, var_42, var_47);
    // r2 = wp.vec3(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))       <L 360>
    var_50 = wp::mul(var_8, var_12);
    var_51 = wp::mul(var_6, var_10);
    var_52 = wp::add(var_50, var_51);
    var_53 = wp::mul(var_49, var_52);
    var_55 = wp::mul(var_10, var_12);
    var_56 = wp::mul(var_6, var_8);
    var_57 = wp::sub(var_55, var_56);
    var_58 = wp::mul(var_54, var_57);
    var_61 = wp::mul(var_8, var_8);
    var_62 = wp::mul(var_10, var_10);
    var_63 = wp::add(var_61, var_62);
    var_64 = wp::mul(var_60, var_63);
    var_65 = wp::sub(var_59, var_64);
    var_66 = wp::vec_t<3, wp::float32>(var_53, var_58, var_65);
    // c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])                                 <L 361>
    var_69 = wp::extract(var_camera, var_67, var_68);
    var_72 = wp::extract(var_camera, var_70, var_71);
    var_75 = wp::extract(var_camera, var_73, var_74);
    var_76 = wp::vec_t<3, wp::float32>(var_69, var_72, var_75);
    // c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])                                 <L 362>
    var_79 = wp::extract(var_camera, var_77, var_78);
    var_82 = wp::extract(var_camera, var_80, var_81);
    var_85 = wp::extract(var_camera, var_83, var_84);
    var_86 = wp::vec_t<3, wp::float32>(var_79, var_82, var_85);
    // c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])                                 <L 363>
    var_89 = wp::extract(var_camera, var_87, var_88);
    var_92 = wp::extract(var_camera, var_90, var_91);
    var_95 = wp::extract(var_camera, var_93, var_94);
    var_96 = wp::vec_t<3, wp::float32>(var_89, var_92, var_95);
    // j0 = wp.vec3(focal / z, 0.0, -focal * x / (z * z))                                     <L 364>
    var_97 = wp::div(var_focal, var_z);
    var_99 = wp::neg(var_focal);
    var_100 = wp::mul(var_99, var_x);
    var_101 = wp::mul(var_z, var_z);
    var_102 = wp::div(var_100, var_101);
    var_103 = wp::vec_t<3, wp::float32>(var_97, var_98, var_102);
    // j1 = wp.vec3(0.0, focal / z, -focal * y / (z * z))                                     <L 365>
    var_105 = wp::div(var_focal, var_z);
    var_106 = wp::neg(var_focal);
    var_107 = wp::mul(var_106, var_y);
    var_108 = wp::mul(var_z, var_z);
    var_109 = wp::div(var_107, var_108);
    var_110 = wp::vec_t<3, wp::float32>(var_104, var_105, var_109);
    // p0 = wp.vec3(wp.dot(j0, c0), wp.dot(j0, c1), wp.dot(j0, c2))                           <L 366>
    var_111 = wp::dot(var_103, var_76);
    var_112 = wp::dot(var_103, var_86);
    var_113 = wp::dot(var_103, var_96);
    var_114 = wp::vec_t<3, wp::float32>(var_111, var_112, var_113);
    // p1 = wp.vec3(wp.dot(j1, c0), wp.dot(j1, c1), wp.dot(j1, c2))                           <L 367>
    var_115 = wp::dot(var_110, var_76);
    var_116 = wp::dot(var_110, var_86);
    var_117 = wp::dot(var_110, var_96);
    var_118 = wp::vec_t<3, wp::float32>(var_115, var_116, var_117);
    // b0 = wp.vec3(wp.dot(p0, r0), wp.dot(p0, r1), wp.dot(p0, r2))                           <L 368>
    var_119 = wp::dot(var_114, var_30);
    var_120 = wp::dot(var_114, var_48);
    var_121 = wp::dot(var_114, var_66);
    var_122 = wp::vec_t<3, wp::float32>(var_119, var_120, var_121);
    // b1 = wp.vec3(wp.dot(p1, r0), wp.dot(p1, r1), wp.dot(p1, r2))                           <L 369>
    var_123 = wp::dot(var_118, var_30);
    var_124 = wp::dot(var_118, var_48);
    var_125 = wp::dot(var_118, var_66);
    var_126 = wp::vec_t<3, wp::float32>(var_123, var_124, var_125);
    // scales_squared = wp.vec3(wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1]), wp.exp(2.0 * log_scale[2]))       <L 370>
    var_129 = wp::extract(var_log_scale, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::exp(var_130);
    var_134 = wp::extract(var_log_scale, var_133);
    var_135 = wp::mul(var_132, var_134);
    var_136 = wp::exp(var_135);
    var_139 = wp::extract(var_log_scale, var_138);
    var_140 = wp::mul(var_137, var_139);
    var_141 = wp::exp(var_140);
    var_142 = wp::vec_t<3, wp::float32>(var_131, var_136, var_141);
    // b0_squared = wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])                      <L 371>
    var_144 = wp::extract(var_122, var_143);
    var_146 = wp::extract(var_122, var_145);
    var_147 = wp::mul(var_144, var_146);
    var_149 = wp::extract(var_122, var_148);
    var_151 = wp::extract(var_122, var_150);
    var_152 = wp::mul(var_149, var_151);
    var_154 = wp::extract(var_122, var_153);
    var_156 = wp::extract(var_122, var_155);
    var_157 = wp::mul(var_154, var_156);
    var_158 = wp::vec_t<3, wp::float32>(var_147, var_152, var_157);
    // b0_b1 = wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2])                           <L 372>
    var_160 = wp::extract(var_122, var_159);
    var_162 = wp::extract(var_126, var_161);
    var_163 = wp::mul(var_160, var_162);
    var_165 = wp::extract(var_122, var_164);
    var_167 = wp::extract(var_126, var_166);
    var_168 = wp::mul(var_165, var_167);
    var_170 = wp::extract(var_122, var_169);
    var_172 = wp::extract(var_126, var_171);
    var_173 = wp::mul(var_170, var_172);
    var_174 = wp::vec_t<3, wp::float32>(var_163, var_168, var_173);
    // b1_squared = wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])                      <L 373>
    var_176 = wp::extract(var_126, var_175);
    var_178 = wp::extract(var_126, var_177);
    var_179 = wp::mul(var_176, var_178);
    var_181 = wp::extract(var_126, var_180);
    var_183 = wp::extract(var_126, var_182);
    var_184 = wp::mul(var_181, var_183);
    var_186 = wp::extract(var_126, var_185);
    var_188 = wp::extract(var_126, var_187);
    var_189 = wp::mul(var_186, var_188);
    var_190 = wp::vec_t<3, wp::float32>(var_179, var_184, var_189);
    // a = wp.dot(scales_squared, b0_squared) + FILTER_VARIANCE                               <L 374>
    var_191 = wp::dot(var_142, var_158);
    var_193 = wp::add(var_191, var_192);
    // b = wp.dot(scales_squared, b0_b1)                                                      <L 375>
    var_194 = wp::dot(var_142, var_174);
    // c = wp.dot(scales_squared, b1_squared) + FILTER_VARIANCE                               <L 376>
    var_195 = wp::dot(var_142, var_190);
    var_196 = wp::add(var_195, var_192);
    // return wp.vec3(a, b, c)                                                                <L 377>
    var_197 = wp::vec_t<3, wp::float32>(var_193, var_194, var_196);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_197 += adj_ret;
    wp::adj_vec_t(var_193, var_194, var_196, adj_193, adj_194, adj_196, adj_197);
    // adj: return wp.vec3(a, b, c)                                                           <L 377>
    wp::adj_add(var_195, var_192, adj_195, adj_192, adj_196);
    wp::adj_dot(var_142, var_190, adj_142, adj_190, adj_195);
    // adj: c = wp.dot(scales_squared, b1_squared) + FILTER_VARIANCE                          <L 376>
    wp::adj_dot(var_142, var_174, adj_142, adj_174, adj_194);
    // adj: b = wp.dot(scales_squared, b0_b1)                                                 <L 375>
    wp::adj_add(var_191, var_192, adj_191, adj_192, adj_193);
    wp::adj_dot(var_142, var_158, adj_142, adj_158, adj_191);
    // adj: a = wp.dot(scales_squared, b0_squared) + FILTER_VARIANCE                          <L 374>
    wp::adj_vec_t(var_179, var_184, var_189, adj_179, adj_184, adj_189, adj_190);
    wp::adj_mul(var_186, var_188, adj_186, adj_188, adj_189);
    wp::adj_extract(var_126, var_187, adj_126, adj_187, adj_188);
    wp::adj_extract(var_126, var_185, adj_126, adj_185, adj_186);
    wp::adj_mul(var_181, var_183, adj_181, adj_183, adj_184);
    wp::adj_extract(var_126, var_182, adj_126, adj_182, adj_183);
    wp::adj_extract(var_126, var_180, adj_126, adj_180, adj_181);
    wp::adj_mul(var_176, var_178, adj_176, adj_178, adj_179);
    wp::adj_extract(var_126, var_177, adj_126, adj_177, adj_178);
    wp::adj_extract(var_126, var_175, adj_126, adj_175, adj_176);
    // adj: b1_squared = wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])                 <L 373>
    wp::adj_vec_t(var_163, var_168, var_173, adj_163, adj_168, adj_173, adj_174);
    wp::adj_mul(var_170, var_172, adj_170, adj_172, adj_173);
    wp::adj_extract(var_126, var_171, adj_126, adj_171, adj_172);
    wp::adj_extract(var_122, var_169, adj_122, adj_169, adj_170);
    wp::adj_mul(var_165, var_167, adj_165, adj_167, adj_168);
    wp::adj_extract(var_126, var_166, adj_126, adj_166, adj_167);
    wp::adj_extract(var_122, var_164, adj_122, adj_164, adj_165);
    wp::adj_mul(var_160, var_162, adj_160, adj_162, adj_163);
    wp::adj_extract(var_126, var_161, adj_126, adj_161, adj_162);
    wp::adj_extract(var_122, var_159, adj_122, adj_159, adj_160);
    // adj: b0_b1 = wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2])                      <L 372>
    wp::adj_vec_t(var_147, var_152, var_157, adj_147, adj_152, adj_157, adj_158);
    wp::adj_mul(var_154, var_156, adj_154, adj_156, adj_157);
    wp::adj_extract(var_122, var_155, adj_122, adj_155, adj_156);
    wp::adj_extract(var_122, var_153, adj_122, adj_153, adj_154);
    wp::adj_mul(var_149, var_151, adj_149, adj_151, adj_152);
    wp::adj_extract(var_122, var_150, adj_122, adj_150, adj_151);
    wp::adj_extract(var_122, var_148, adj_122, adj_148, adj_149);
    wp::adj_mul(var_144, var_146, adj_144, adj_146, adj_147);
    wp::adj_extract(var_122, var_145, adj_122, adj_145, adj_146);
    wp::adj_extract(var_122, var_143, adj_122, adj_143, adj_144);
    // adj: b0_squared = wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])                 <L 371>
    wp::adj_vec_t(var_131, var_136, var_141, adj_131, adj_136, adj_141, adj_142);
    wp::adj_exp(var_140, var_141, adj_140, adj_141);
    wp::adj_mul(var_137, var_139, adj_137, adj_139, adj_140);
    wp::adj_extract(var_log_scale, var_138, adj_log_scale, adj_138, adj_139);
    wp::adj_exp(var_135, var_136, adj_135, adj_136);
    wp::adj_mul(var_132, var_134, adj_132, adj_134, adj_135);
    wp::adj_extract(var_log_scale, var_133, adj_log_scale, adj_133, adj_134);
    wp::adj_exp(var_130, var_131, adj_130, adj_131);
    wp::adj_mul(var_127, var_129, adj_127, adj_129, adj_130);
    wp::adj_extract(var_log_scale, var_128, adj_log_scale, adj_128, adj_129);
    // adj: scales_squared = wp.vec3(wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1]), wp.exp(2.0 * log_scale[2]))  <L 370>
    wp::adj_vec_t(var_123, var_124, var_125, adj_123, adj_124, adj_125, adj_126);
    wp::adj_dot(var_118, var_66, adj_118, adj_66, adj_125);
    wp::adj_dot(var_118, var_48, adj_118, adj_48, adj_124);
    wp::adj_dot(var_118, var_30, adj_118, adj_30, adj_123);
    // adj: b1 = wp.vec3(wp.dot(p1, r0), wp.dot(p1, r1), wp.dot(p1, r2))                      <L 369>
    wp::adj_vec_t(var_119, var_120, var_121, adj_119, adj_120, adj_121, adj_122);
    wp::adj_dot(var_114, var_66, adj_114, adj_66, adj_121);
    wp::adj_dot(var_114, var_48, adj_114, adj_48, adj_120);
    wp::adj_dot(var_114, var_30, adj_114, adj_30, adj_119);
    // adj: b0 = wp.vec3(wp.dot(p0, r0), wp.dot(p0, r1), wp.dot(p0, r2))                      <L 368>
    wp::adj_vec_t(var_115, var_116, var_117, adj_115, adj_116, adj_117, adj_118);
    wp::adj_dot(var_110, var_96, adj_110, adj_96, adj_117);
    wp::adj_dot(var_110, var_86, adj_110, adj_86, adj_116);
    wp::adj_dot(var_110, var_76, adj_110, adj_76, adj_115);
    // adj: p1 = wp.vec3(wp.dot(j1, c0), wp.dot(j1, c1), wp.dot(j1, c2))                      <L 367>
    wp::adj_vec_t(var_111, var_112, var_113, adj_111, adj_112, adj_113, adj_114);
    wp::adj_dot(var_103, var_96, adj_103, adj_96, adj_113);
    wp::adj_dot(var_103, var_86, adj_103, adj_86, adj_112);
    wp::adj_dot(var_103, var_76, adj_103, adj_76, adj_111);
    // adj: p0 = wp.vec3(wp.dot(j0, c0), wp.dot(j0, c1), wp.dot(j0, c2))                      <L 366>
    wp::adj_vec_t(var_104, var_105, var_109, adj_104, adj_105, adj_109, adj_110);
    wp::adj_div(var_107, var_108, var_109, adj_107, adj_108, adj_109);
    wp::adj_mul(var_z, var_z, adj_z, adj_z, adj_108);
    wp::adj_mul(var_106, var_y, adj_106, adj_y, adj_107);
    wp::adj_neg(var_focal, adj_focal, adj_106);
    wp::adj_div(var_focal, var_z, var_105, adj_focal, adj_z, adj_105);
    // adj: j1 = wp.vec3(0.0, focal / z, -focal * y / (z * z))                                <L 365>
    wp::adj_vec_t(var_97, var_98, var_102, adj_97, adj_98, adj_102, adj_103);
    wp::adj_div(var_100, var_101, var_102, adj_100, adj_101, adj_102);
    wp::adj_mul(var_z, var_z, adj_z, adj_z, adj_101);
    wp::adj_mul(var_99, var_x, adj_99, adj_x, adj_100);
    wp::adj_neg(var_focal, adj_focal, adj_99);
    wp::adj_div(var_focal, var_z, var_97, adj_focal, adj_z, adj_97);
    // adj: j0 = wp.vec3(focal / z, 0.0, -focal * x / (z * z))                                <L 364>
    wp::adj_vec_t(var_89, var_92, var_95, adj_89, adj_92, adj_95, adj_96);
    wp::adj_extract(var_camera, var_93, var_94, adj_camera, adj_93, adj_94, adj_95);
    wp::adj_extract(var_camera, var_90, var_91, adj_camera, adj_90, adj_91, adj_92);
    wp::adj_extract(var_camera, var_87, var_88, adj_camera, adj_87, adj_88, adj_89);
    // adj: c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])                            <L 363>
    wp::adj_vec_t(var_79, var_82, var_85, adj_79, adj_82, adj_85, adj_86);
    wp::adj_extract(var_camera, var_83, var_84, adj_camera, adj_83, adj_84, adj_85);
    wp::adj_extract(var_camera, var_80, var_81, adj_camera, adj_80, adj_81, adj_82);
    wp::adj_extract(var_camera, var_77, var_78, adj_camera, adj_77, adj_78, adj_79);
    // adj: c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])                            <L 362>
    wp::adj_vec_t(var_69, var_72, var_75, adj_69, adj_72, adj_75, adj_76);
    wp::adj_extract(var_camera, var_73, var_74, adj_camera, adj_73, adj_74, adj_75);
    wp::adj_extract(var_camera, var_70, var_71, adj_camera, adj_70, adj_71, adj_72);
    wp::adj_extract(var_camera, var_67, var_68, adj_camera, adj_67, adj_68, adj_69);
    // adj: c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])                            <L 361>
    wp::adj_vec_t(var_53, var_58, var_65, adj_53, adj_58, adj_65, adj_66);
    wp::adj_sub(var_59, var_64, adj_59, adj_64, adj_65);
    wp::adj_mul(var_60, var_63, adj_60, adj_63, adj_64);
    wp::adj_add(var_61, var_62, adj_61, adj_62, adj_63);
    wp::adj_mul(var_10, var_10, adj_10, adj_10, adj_62);
    wp::adj_mul(var_8, var_8, adj_8, adj_8, adj_61);
    wp::adj_mul(var_54, var_57, adj_54, adj_57, adj_58);
    wp::adj_sub(var_55, var_56, adj_55, adj_56, adj_57);
    wp::adj_mul(var_6, var_8, adj_6, adj_8, adj_56);
    wp::adj_mul(var_10, var_12, adj_10, adj_12, adj_55);
    wp::adj_mul(var_49, var_52, adj_49, adj_52, adj_53);
    wp::adj_add(var_50, var_51, adj_50, adj_51, adj_52);
    wp::adj_mul(var_6, var_10, adj_6, adj_10, adj_51);
    wp::adj_mul(var_8, var_12, adj_8, adj_12, adj_50);
    // adj: r2 = wp.vec3(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))  <L 360>
    wp::adj_vec_t(var_35, var_42, var_47, adj_35, adj_42, adj_47, adj_48);
    wp::adj_mul(var_43, var_46, adj_43, adj_46, adj_47);
    wp::adj_add(var_44, var_45, adj_44, adj_45, adj_46);
    wp::adj_mul(var_6, var_8, adj_6, adj_8, adj_45);
    wp::adj_mul(var_10, var_12, adj_10, adj_12, adj_44);
    wp::adj_sub(var_36, var_41, adj_36, adj_41, adj_42);
    wp::adj_mul(var_37, var_40, adj_37, adj_40, adj_41);
    wp::adj_add(var_38, var_39, adj_38, adj_39, adj_40);
    wp::adj_mul(var_12, var_12, adj_12, adj_12, adj_39);
    wp::adj_mul(var_8, var_8, adj_8, adj_8, adj_38);
    wp::adj_mul(var_31, var_34, adj_31, adj_34, adj_35);
    wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_34);
    wp::adj_mul(var_6, var_12, adj_6, adj_12, adj_33);
    wp::adj_mul(var_8, var_10, adj_8, adj_10, adj_32);
    // adj: r1 = wp.vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx))  <L 359>
    wp::adj_vec_t(var_19, var_24, var_29, adj_19, adj_24, adj_29, adj_30);
    wp::adj_mul(var_25, var_28, adj_25, adj_28, adj_29);
    wp::adj_sub(var_26, var_27, adj_26, adj_27, adj_28);
    wp::adj_mul(var_6, var_10, adj_6, adj_10, adj_27);
    wp::adj_mul(var_8, var_12, adj_8, adj_12, adj_26);
    wp::adj_mul(var_20, var_23, adj_20, adj_23, adj_24);
    wp::adj_add(var_21, var_22, adj_21, adj_22, adj_23);
    wp::adj_mul(var_6, var_12, adj_6, adj_12, adj_22);
    wp::adj_mul(var_8, var_10, adj_8, adj_10, adj_21);
    wp::adj_sub(var_13, var_18, adj_13, adj_18, adj_19);
    wp::adj_mul(var_14, var_17, adj_14, adj_17, adj_18);
    wp::adj_add(var_15, var_16, adj_15, adj_16, adj_17);
    wp::adj_mul(var_12, var_12, adj_12, adj_12, adj_16);
    wp::adj_mul(var_10, var_10, adj_10, adj_10, adj_15);
    // adj: r0 = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy))  <L 358>
    wp::adj_extract(var_4, var_11, adj_4, adj_11, adj_12);
    wp::adj_extract(var_4, var_9, adj_4, adj_9, adj_10);
    wp::adj_extract(var_4, var_7, adj_4, adj_7, adj_8);
    wp::adj_extract(var_4, var_5, adj_4, adj_5, adj_6);
    // adj: qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]       <L 357>
    wp::adj_div(var_quaternion, var_3, adj_quaternion, adj_3, adj_4);
    wp::adj_sqrt(var_2, var_3, adj_2, adj_3);
    wp::adj_add(var_0, var_1, adj_0, adj_1, adj_2);
    wp::adj_dot(var_quaternion, var_quaternion, adj_quaternion, adj_quaternion, adj_0);
    // adj: quaternion = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)        <L 356>
    // adj: def projected_covariance(                                                         <L 347>
    return;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:380
static CUDA_CALLABLE void adj_conic_at_centre_0(
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 var_z,
    wp::float32 var_focal,
    wp::vec_t<3, wp::float32> & adj_log_scale,
    wp::vec_t<4, wp::float32> & adj_quaternion,
    wp::mat_t<4, 4, wp::float32> & adj_camera,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_z,
    wp::float32 & adj_focal,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1e-08;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
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
    wp::vec_t<3, wp::float32> adj_16 = {};
    //---------
    // forward
    // def conic_at_centre(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44, x: float, y: float, z: float, focal: float):       <L 381>
    // covariance = projected_covariance(log_scale, quaternion, camera, x, y, z, focal)       <L 382>
    var_0 = projected_covariance_0(var_log_scale, var_quaternion, var_camera, var_x, var_y, var_z, var_focal);
    // a, b, c = covariance[0], covariance[1], covariance[2]                                  <L 383>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_0, var_5);
    // determinant = wp.max(a * c - b * b, 1.0e-8)                                            <L 384>
    var_7 = wp::mul(var_2, var_6);
    var_8 = wp::mul(var_4, var_4);
    var_9 = wp::sub(var_7, var_8);
    var_11 = wp::max(var_9, var_10);
    // return wp.vec3(c / determinant, -b / determinant, a / determinant)                     <L 385>
    var_12 = wp::div(var_6, var_11);
    var_13 = wp::neg(var_4);
    var_14 = wp::div(var_13, var_11);
    var_15 = wp::div(var_2, var_11);
    var_16 = wp::vec_t<3, wp::float32>(var_12, var_14, var_15);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_16 += adj_ret;
    wp::adj_vec_t(var_12, var_14, var_15, adj_12, adj_14, adj_15, adj_16);
    wp::adj_div(var_2, var_11, var_15, adj_2, adj_11, adj_15);
    wp::adj_div(var_13, var_11, var_14, adj_13, adj_11, adj_14);
    wp::adj_neg(var_4, adj_4, adj_13);
    wp::adj_div(var_6, var_11, var_12, adj_6, adj_11, adj_12);
    // adj: return wp.vec3(c / determinant, -b / determinant, a / determinant)                <L 385>
    wp::adj_max(var_9, var_10, adj_9, adj_10, adj_11);
    wp::adj_sub(var_7, var_8, adj_7, adj_8, adj_9);
    wp::adj_mul(var_4, var_4, adj_4, adj_4, adj_8);
    wp::adj_mul(var_2, var_6, adj_2, adj_6, adj_7);
    // adj: determinant = wp.max(a * c - b * b, 1.0e-8)                                       <L 384>
    wp::adj_extract(var_0, var_5, adj_0, adj_5, adj_6);
    wp::adj_extract(var_0, var_3, adj_0, adj_3, adj_4);
    wp::adj_extract(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: a, b, c = covariance[0], covariance[1], covariance[2]                             <L 383>
    adj_projected_covariance_0(var_log_scale, var_quaternion, var_camera, var_x, var_y, var_z, var_focal, adj_log_scale, adj_quaternion, adj_camera, adj_x, adj_y, adj_z, adj_focal, adj_0);
    // adj: covariance = projected_covariance(log_scale, quaternion, camera, x, y, z, focal)  <L 382>
    // adj: def conic_at_centre(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44, x: float, y: float, z: float, focal: float):  <L 381>
    return;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:388
static CUDA_CALLABLE void adj_alpha_at_pixel_0(
    wp::vec_t<3, wp::float32> var_mean,
    wp::vec_t<3, wp::float32> var_log_scale,
    wp::vec_t<4, wp::float32> var_quaternion,
    wp::float32 var_opacity_logit,
    wp::mat_t<4, 4, wp::float32> var_camera,
    wp::float32 var_px,
    wp::float32 var_py,
    wp::float32 var_width,
    wp::float32 var_height,
    wp::float32 var_focal,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::vec_t<3, wp::float32> & adj_mean,
    wp::vec_t<3, wp::float32> & adj_log_scale,
    wp::vec_t<4, wp::float32> & adj_quaternion,
    wp::float32 & adj_opacity_logit,
    wp::mat_t<4, 4, wp::float32> & adj_camera,
    wp::float32 & adj_px,
    wp::float32 & adj_py,
    wp::float32 & adj_width,
    wp::float32 & adj_height,
    wp::float32 & adj_focal,
    wp::int32 & adj_compact_enabled,
    wp::float32 & adj_compact_beta,
    wp::float32 & adj_compact_alpha_min,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 1.0;
    wp::vec_t<4, wp::float32> var_7;
    wp::vec_t<4, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.1;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.5;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 2.0;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 2;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 1.0;
    const wp::float32 var_51 = 1.0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 9.0;
    wp::float32 var_57;
    const wp::int32 var_58 = 0;
    bool var_59;
    const wp::float32 var_60 = 0.0;
    bool var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = 2.0;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    wp::float32 var_72;
    bool var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    bool var_76;
    const wp::float32 var_77 = -0.5;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.99;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.00392156862745098;
    bool var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::vec_t<4, wp::float32> adj_7 = {};
    wp::vec_t<4, wp::float32> adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float32 adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::int32 adj_58 = {};
    bool adj_59 = {};
    wp::float32 adj_60 = {};
    bool adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    wp::float32 adj_72 = {};
    bool adj_73 = {};
    wp::float32 adj_74 = {};
    bool adj_75 = {};
    bool adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    bool adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float32 adj_87 = {};
    //---------
    // forward
    // def alpha_at_pixel(                                                                    <L 389>
    // point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)                               <L 413>
    var_1 = wp::extract(var_mean, var_0);
    var_3 = wp::extract(var_mean, var_2);
    var_5 = wp::extract(var_mean, var_4);
    var_7 = wp::vec_t<4, wp::float32>(var_1, var_3, var_5, var_6);
    var_8 = wp::mul(var_camera, var_7);
    // z = wp.max(point[2], NEAR_PLANE)                                                       <L 414>
    var_10 = wp::extract(var_8, var_9);
    var_12 = wp::max(var_10, var_11);
    // centre_x = focal * point[0] / z + 0.5 * width                                          <L 415>
    var_14 = wp::extract(var_8, var_13);
    var_15 = wp::mul(var_focal, var_14);
    var_16 = wp::div(var_15, var_12);
    var_18 = wp::mul(var_17, var_width);
    var_19 = wp::add(var_16, var_18);
    // centre_y = focal * point[1] / z + 0.5 * height                                         <L 416>
    var_21 = wp::extract(var_8, var_20);
    var_22 = wp::mul(var_focal, var_21);
    var_23 = wp::div(var_22, var_12);
    var_25 = wp::mul(var_24, var_height);
    var_26 = wp::add(var_23, var_25);
    // conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)       <L 417>
    var_28 = wp::extract(var_8, var_27);
    var_30 = wp::extract(var_8, var_29);
    var_31 = conic_at_centre_0(var_log_scale, var_quaternion, var_camera, var_28, var_30, var_12, var_focal);
    // dx, dy = px - centre_x, py - centre_y                                                  <L 418>
    var_32 = wp::sub(var_px, var_19);
    var_33 = wp::sub(var_py, var_26);
    // q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy                 <L 419>
    var_35 = wp::extract(var_31, var_34);
    var_36 = wp::mul(var_35, var_32);
    var_37 = wp::mul(var_36, var_32);
    var_40 = wp::extract(var_31, var_39);
    var_41 = wp::mul(var_38, var_40);
    var_42 = wp::mul(var_41, var_32);
    var_43 = wp::mul(var_42, var_33);
    var_44 = wp::add(var_37, var_43);
    var_46 = wp::extract(var_31, var_45);
    var_47 = wp::mul(var_46, var_33);
    var_48 = wp::mul(var_47, var_33);
    var_49 = wp::add(var_44, var_48);
    // opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))                                         <L 420>
    var_52 = wp::neg(var_opacity_logit);
    var_53 = wp::exp(var_52);
    var_54 = wp::add(var_51, var_53);
    var_55 = wp::div(var_50, var_54);
    // support = float(SUPPORT_RADIUS_SQUARED)                                                <L 421>
    var_57 = wp::float(var_56);
    // if compact_enabled != 0:                                                               <L 422>
    var_59 = (var_compact_enabled != var_58);
    if (var_59) {
        // support = 0.0                                                                      <L 423>
        // if opacity > compact_alpha_min:                                                    <L 424>
        var_61 = (var_55 > var_compact_alpha_min);
        if (var_61) {
            // support = wp.min(                                                              <L 425>
            // float(SUPPORT_RADIUS_SQUARED),                                                 <L 426>
            var_62 = wp::float(var_56);
            // compact_beta * 2.0 * wp.log(opacity / compact_alpha_min),                      <L 427>
            var_64 = wp::mul(var_compact_beta, var_63);
            var_65 = wp::div(var_55, var_compact_alpha_min);
            var_66 = wp::log(var_65);
            var_67 = wp::mul(var_64, var_66);
            var_68 = wp::min(var_62, var_67);
        }
        var_69 = wp::where(var_61, var_68, var_60);
    }
    var_70 = wp::where(var_59, var_69, var_57);
    // alpha = float(0.0)                                                                     <L 429>
    var_72 = wp::float(var_71);
    // if support > 0.0 and q <= support:                                                     <L 430>
    var_75 = (var_70 > var_74);
    var_73 = var_75;
    if (var_73) {
        var_76 = (var_49 <= var_70);
        var_73 = var_73 && var_76;
    }
    if (var_73) {
        // candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)                               <L 431>
        var_78 = wp::mul(var_77, var_49);
        var_79 = wp::exp(var_78);
        var_80 = wp::mul(var_55, var_79);
        var_82 = wp::min(var_80, var_81);
        // if candidate >= ALPHA_CUTOFF:                                                      <L 432>
        var_84 = (var_82 >= var_83);
        if (var_84) {
            // alpha = candidate                                                              <L 433>
            var_85 = wp::copy(var_82);
        }
        var_86 = wp::where(var_84, var_85, var_72);
    }
    var_87 = wp::where(var_73, var_86, var_72);
    // return alpha                                                                           <L 434>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_87 += adj_ret;
    // adj: return alpha                                                                      <L 434>
    wp::adj_where(var_73, var_86, var_72, adj_73, adj_86, adj_72, adj_87);
    if (var_73) {
        wp::adj_where(var_84, var_85, var_72, adj_84, adj_85, adj_72, adj_86);
        if (var_84) {
            wp::adj_copy(var_82, adj_82, adj_85);
            // adj: alpha = candidate                                                         <L 433>
        }
        // adj: if candidate >= ALPHA_CUTOFF:                                                 <L 432>
        wp::adj_min(var_80, var_81, adj_80, adj_81, adj_82);
        wp::adj_mul(var_55, var_79, adj_55, adj_79, adj_80);
        wp::adj_exp(var_78, var_79, adj_78, adj_79);
        wp::adj_mul(var_77, var_49, adj_77, adj_49, adj_78);
        // adj: candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)                          <L 431>
    }
    if (var_73) {
    }
    // adj: if support > 0.0 and q <= support:                                                <L 430>
    wp::adj_float(var_71, adj_71, adj_72);
    // adj: alpha = float(0.0)                                                                <L 429>
    wp::adj_where(var_59, var_69, var_57, adj_59, adj_69, adj_57, adj_70);
    if (var_59) {
        wp::adj_where(var_61, var_68, var_60, adj_61, adj_68, adj_60, adj_69);
        if (var_61) {
            wp::adj_min(var_62, var_67, adj_62, adj_67, adj_68);
            wp::adj_mul(var_64, var_66, adj_64, adj_66, adj_67);
            wp::adj_log(var_65, adj_65, adj_66);
            wp::adj_div(var_55, var_compact_alpha_min, var_65, adj_55, adj_compact_alpha_min, adj_65);
            wp::adj_mul(var_compact_beta, var_63, adj_compact_beta, adj_63, adj_64);
            // adj: compact_beta * 2.0 * wp.log(opacity / compact_alpha_min),                 <L 427>
            wp::adj_float(var_56, adj_56, adj_62);
            // adj: float(SUPPORT_RADIUS_SQUARED),                                            <L 426>
            // adj: support = wp.min(                                                         <L 425>
        }
        // adj: if opacity > compact_alpha_min:                                               <L 424>
        // adj: support = 0.0                                                                 <L 423>
    }
    // adj: if compact_enabled != 0:                                                          <L 422>
    wp::adj_float(var_56, adj_56, adj_57);
    // adj: support = float(SUPPORT_RADIUS_SQUARED)                                           <L 421>
    wp::adj_div(var_50, var_54, var_55, adj_50, adj_54, adj_55);
    wp::adj_add(var_51, var_53, adj_51, adj_53, adj_54);
    wp::adj_exp(var_52, var_53, adj_52, adj_53);
    wp::adj_neg(var_opacity_logit, adj_opacity_logit, adj_52);
    // adj: opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))                                    <L 420>
    wp::adj_add(var_44, var_48, adj_44, adj_48, adj_49);
    wp::adj_mul(var_47, var_33, adj_47, adj_33, adj_48);
    wp::adj_mul(var_46, var_33, adj_46, adj_33, adj_47);
    wp::adj_extract(var_31, var_45, adj_31, adj_45, adj_46);
    wp::adj_add(var_37, var_43, adj_37, adj_43, adj_44);
    wp::adj_mul(var_42, var_33, adj_42, adj_33, adj_43);
    wp::adj_mul(var_41, var_32, adj_41, adj_32, adj_42);
    wp::adj_mul(var_38, var_40, adj_38, adj_40, adj_41);
    wp::adj_extract(var_31, var_39, adj_31, adj_39, adj_40);
    wp::adj_mul(var_36, var_32, adj_36, adj_32, adj_37);
    wp::adj_mul(var_35, var_32, adj_35, adj_32, adj_36);
    wp::adj_extract(var_31, var_34, adj_31, adj_34, adj_35);
    // adj: q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy            <L 419>
    wp::adj_sub(var_py, var_26, adj_py, adj_26, adj_33);
    wp::adj_sub(var_px, var_19, adj_px, adj_19, adj_32);
    // adj: dx, dy = px - centre_x, py - centre_y                                             <L 418>
    adj_conic_at_centre_0(var_log_scale, var_quaternion, var_camera, var_28, var_30, var_12, var_focal, adj_log_scale, adj_quaternion, adj_camera, adj_28, adj_30, adj_12, adj_focal, adj_31);
    wp::adj_extract(var_8, var_29, adj_8, adj_29, adj_30);
    wp::adj_extract(var_8, var_27, adj_8, adj_27, adj_28);
    // adj: conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)  <L 417>
    wp::adj_add(var_23, var_25, adj_23, adj_25, adj_26);
    wp::adj_mul(var_24, var_height, adj_24, adj_height, adj_25);
    wp::adj_div(var_22, var_12, var_23, adj_22, adj_12, adj_23);
    wp::adj_mul(var_focal, var_21, adj_focal, adj_21, adj_22);
    wp::adj_extract(var_8, var_20, adj_8, adj_20, adj_21);
    // adj: centre_y = focal * point[1] / z + 0.5 * height                                    <L 416>
    wp::adj_add(var_16, var_18, adj_16, adj_18, adj_19);
    wp::adj_mul(var_17, var_width, adj_17, adj_width, adj_18);
    wp::adj_div(var_15, var_12, var_16, adj_15, adj_12, adj_16);
    wp::adj_mul(var_focal, var_14, adj_focal, adj_14, adj_15);
    wp::adj_extract(var_8, var_13, adj_8, adj_13, adj_14);
    // adj: centre_x = focal * point[0] / z + 0.5 * width                                     <L 415>
    wp::adj_max(var_10, var_11, adj_10, adj_11, adj_12);
    wp::adj_extract(var_8, var_9, adj_8, adj_9, adj_10);
    // adj: z = wp.max(point[2], NEAR_PLANE)                                                  <L 414>
    wp::adj_mul(var_camera, var_7, adj_camera, adj_7, adj_8);
    wp::adj_vec_t(var_1, var_3, var_5, var_6, adj_1, adj_3, adj_5, adj_6, adj_7);
    wp::adj_extract(var_mean, var_4, adj_mean, adj_4, adj_5);
    wp::adj_extract(var_mean, var_2, adj_mean, adj_2, adj_3);
    wp::adj_extract(var_mean, var_0, adj_mean, adj_0, adj_1);
    // adj: point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)                          <L 413>
    // adj: def alpha_at_pixel(                                                               <L 389>
    return;
}


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:437
static CUDA_CALLABLE void adj_colour_at_view_0(
    wp::vec_t<3, wp::float32> var_color,
    wp::vec_t<3, wp::float32> & adj_color,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 1.0;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    //---------
    // forward
    // def colour_at_view(                                                                    <L 438>
    // return wp.vec3(                                                                        <L 441>
    // wp.clamp(color[0], 0.0, 1.0),                                                          <L 442>
    var_1 = wp::extract(var_color, var_0);
    var_4 = wp::clamp(var_1, var_2, var_3);
    // wp.clamp(color[1], 0.0, 1.0),                                                          <L 443>
    var_6 = wp::extract(var_color, var_5);
    var_9 = wp::clamp(var_6, var_7, var_8);
    // wp.clamp(color[2], 0.0, 1.0),                                                          <L 444>
    var_11 = wp::extract(var_color, var_10);
    var_14 = wp::clamp(var_11, var_12, var_13);
    var_15 = wp::vec_t<3, wp::float32>(var_4, var_9, var_14);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_15 += adj_ret;
    wp::adj_vec_t(var_4, var_9, var_14, adj_4, adj_9, adj_14, adj_15);
    wp::adj_clamp(var_11, var_12, var_13, adj_11, adj_12, adj_13, adj_14);
    wp::adj_extract(var_color, var_10, adj_color, adj_10, adj_11);
    // adj: wp.clamp(color[2], 0.0, 1.0),                                                     <L 444>
    wp::adj_clamp(var_6, var_7, var_8, adj_6, adj_7, adj_8, adj_9);
    wp::adj_extract(var_color, var_5, adj_color, adj_5, adj_6);
    // adj: wp.clamp(color[1], 0.0, 1.0),                                                     <L 443>
    wp::adj_clamp(var_1, var_2, var_3, adj_1, adj_2, adj_3, adj_4);
    wp::adj_extract(var_color, var_0, adj_color, adj_0, adj_1);
    // adj: wp.clamp(color[0], 0.0, 1.0),                                                     <L 442>
    // adj: return wp.vec3(                                                                   <L 441>
    // adj: def colour_at_view(                                                               <L 438>
    return;
}



extern "C" __global__ void render_sparse_forward_25c85e28_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::int32 var_view,
    wp::array_t<wp::vec_t<2, wp::int32>> var_sample_xy,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_samples_per_tile,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::float32 var_focal,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt,
    wp::vec_t<3, wp::float32> var_background,
    wp::int32 var_num_samples,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rendered_rgb,
    wp::array_t<wp::int32> var_last_contributor,
    wp::array_t<wp::float32> var_loss)
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
        wp::int32 var_1;
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.5;
        wp::float32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.5;
        wp::float32 var_14;
        wp::mat_t<4, 4, wp::float32>* var_15;
        wp::mat_t<4, 4, wp::float32> var_16;
        wp::mat_t<4, 4, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<3, wp::float32> var_21;
        const wp::float32 var_22 = 1.0;
        wp::float32 var_23;
        wp::int32* var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::uint64* var_36;
        wp::uint64 var_37;
        wp::uint64 var_38;
        wp::uint64 var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<4, wp::float32>* var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<4, wp::float32> var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        bool var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::float32 var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        const wp::float32 var_60 = 1.0;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::int32 var_63;
        const wp::float32 var_64 = 0.0001;
        bool var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32>* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        const wp::int32 var_77 = 0;
        wp::float32 var_78;
        const wp::int32 var_79 = 3;
        wp::int32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        //---------
        // forward
        // def render_sparse_forward(                                                             <L 778>
        // thread = wp.tid()                                                                      <L 803>
        var_0 = builtin_tid1d();
        // record = thread // samples_per_tile                                                    <L 804>
        var_1 = wp::floordiv(var_0, var_samples_per_tile);
        // xy = sample_xy[thread]                                                                 <L 805>
        var_2 = wp::address(var_sample_xy, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // px = float(xy[0]) + 0.5                                                                <L 806>
        var_6 = wp::extract(var_3, var_5);
        var_7 = wp::float(var_6);
        var_9 = wp::add(var_7, var_8);
        // py = float(xy[1]) + 0.5                                                                <L 807>
        var_11 = wp::extract(var_3, var_10);
        var_12 = wp::float(var_11);
        var_14 = wp::add(var_12, var_13);
        // camera = cameras[view]                                                                 <L 808>
        var_15 = wp::address(var_cameras, var_view);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rgb = wp.vec3(0.0, 0.0, 0.0)                                                           <L 809>
        var_21 = wp::vec_t<3, wp::float32>(var_18, var_19, var_20);
        // transmittance = float(1.0)                                                             <L 810>
        var_23 = wp::float(var_22);
        // last_entry = offsets[record] - 1                                                       <L 811>
        var_24 = wp::address(var_offsets, var_1);
        var_27 = wp::load(var_24);
        var_26 = wp::sub(var_27, var_25);
        // for entry in range(offsets[record], offsets[record + 1]):                              <L 812>
        var_28 = wp::address(var_offsets, var_1);
        var_30 = wp::add(var_1, var_29);
        var_31 = wp::address(var_offsets, var_30);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_31);
        var_32 = wp::range(var_33, var_34);
        start_for_0:;
            if (iter_cmp(var_32) == 0) goto end_for_0;
            var_35 = wp::iter_next(var_32);
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 813>
            var_36 = wp::address(var_pairs, var_35);
            var_37 = 4294967295ull;
            var_39 = wp::load(var_36);
            var_38 = wp::bit_and(var_39, var_37);
            var_40 = wp::int(var_38);
            // alpha = alpha_at_pixel(                                                            <L 814>
            // means[splat], log_scales[splat], quaternions[splat], opacity_logits[splat],        <L 815>
            var_41 = wp::address(var_means, var_40);
            var_42 = wp::address(var_log_scales, var_40);
            var_43 = wp::address(var_quaternions, var_40);
            var_44 = wp::address(var_opacity_logits, var_40);
            // camera, px, py, float(width), float(height), focal,                                <L 816>
            var_45 = wp::float(var_width);
            var_46 = wp::float(var_height);
            // compact_enabled, compact_beta, compact_alpha_min,                                  <L 817>
            var_48 = wp::load(var_41);
            var_49 = wp::load(var_42);
            var_50 = wp::load(var_43);
            var_51 = wp::load(var_44);
            var_47 = alpha_at_pixel_0(var_48, var_49, var_50, var_51, var_16, var_9, var_14, var_45, var_46, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 819>
            var_53 = (var_47 > var_52);
            if (var_53) {
                // colour = colour_at_view(color[splat])                                          <L 820>
                var_54 = wp::address(var_color, var_40);
                var_56 = wp::load(var_54);
                var_55 = colour_at_view_0(var_56);
                // rgb = rgb + transmittance * alpha * colour                                     <L 821>
                var_57 = wp::mul(var_23, var_47);
                var_58 = wp::mul(var_57, var_55);
                var_59 = wp::add(var_21, var_58);
                // transmittance = transmittance * (1.0 - alpha)                                  <L 822>
                var_61 = wp::sub(var_60, var_47);
                var_62 = wp::mul(var_23, var_61);
                // last_entry = entry                                                             <L 823>
                var_63 = wp::copy(var_35);
                // if transmittance < TRANSMITTANCE_CUTOFF:                                       <L 824>
                var_65 = (var_62 < var_64);
                if (var_65) {
                    // break                                                                      <L 825>
                    wp::assign(var_21, var_59);
                    wp::assign(var_23, var_62);
                    wp::assign(var_26, var_63);
                    goto end_for_0;
                }
                var_66 = wp::where(var_65, var_21, var_59);
                var_67 = wp::where(var_65, var_23, var_62);
                var_68 = wp::where(var_65, var_26, var_63);
            }
            var_69 = wp::where(var_53, var_66, var_21);
            var_70 = wp::where(var_53, var_67, var_23);
            var_71 = wp::where(var_53, var_68, var_26);
            wp::assign(var_21, var_69);
            wp::assign(var_23, var_70);
            wp::assign(var_26, var_71);
            goto start_for_0;
        end_for_0:;
        // rgb = rgb + transmittance * background                                                 <L 826>
        var_72 = wp::mul(var_23, var_background);
        var_73 = wp::add(var_21, var_72);
        // rendered_rgb[thread] = rgb                                                             <L 827>
        wp::array_store(var_rendered_rgb, var_0, var_73);
        // last_contributor[thread] = last_entry                                                  <L 828>
        wp::array_store(var_last_contributor, var_0, var_26);
        // difference = rgb - sample_gt[thread]                                                   <L 829>
        var_74 = wp::address(var_sample_gt, var_0);
        var_76 = wp::load(var_74);
        var_75 = wp::sub(var_73, var_76);
        // wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * num_samples))        <L 830>
        var_78 = wp::dot(var_75, var_75);
        var_80 = wp::mul(var_79, var_num_samples);
        var_81 = wp::float(var_80);
        var_82 = wp::div(var_78, var_81);
        var_83 = wp::atomic_add(var_loss, var_77, var_82);
    }
}



extern "C" __global__ void render_sparse_forward_25c85e28_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::int32 var_view,
    wp::array_t<wp::vec_t<2, wp::int32>> var_sample_xy,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_samples_per_tile,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::float32 var_focal,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt,
    wp::vec_t<3, wp::float32> var_background,
    wp::int32 var_num_samples,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rendered_rgb,
    wp::array_t<wp::int32> var_last_contributor,
    wp::array_t<wp::float32> var_loss,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_means,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_quaternions,
    wp::array_t<wp::float32> adj_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> adj_cameras,
    wp::int32 adj_view,
    wp::array_t<wp::vec_t<2, wp::int32>> adj_sample_xy,
    wp::array_t<wp::int32> adj_offsets,
    wp::array_t<wp::uint64> adj_pairs,
    wp::int32 adj_samples_per_tile,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::float32 adj_focal,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sample_gt,
    wp::vec_t<3, wp::float32> adj_background,
    wp::int32 adj_num_samples,
    wp::int32 adj_compact_enabled,
    wp::float32 adj_compact_beta,
    wp::float32 adj_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_rendered_rgb,
    wp::array_t<wp::int32> adj_last_contributor,
    wp::array_t<wp::float32> adj_loss)
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
        wp::int32 var_1;
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.5;
        wp::float32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.5;
        wp::float32 var_14;
        wp::mat_t<4, 4, wp::float32>* var_15;
        wp::mat_t<4, 4, wp::float32> var_16;
        wp::mat_t<4, 4, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<3, wp::float32> var_21;
        const wp::float32 var_22 = 1.0;
        wp::float32 var_23;
        wp::int32* var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::uint64* var_36;
        wp::uint64 var_37;
        wp::uint64 var_38;
        wp::uint64 var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<4, wp::float32>* var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<4, wp::float32> var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        bool var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::float32 var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        const wp::float32 var_60 = 1.0;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::int32 var_63;
        const wp::float32 var_64 = 0.0001;
        bool var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::float32 var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32>* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        const wp::int32 var_77 = 0;
        wp::float32 var_78;
        const wp::int32 var_79 = 3;
        wp::int32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<2, wp::int32> adj_2 = {};
        wp::vec_t<2, wp::int32> adj_3 = {};
        wp::vec_t<2, wp::int32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::mat_t<4, 4, wp::float32> adj_15 = {};
        wp::mat_t<4, 4, wp::float32> adj_16 = {};
        wp::mat_t<4, 4, wp::float32> adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::range_t adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::uint64 adj_36 = {};
        wp::uint64 adj_37 = {};
        wp::uint64 adj_38 = {};
        wp::uint64 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::vec_t<4, wp::float32> adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::vec_t<4, wp::float32> adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        bool adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::vec_t<3, wp::float32> adj_56 = {};
        wp::float32 adj_57 = {};
        wp::vec_t<3, wp::float32> adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::float32 adj_64 = {};
        bool adj_65 = {};
        wp::vec_t<3, wp::float32> adj_66 = {};
        wp::float32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::vec_t<3, wp::float32> adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::vec_t<3, wp::float32> adj_73 = {};
        wp::vec_t<3, wp::float32> adj_74 = {};
        wp::vec_t<3, wp::float32> adj_75 = {};
        wp::vec_t<3, wp::float32> adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        //---------
        // forward
        // def render_sparse_forward(                                                             <L 778>
        // thread = wp.tid()                                                                      <L 803>
        var_0 = builtin_tid1d();
        // record = thread // samples_per_tile                                                    <L 804>
        var_1 = wp::floordiv(var_0, var_samples_per_tile);
        // xy = sample_xy[thread]                                                                 <L 805>
        var_2 = wp::address(var_sample_xy, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // px = float(xy[0]) + 0.5                                                                <L 806>
        var_6 = wp::extract(var_3, var_5);
        var_7 = wp::float(var_6);
        var_9 = wp::add(var_7, var_8);
        // py = float(xy[1]) + 0.5                                                                <L 807>
        var_11 = wp::extract(var_3, var_10);
        var_12 = wp::float(var_11);
        var_14 = wp::add(var_12, var_13);
        // camera = cameras[view]                                                                 <L 808>
        var_15 = wp::address(var_cameras, var_view);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rgb = wp.vec3(0.0, 0.0, 0.0)                                                           <L 809>
        var_21 = wp::vec_t<3, wp::float32>(var_18, var_19, var_20);
        // transmittance = float(1.0)                                                             <L 810>
        var_23 = wp::float(var_22);
        // last_entry = offsets[record] - 1                                                       <L 811>
        var_24 = wp::address(var_offsets, var_1);
        var_27 = wp::load(var_24);
        var_26 = wp::sub(var_27, var_25);
        // for entry in range(offsets[record], offsets[record + 1]):                              <L 812>
        var_28 = wp::address(var_offsets, var_1);
        var_30 = wp::add(var_1, var_29);
        var_31 = wp::address(var_offsets, var_30);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_31);
        var_32 = wp::range(var_33, var_34);
        // rgb = rgb + transmittance * background                                                 <L 826>
        var_72 = wp::mul(var_23, var_background);
        var_73 = wp::add(var_21, var_72);
        // rendered_rgb[thread] = rgb                                                             <L 827>
        // wp::array_store(var_rendered_rgb, var_0, var_73);
        // last_contributor[thread] = last_entry                                                  <L 828>
        // wp::array_store(var_last_contributor, var_0, var_26);
        // difference = rgb - sample_gt[thread]                                                   <L 829>
        var_74 = wp::address(var_sample_gt, var_0);
        var_76 = wp::load(var_74);
        var_75 = wp::sub(var_73, var_76);
        // wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * num_samples))        <L 830>
        var_78 = wp::dot(var_75, var_75);
        var_80 = wp::mul(var_79, var_num_samples);
        var_81 = wp::float(var_80);
        var_82 = wp::div(var_78, var_81);
        // var_83 = wp::atomic_add(var_loss, var_77, var_82);
        //---------
        // reverse
        wp::adj_atomic_add(var_loss, var_77, var_82, adj_loss, adj_77, adj_82, adj_83);
        wp::adj_div(var_78, var_81, var_82, adj_78, adj_81, adj_82);
        wp::adj_float(var_80, adj_80, adj_81);
        wp::adj_mul(var_79, var_num_samples, adj_79, adj_num_samples, adj_80);
        wp::adj_dot(var_75, var_75, adj_75, adj_75, adj_78);
        // adj: wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * num_samples))   <L 830>
        wp::adj_sub(var_73, var_76, adj_73, adj_74, adj_75);
        wp::adj_address(var_sample_gt, var_0, adj_sample_gt, adj_0, adj_74);
        // adj: difference = rgb - sample_gt[thread]                                              <L 829>
        wp::adj_array_store(var_last_contributor, var_0, var_26, adj_last_contributor, adj_0, adj_26);
        // adj: last_contributor[thread] = last_entry                                             <L 828>
        wp::adj_array_store(var_rendered_rgb, var_0, var_73, adj_rendered_rgb, adj_0, adj_73);
        // adj: rendered_rgb[thread] = rgb                                                        <L 827>
        wp::adj_add(var_21, var_72, adj_21, adj_72, adj_73);
        wp::adj_mul(var_23, var_background, adj_23, adj_background, adj_72);
        // adj: rgb = rgb + transmittance * background                                            <L 826>
        var_32 = wp::iter_reverse(var_32);
        start_for_0:;
            if (iter_cmp(var_32) == 0) goto end_for_0;
            var_35 = wp::iter_next(var_32);
        	adj_36 = {};
        	adj_37 = {};
        	adj_38 = {};
        	adj_39 = {};
        	adj_40 = {};
        	adj_41 = {};
        	adj_42 = {};
        	adj_43 = {};
        	adj_44 = {};
        	adj_45 = {};
        	adj_46 = {};
        	adj_47 = {};
        	adj_48 = {};
        	adj_49 = {};
        	adj_50 = {};
        	adj_51 = {};
        	adj_52 = {};
        	adj_53 = {};
        	adj_54 = {};
        	adj_55 = {};
        	adj_56 = {};
        	adj_57 = {};
        	adj_58 = {};
        	adj_59 = {};
        	adj_60 = {};
        	adj_61 = {};
        	adj_62 = {};
        	adj_63 = {};
        	adj_64 = {};
        	adj_65 = {};
        	adj_66 = {};
        	adj_67 = {};
        	adj_68 = {};
        	adj_69 = {};
        	adj_70 = {};
        	adj_71 = {};
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 813>
            var_36 = wp::address(var_pairs, var_35);
            var_37 = 4294967295ull;
            var_39 = wp::load(var_36);
            var_38 = wp::bit_and(var_39, var_37);
            var_40 = wp::int(var_38);
            // alpha = alpha_at_pixel(                                                            <L 814>
            // means[splat], log_scales[splat], quaternions[splat], opacity_logits[splat],        <L 815>
            var_41 = wp::address(var_means, var_40);
            var_42 = wp::address(var_log_scales, var_40);
            var_43 = wp::address(var_quaternions, var_40);
            var_44 = wp::address(var_opacity_logits, var_40);
            // camera, px, py, float(width), float(height), focal,                                <L 816>
            var_45 = wp::float(var_width);
            var_46 = wp::float(var_height);
            // compact_enabled, compact_beta, compact_alpha_min,                                  <L 817>
            var_48 = wp::load(var_41);
            var_49 = wp::load(var_42);
            var_50 = wp::load(var_43);
            var_51 = wp::load(var_44);
            var_47 = alpha_at_pixel_0(var_48, var_49, var_50, var_51, var_16, var_9, var_14, var_45, var_46, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 819>
            var_53 = (var_47 > var_52);
            if (var_53) {
                // colour = colour_at_view(color[splat])                                          <L 820>
                var_54 = wp::address(var_color, var_40);
                var_56 = wp::load(var_54);
                var_55 = colour_at_view_0(var_56);
                // rgb = rgb + transmittance * alpha * colour                                     <L 821>
                var_57 = wp::mul(var_23, var_47);
                var_58 = wp::mul(var_57, var_55);
                var_59 = wp::add(var_21, var_58);
                // transmittance = transmittance * (1.0 - alpha)                                  <L 822>
                var_61 = wp::sub(var_60, var_47);
                var_62 = wp::mul(var_23, var_61);
                // last_entry = entry                                                             <L 823>
                var_63 = wp::copy(var_35);
                // if transmittance < TRANSMITTANCE_CUTOFF:                                       <L 824>
                var_65 = (var_62 < var_64);
                if (var_65) {
                    // break                                                                      <L 825>
                    wp::assign(var_21, var_59);
                    wp::assign(var_23, var_62);
                    wp::assign(var_26, var_63);
                    goto end_for_0;
                }
                var_66 = wp::where(var_65, var_21, var_59);
                var_67 = wp::where(var_65, var_23, var_62);
                var_68 = wp::where(var_65, var_26, var_63);
            }
            var_69 = wp::where(var_53, var_66, var_21);
            var_70 = wp::where(var_53, var_67, var_23);
            var_71 = wp::where(var_53, var_68, var_26);
            wp::assign(var_21, var_69);
            wp::assign(var_23, var_70);
            wp::assign(var_26, var_71);
            wp::adj_assign(var_26, var_71, adj_26, adj_71);
            wp::adj_assign(var_23, var_70, adj_23, adj_70);
            wp::adj_assign(var_21, var_69, adj_21, adj_69);
            wp::adj_where(var_53, var_68, var_26, adj_53, adj_68, adj_26, adj_71);
            wp::adj_where(var_53, var_67, var_23, adj_53, adj_67, adj_23, adj_70);
            wp::adj_where(var_53, var_66, var_21, adj_53, adj_66, adj_21, adj_69);
            if (var_53) {
                wp::adj_where(var_65, var_26, var_63, adj_65, adj_26, adj_63, adj_68);
                wp::adj_where(var_65, var_23, var_62, adj_65, adj_23, adj_62, adj_67);
                wp::adj_where(var_65, var_21, var_59, adj_65, adj_21, adj_59, adj_66);
                if (var_65) {
                    wp::adj_assign(var_26, var_63, adj_26, adj_63);
                    wp::adj_assign(var_23, var_62, adj_23, adj_62);
                    wp::adj_assign(var_21, var_59, adj_21, adj_59);
                    // adj: break                                                                 <L 825>
                }
                // adj: if transmittance < TRANSMITTANCE_CUTOFF:                                  <L 824>
                wp::adj_copy(var_35, adj_35, adj_63);
                // adj: last_entry = entry                                                        <L 823>
                wp::adj_mul(var_23, var_61, adj_23, adj_61, adj_62);
                wp::adj_sub(var_60, var_47, adj_60, adj_47, adj_61);
                // adj: transmittance = transmittance * (1.0 - alpha)                             <L 822>
                wp::adj_add(var_21, var_58, adj_21, adj_58, adj_59);
                wp::adj_mul(var_57, var_55, adj_57, adj_55, adj_58);
                wp::adj_mul(var_23, var_47, adj_23, adj_47, adj_57);
                // adj: rgb = rgb + transmittance * alpha * colour                                <L 821>
                adj_colour_at_view_0(var_56, adj_54, adj_55);
                wp::adj_address(var_color, var_40, adj_color, adj_40, adj_54);
                // adj: colour = colour_at_view(color[splat])                                     <L 820>
            }
            // adj: if alpha > 0.0:                                                               <L 819>
            adj_alpha_at_pixel_0(var_48, var_49, var_50, var_51, var_16, var_9, var_14, var_45, var_46, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min, adj_41, adj_42, adj_43, adj_44, adj_16, adj_9, adj_14, adj_45, adj_46, adj_focal, adj_compact_enabled, adj_compact_beta, adj_compact_alpha_min, adj_47);
            // adj: compact_enabled, compact_beta, compact_alpha_min,                             <L 817>
            wp::adj_float(var_height, adj_height, adj_46);
            wp::adj_float(var_width, adj_width, adj_45);
            // adj: camera, px, py, float(width), float(height), focal,                           <L 816>
            wp::adj_address(var_opacity_logits, var_40, adj_opacity_logits, adj_40, adj_44);
            wp::adj_address(var_quaternions, var_40, adj_quaternions, adj_40, adj_43);
            wp::adj_address(var_log_scales, var_40, adj_log_scales, adj_40, adj_42);
            wp::adj_address(var_means, var_40, adj_means, adj_40, adj_41);
            // adj: means[splat], log_scales[splat], quaternions[splat], opacity_logits[splat],   <L 815>
            // adj: alpha = alpha_at_pixel(                                                       <L 814>
            wp::adj_address(var_pairs, var_35, adj_pairs, adj_35, adj_36);
            // adj: splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                             <L 813>
        	goto start_for_0;
        end_for_0:;
        wp::adj_address(var_offsets, var_30, adj_offsets, adj_30, adj_31);
        wp::adj_add(var_1, var_29, adj_1, adj_29, adj_30);
        wp::adj_address(var_offsets, var_1, adj_offsets, adj_1, adj_28);
        // adj: for entry in range(offsets[record], offsets[record + 1]):                         <L 812>
        wp::adj_sub(var_27, var_25, adj_24, adj_25, adj_26);
        wp::adj_address(var_offsets, var_1, adj_offsets, adj_1, adj_24);
        // adj: last_entry = offsets[record] - 1                                                  <L 811>
        wp::adj_float(var_22, adj_22, adj_23);
        // adj: transmittance = float(1.0)                                                        <L 810>
        wp::adj_vec_t(var_18, var_19, var_20, adj_18, adj_19, adj_20, adj_21);
        // adj: rgb = wp.vec3(0.0, 0.0, 0.0)                                                      <L 809>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_cameras, var_view, adj_cameras, adj_view, adj_15);
        // adj: camera = cameras[view]                                                            <L 808>
        wp::adj_add(var_12, var_13, adj_12, adj_13, adj_14);
        wp::adj_float(var_11, adj_11, adj_12);
        wp::adj_extract(var_3, var_10, adj_3, adj_10, adj_11);
        // adj: py = float(xy[1]) + 0.5                                                           <L 807>
        wp::adj_add(var_7, var_8, adj_7, adj_8, adj_9);
        wp::adj_float(var_6, adj_6, adj_7);
        wp::adj_extract(var_3, var_5, adj_3, adj_5, adj_6);
        // adj: px = float(xy[0]) + 0.5                                                           <L 806>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_sample_xy, var_0, adj_sample_xy, adj_0, adj_2);
        // adj: xy = sample_xy[thread]                                                            <L 805>
        // adj: record = thread // samples_per_tile                                               <L 804>
        // adj: thread = wp.tid()                                                                 <L 803>
        // adj: def render_sparse_forward(                                                        <L 778>
        continue;
    }
}



extern "C" __global__ void render_backward_2c27b4c8_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::array_t<wp::int32> var_view_ids,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_tiles_x,
    wp::int32 var_tiles_y,
    wp::int32 var_tile,
    wp::float32 var_focal,
    wp::array_t<wp::vec_t<3, wp::float32>> var_image,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::float32> var_mean_grad_flat,
    wp::array_t<wp::float32> var_scale_grad_flat,
    wp::array_t<wp::float32> var_quaternion_grad_flat,
    wp::array_t<wp::float32> var_opacity_grad,
    wp::array_t<wp::float32> var_color_grad_flat)
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
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.5;
        wp::float32 var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.5;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.0;
        wp::vec_t<3, wp::float32> var_34;
        const wp::float32 var_35 = 1.0;
        wp::float32 var_36;
        wp::int32* var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::range_t var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::uint64* var_45;
        wp::uint64 var_46;
        wp::uint64 var_47;
        wp::uint64 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<4, wp::float32>* var_56;
        wp::vec_t<4, wp::float32> var_57;
        wp::vec_t<4, wp::float32> var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::mat_t<4, 4, wp::float32>* var_62;
        wp::mat_t<4, 4, wp::float32> var_63;
        wp::mat_t<4, 4, wp::float32> var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 0.0;
        bool var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        const wp::float32 var_73 = 1.0;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::float32 var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        const wp::float32 var_80 = 1e-08;
        wp::float32 var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::float32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32>* var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        bool var_88;
        const wp::int32 var_89 = 0;
        wp::float32 var_90;
        const wp::float32 var_91 = 0.0;
        bool var_92;
        const wp::int32 var_93 = 0;
        wp::float32 var_94;
        const wp::float32 var_95 = 1.0;
        bool var_96;
        const wp::float32 var_97 = 0.0;
        const wp::int32 var_98 = 0;
        bool var_99;
        const wp::int32 var_100 = 1;
        wp::float32 var_101;
        const wp::float32 var_102 = 0.0;
        bool var_103;
        const wp::int32 var_104 = 1;
        wp::float32 var_105;
        const wp::float32 var_106 = 1.0;
        bool var_107;
        const wp::float32 var_108 = 0.0;
        const wp::int32 var_109 = 1;
        bool var_110;
        const wp::int32 var_111 = 2;
        wp::float32 var_112;
        const wp::float32 var_113 = 0.0;
        bool var_114;
        const wp::int32 var_115 = 2;
        wp::float32 var_116;
        const wp::float32 var_117 = 1.0;
        bool var_118;
        const wp::float32 var_119 = 0.0;
        const wp::int32 var_120 = 2;
        wp::vec_t<3, wp::float32> var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<4, wp::float32> var_128;
        wp::float32 var_129;
        wp::mat_t<4, 4, wp::float32> var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::int32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::tuple_t<wp::vec_t<3, wp::float32>, wp::vec_t<3, wp::float32>, wp::vec_t<4, wp::float32>, float, wp::mat_t<4, 4, wp::float32>, float, float, float, float, float, int, float, float> var_140;
        const wp::int32 var_141 = 0;
        wp::vec_t<3, wp::float32> var_142;
        const wp::int32 var_143 = 1;
        wp::vec_t<3, wp::float32> var_144;
        const wp::int32 var_145 = 2;
        wp::vec_t<4, wp::float32> var_146;
        const wp::int32 var_147 = 3;
        wp::float32 var_148;
        const wp::int32 var_149 = 4;
        wp::mat_t<4, 4, wp::float32> var_150;
        const wp::int32 var_151 = 5;
        wp::float32 var_152;
        const wp::int32 var_153 = 6;
        wp::float32 var_154;
        const wp::int32 var_155 = 7;
        wp::float32 var_156;
        const wp::int32 var_157 = 8;
        wp::float32 var_158;
        const wp::int32 var_159 = 9;
        wp::float32 var_160;
        const wp::int32 var_161 = 10;
        wp::int32 var_162;
        const wp::int32 var_163 = 11;
        wp::float32 var_164;
        const wp::int32 var_165 = 12;
        wp::float32 var_166;
        const wp::int32 var_167 = 3;
        wp::int32 var_168;
        const wp::int32 var_169 = 0;
        wp::int32 var_170;
        const wp::int32 var_171 = 0;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        const wp::int32 var_175 = 3;
        wp::int32 var_176;
        const wp::int32 var_177 = 1;
        wp::int32 var_178;
        const wp::int32 var_179 = 1;
        wp::float32 var_180;
        wp::float32 var_181;
        wp::float32 var_182;
        const wp::int32 var_183 = 3;
        wp::int32 var_184;
        const wp::int32 var_185 = 2;
        wp::int32 var_186;
        const wp::int32 var_187 = 2;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        const wp::int32 var_191 = 3;
        wp::int32 var_192;
        const wp::int32 var_193 = 0;
        wp::int32 var_194;
        const wp::int32 var_195 = 0;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        const wp::int32 var_199 = 3;
        wp::int32 var_200;
        const wp::int32 var_201 = 1;
        wp::int32 var_202;
        const wp::int32 var_203 = 1;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        const wp::int32 var_207 = 3;
        wp::int32 var_208;
        const wp::int32 var_209 = 2;
        wp::int32 var_210;
        const wp::int32 var_211 = 2;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::int32 var_215 = 4;
        wp::int32 var_216;
        const wp::int32 var_217 = 0;
        wp::int32 var_218;
        const wp::int32 var_219 = 0;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        const wp::int32 var_223 = 4;
        wp::int32 var_224;
        const wp::int32 var_225 = 1;
        wp::int32 var_226;
        const wp::int32 var_227 = 1;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        const wp::int32 var_231 = 4;
        wp::int32 var_232;
        const wp::int32 var_233 = 2;
        wp::int32 var_234;
        const wp::int32 var_235 = 2;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        const wp::int32 var_239 = 4;
        wp::int32 var_240;
        const wp::int32 var_241 = 3;
        wp::int32 var_242;
        const wp::int32 var_243 = 3;
        wp::float32 var_244;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        const wp::int32 var_249 = 3;
        wp::int32 var_250;
        const wp::int32 var_251 = 0;
        wp::int32 var_252;
        const wp::int32 var_253 = 0;
        wp::float32 var_254;
        wp::float32 var_255;
        const wp::int32 var_256 = 3;
        wp::int32 var_257;
        const wp::int32 var_258 = 1;
        wp::int32 var_259;
        const wp::int32 var_260 = 1;
        wp::float32 var_261;
        wp::float32 var_262;
        const wp::int32 var_263 = 3;
        wp::int32 var_264;
        const wp::int32 var_265 = 2;
        wp::int32 var_266;
        const wp::int32 var_267 = 2;
        wp::float32 var_268;
        wp::float32 var_269;
        wp::float32 var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::vec_t<3, wp::float32> var_272;
        wp::float32 var_273;
        const wp::float32 var_274 = 0.0001;
        bool var_275;
        wp::vec_t<3, wp::float32> var_276;
        wp::float32 var_277;
        wp::vec_t<3, wp::float32> var_278;
        wp::float32 var_279;
        //---------
        // forward
        // def render_backward(                                                                   <L 504>
        // thread = wp.tid()                                                                      <L 531>
        var_0 = builtin_tid1d();
        // pixels = width * height                                                                <L 532>
        var_1 = wp::mul(var_width, var_height);
        // batch_view = thread // pixels                                                          <L 533>
        var_2 = wp::floordiv(var_0, var_1);
        // view = view_ids[batch_view]                                                            <L 534>
        var_3 = wp::address(var_view_ids, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // pixel = thread - batch_view * pixels                                                   <L 535>
        var_6 = wp::mul(var_2, var_1);
        var_7 = wp::sub(var_0, var_6);
        // px = float(pixel % width) + 0.5                                                        <L 536>
        var_8 = wp::mod(var_7, var_width);
        var_9 = wp::float(var_8);
        var_11 = wp::add(var_9, var_10);
        // py = float(pixel // width) + 0.5                                                       <L 537>
        var_12 = wp::floordiv(var_7, var_width);
        var_13 = wp::float(var_12);
        var_15 = wp::add(var_13, var_14);
        // record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)       <L 538>
        var_16 = wp::mul(var_2, var_tiles_x);
        var_17 = wp::mul(var_16, var_tiles_y);
        var_18 = wp::floordiv(var_7, var_width);
        var_19 = wp::floordiv(var_18, var_tile);
        var_20 = wp::mul(var_19, var_tiles_x);
        var_21 = wp::add(var_17, var_20);
        var_22 = wp::mod(var_7, var_width);
        var_23 = wp::floordiv(var_22, var_tile);
        var_24 = wp::add(var_21, var_23);
        // final_rgb = image[thread]                                                              <L 539>
        var_25 = wp::address(var_image, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // pixel_grad = pixel_gradient[thread]                                                    <L 540>
        var_28 = wp::address(var_pixel_gradient, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // prefix_rgb = wp.vec3(0.0, 0.0, 0.0)                                                    <L 541>
        var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
        // transmittance = float(1.0)                                                             <L 542>
        var_36 = wp::float(var_35);
        // for entry in range(offsets[record], offsets[record + 1]):                              <L 546>
        var_37 = wp::address(var_offsets, var_24);
        var_39 = wp::add(var_24, var_38);
        var_40 = wp::address(var_offsets, var_39);
        var_42 = wp::load(var_37);
        var_43 = wp::load(var_40);
        var_41 = wp::range(var_42, var_43);
        start_for_0:;
            if (iter_cmp(var_41) == 0) goto end_for_0;
            var_44 = wp::iter_next(var_41);
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 547>
            var_45 = wp::address(var_pairs, var_44);
            var_46 = 4294967295ull;
            var_48 = wp::load(var_45);
            var_47 = wp::bit_and(var_48, var_46);
            var_49 = wp::int(var_47);
            // mean = means[splat]                                                                <L 548>
            var_50 = wp::address(var_means, var_49);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // log_scale = log_scales[splat]                                                      <L 549>
            var_53 = wp::address(var_log_scales, var_49);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // quaternion = quaternions[splat]                                                    <L 550>
            var_56 = wp::address(var_quaternions, var_49);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // opacity_logit = opacity_logits[splat]                                              <L 551>
            var_59 = wp::address(var_opacity_logits, var_49);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // camera = cameras[view]                                                             <L 552>
            var_62 = wp::address(var_cameras, var_4);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // alpha = alpha_at_pixel(mean, log_scale, quaternion, opacity_logit, camera,         <L 553>
            // px, py, float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)       <L 554>
            var_65 = wp::float(var_width);
            var_66 = wp::float(var_height);
            var_67 = alpha_at_pixel_0(var_51, var_54, var_57, var_60, var_63, var_11, var_15, var_65, var_66, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 555>
            var_69 = (var_67 > var_68);
            if (var_69) {
                // colour = colour_at_view(color[splat])                                          <L 556>
                var_70 = wp::address(var_color, var_49);
                var_72 = wp::load(var_70);
                var_71 = colour_at_view_0(var_72);
                // next_transmittance = transmittance * (1.0 - alpha)                             <L 557>
                var_74 = wp::sub(var_73, var_67);
                var_75 = wp::mul(var_36, var_74);
                // remaining_rgb = (                                                              <L 589>
                // final_rgb                                                                      <L 590>
                // - prefix_rgb                                                                   <L 591>
                var_76 = wp::sub(var_26, var_34);
                // - transmittance * alpha * colour                                               <L 592>
                var_77 = wp::mul(var_36, var_67);
                var_78 = wp::mul(var_77, var_71);
                var_79 = wp::sub(var_76, var_78);
                // ) / wp.max(next_transmittance, 1.0e-8)                                         <L 593>
                var_81 = wp::max(var_75, var_80);
                var_82 = wp::div(var_79, var_81);
                // colour_adjoint = transmittance * alpha * pixel_grad                            <L 595>
                var_83 = wp::mul(var_36, var_67);
                var_84 = wp::mul(var_83, var_29);
                // raw_colour = color[splat]                                                      <L 597>
                var_85 = wp::address(var_color, var_49);
                var_87 = wp::load(var_85);
                var_86 = wp::copy(var_87);
                // if raw_colour[0] <= 0.0 or raw_colour[0] >= 1.0:                               <L 599>
                var_90 = wp::extract(var_86, var_89);
                var_92 = (var_90 <= var_91);
                var_88 = var_92;
                if (!var_88) {
                    var_94 = wp::extract(var_86, var_93);
                    var_96 = (var_94 >= var_95);
                    var_88 = var_88 || var_96;
                }
                if (var_88) {
                    // colour_adjoint[0] = 0.0                                                    <L 600>
                    wp::assign_inplace(var_84, var_98, var_97);
                }
                // if raw_colour[1] <= 0.0 or raw_colour[1] >= 1.0:                               <L 601>
                var_101 = wp::extract(var_86, var_100);
                var_103 = (var_101 <= var_102);
                var_99 = var_103;
                if (!var_99) {
                    var_105 = wp::extract(var_86, var_104);
                    var_107 = (var_105 >= var_106);
                    var_99 = var_99 || var_107;
                }
                if (var_99) {
                    // colour_adjoint[1] = 0.0                                                    <L 602>
                    wp::assign_inplace(var_84, var_109, var_108);
                }
                // if raw_colour[2] <= 0.0 or raw_colour[2] >= 1.0:                               <L 603>
                var_112 = wp::extract(var_86, var_111);
                var_114 = (var_112 <= var_113);
                var_110 = var_114;
                if (!var_110) {
                    var_116 = wp::extract(var_86, var_115);
                    var_118 = (var_116 >= var_117);
                    var_110 = var_110 || var_118;
                }
                if (var_110) {
                    // colour_adjoint[2] = 0.0                                                    <L 604>
                    wp::assign_inplace(var_84, var_120, var_119);
                }
                // alpha_adjoint = transmittance * wp.dot(                                        <L 606>
                // pixel_grad,                                                                    <L 607>
                // colour - remaining_rgb,                                                        <L 608>
                var_121 = wp::sub(var_71, var_82);
                var_122 = wp::dot(var_29, var_121);
                var_123 = wp::mul(var_36, var_122);
                // (                                                                              <L 611>
                // ) = wp.grad(alpha_at_pixel)(                                                   <L 625>
                // mean,                                                                          <L 626>
                // log_scale,                                                                     <L 627>
                // quaternion,                                                                    <L 628>
                // opacity_logit,                                                                 <L 629>
                // camera,                                                                        <L 630>
                // px,                                                                            <L 631>
                // py,                                                                            <L 632>
                // float(width),                                                                  <L 633>
                var_124 = wp::float(var_width);
                // float(height),                                                                 <L 634>
                var_125 = wp::float(var_height);
                // focal,                                                                         <L 635>
                // compact_enabled,                                                               <L 636>
                // compact_beta,                                                                  <L 637>
                // compact_alpha_min,                                                             <L 638>
                var_126 = {};
                var_127 = {};
                var_128 = {};
                var_129 = {};
                var_130 = {};
                var_131 = {};
                var_132 = {};
                var_133 = {};
                var_134 = {};
                var_135 = {};
                var_136 = {};
                var_137 = {};
                var_138 = {};
                var_139 = wp::float32(1);
                adj_alpha_at_pixel_0(var_51, var_54, var_57, var_60, var_63, var_11, var_15, var_124, var_125, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min, var_126, var_127, var_128, var_129, var_130, var_131, var_132, var_133, var_134, var_135, var_136, var_137, var_138, var_139);
                var_140 = wp::tuple_t<wp::vec_t<3, wp::float32>, wp::vec_t<3, wp::float32>, wp::vec_t<4, wp::float32>, float, wp::mat_t<4, 4, wp::float32>, float, float, float, float, float, int, float, float>(var_126, var_127, var_128, var_129, var_130, var_131, var_132, var_133, var_134, var_135, var_136, var_137, var_138);
                var_142 = wp::extract<0>(var_140);
                var_144 = wp::extract<1>(var_140);
                var_146 = wp::extract<2>(var_140);
                var_148 = wp::extract<3>(var_140);
                var_150 = wp::extract<4>(var_140);
                var_152 = wp::extract<5>(var_140);
                var_154 = wp::extract<6>(var_140);
                var_156 = wp::extract<7>(var_140);
                var_158 = wp::extract<8>(var_140);
                var_160 = wp::extract<9>(var_140);
                var_162 = wp::extract<10>(var_140);
                var_164 = wp::extract<11>(var_140);
                var_166 = wp::extract<12>(var_140);
                // wp.atomic_add(                                                                 <L 641>
                // mean_grad_flat,                                                                <L 642>
                // splat * 3 + 0,                                                                 <L 643>
                var_168 = wp::mul(var_49, var_167);
                var_170 = wp::add(var_168, var_169);
                // alpha_adjoint * mean_alpha_grad[0],                                            <L 644>
                var_172 = wp::extract(var_142, var_171);
                var_173 = wp::mul(var_123, var_172);
                var_174 = wp::atomic_add(var_mean_grad_flat, var_170, var_173);
                // wp.atomic_add(                                                                 <L 646>
                // mean_grad_flat,                                                                <L 647>
                // splat * 3 + 1,                                                                 <L 648>
                var_176 = wp::mul(var_49, var_175);
                var_178 = wp::add(var_176, var_177);
                // alpha_adjoint * mean_alpha_grad[1],                                            <L 649>
                var_180 = wp::extract(var_142, var_179);
                var_181 = wp::mul(var_123, var_180);
                var_182 = wp::atomic_add(var_mean_grad_flat, var_178, var_181);
                // wp.atomic_add(                                                                 <L 651>
                // mean_grad_flat,                                                                <L 652>
                // splat * 3 + 2,                                                                 <L 653>
                var_184 = wp::mul(var_49, var_183);
                var_186 = wp::add(var_184, var_185);
                // alpha_adjoint * mean_alpha_grad[2],                                            <L 654>
                var_188 = wp::extract(var_142, var_187);
                var_189 = wp::mul(var_123, var_188);
                var_190 = wp::atomic_add(var_mean_grad_flat, var_186, var_189);
                // wp.atomic_add(                                                                 <L 657>
                // scale_grad_flat,                                                               <L 658>
                // splat * 3 + 0,                                                                 <L 659>
                var_192 = wp::mul(var_49, var_191);
                var_194 = wp::add(var_192, var_193);
                // alpha_adjoint * log_scale_alpha_grad[0],                                       <L 660>
                var_196 = wp::extract(var_144, var_195);
                var_197 = wp::mul(var_123, var_196);
                var_198 = wp::atomic_add(var_scale_grad_flat, var_194, var_197);
                // wp.atomic_add(                                                                 <L 662>
                // scale_grad_flat,                                                               <L 663>
                // splat * 3 + 1,                                                                 <L 664>
                var_200 = wp::mul(var_49, var_199);
                var_202 = wp::add(var_200, var_201);
                // alpha_adjoint * log_scale_alpha_grad[1],                                       <L 665>
                var_204 = wp::extract(var_144, var_203);
                var_205 = wp::mul(var_123, var_204);
                var_206 = wp::atomic_add(var_scale_grad_flat, var_202, var_205);
                // wp.atomic_add(                                                                 <L 667>
                // scale_grad_flat,                                                               <L 668>
                // splat * 3 + 2,                                                                 <L 669>
                var_208 = wp::mul(var_49, var_207);
                var_210 = wp::add(var_208, var_209);
                // alpha_adjoint * log_scale_alpha_grad[2],                                       <L 670>
                var_212 = wp::extract(var_144, var_211);
                var_213 = wp::mul(var_123, var_212);
                var_214 = wp::atomic_add(var_scale_grad_flat, var_210, var_213);
                // wp.atomic_add(                                                                 <L 673>
                // quaternion_grad_flat,                                                          <L 674>
                // splat * 4 + 0,                                                                 <L 675>
                var_216 = wp::mul(var_49, var_215);
                var_218 = wp::add(var_216, var_217);
                // alpha_adjoint * quaternion_alpha_grad[0],                                      <L 676>
                var_220 = wp::extract(var_146, var_219);
                var_221 = wp::mul(var_123, var_220);
                var_222 = wp::atomic_add(var_quaternion_grad_flat, var_218, var_221);
                // wp.atomic_add(                                                                 <L 678>
                // quaternion_grad_flat,                                                          <L 679>
                // splat * 4 + 1,                                                                 <L 680>
                var_224 = wp::mul(var_49, var_223);
                var_226 = wp::add(var_224, var_225);
                // alpha_adjoint * quaternion_alpha_grad[1],                                      <L 681>
                var_228 = wp::extract(var_146, var_227);
                var_229 = wp::mul(var_123, var_228);
                var_230 = wp::atomic_add(var_quaternion_grad_flat, var_226, var_229);
                // wp.atomic_add(                                                                 <L 683>
                // quaternion_grad_flat,                                                          <L 684>
                // splat * 4 + 2,                                                                 <L 685>
                var_232 = wp::mul(var_49, var_231);
                var_234 = wp::add(var_232, var_233);
                // alpha_adjoint * quaternion_alpha_grad[2],                                      <L 686>
                var_236 = wp::extract(var_146, var_235);
                var_237 = wp::mul(var_123, var_236);
                var_238 = wp::atomic_add(var_quaternion_grad_flat, var_234, var_237);
                // wp.atomic_add(                                                                 <L 688>
                // quaternion_grad_flat,                                                          <L 689>
                // splat * 4 + 3,                                                                 <L 690>
                var_240 = wp::mul(var_49, var_239);
                var_242 = wp::add(var_240, var_241);
                // alpha_adjoint * quaternion_alpha_grad[3],                                      <L 691>
                var_244 = wp::extract(var_146, var_243);
                var_245 = wp::mul(var_123, var_244);
                var_246 = wp::atomic_add(var_quaternion_grad_flat, var_242, var_245);
                // wp.atomic_add(                                                                 <L 694>
                // opacity_grad,                                                                  <L 695>
                // splat,                                                                         <L 696>
                // alpha_adjoint * opacity_alpha_grad,                                            <L 697>
                var_247 = wp::mul(var_123, var_148);
                var_248 = wp::atomic_add(var_opacity_grad, var_49, var_247);
                // wp.atomic_add(                                                                 <L 700>
                // color_grad_flat,                                                               <L 701>
                // splat * 3 + 0,                                                                 <L 702>
                var_250 = wp::mul(var_49, var_249);
                var_252 = wp::add(var_250, var_251);
                // colour_adjoint[0],                                                             <L 703>
                var_254 = wp::extract(var_84, var_253);
                var_255 = wp::atomic_add(var_color_grad_flat, var_252, var_254);
                // wp.atomic_add(                                                                 <L 705>
                // color_grad_flat,                                                               <L 706>
                // splat * 3 + 1,                                                                 <L 707>
                var_257 = wp::mul(var_49, var_256);
                var_259 = wp::add(var_257, var_258);
                // colour_adjoint[1],                                                             <L 708>
                var_261 = wp::extract(var_84, var_260);
                var_262 = wp::atomic_add(var_color_grad_flat, var_259, var_261);
                // wp.atomic_add(                                                                 <L 710>
                // color_grad_flat,                                                               <L 711>
                // splat * 3 + 2,                                                                 <L 712>
                var_264 = wp::mul(var_49, var_263);
                var_266 = wp::add(var_264, var_265);
                // colour_adjoint[2],                                                             <L 713>
                var_268 = wp::extract(var_84, var_267);
                var_269 = wp::atomic_add(var_color_grad_flat, var_266, var_268);
                // prefix_rgb = prefix_rgb + transmittance * alpha * colour                       <L 717>
                var_270 = wp::mul(var_36, var_67);
                var_271 = wp::mul(var_270, var_71);
                var_272 = wp::add(var_34, var_271);
                // transmittance = next_transmittance                                             <L 718>
                var_273 = wp::copy(var_75);
                // if transmittance < TRANSMITTANCE_CUTOFF:                                       <L 719>
                var_275 = (var_273 < var_274);
                if (var_275) {
                    // break                                                                      <L 720>
                    wp::assign(var_34, var_272);
                    wp::assign(var_36, var_273);
                    goto end_for_0;
                }
                var_276 = wp::where(var_275, var_34, var_272);
                var_277 = wp::where(var_275, var_36, var_273);
            }
            var_278 = wp::where(var_69, var_276, var_34);
            var_279 = wp::where(var_69, var_277, var_36);
            wp::assign(var_34, var_278);
            wp::assign(var_36, var_279);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void gather_sample_targets_4043906f_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_targets,
    wp::array_t<wp::vec_t<2, wp::int32>> var_sample_xy,
    wp::int32 var_view,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt)
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
        wp::vec_t<2, wp::int32>* var_1;
        wp::vec_t<2, wp::int32> var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        //---------
        // forward
        // def gather_sample_targets(                                                             <L 751>
        // sample = wp.tid()                                                                      <L 759>
        var_0 = builtin_tid1d();
        // xy = sample_xy[sample]                                                                 <L 760>
        var_1 = wp::address(var_sample_xy, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // sample_gt[sample] = targets[view * width * height + xy[1] * width + xy[0]]             <L 761>
        var_4 = wp::mul(var_view, var_width);
        var_5 = wp::mul(var_4, var_height);
        var_7 = wp::extract(var_2, var_6);
        var_8 = wp::mul(var_7, var_width);
        var_9 = wp::add(var_5, var_8);
        var_11 = wp::extract(var_2, var_10);
        var_12 = wp::add(var_9, var_11);
        var_13 = wp::address(var_targets, var_12);
        var_14 = wp::load(var_13);
        wp::array_store(var_sample_gt, var_0, var_14);
    }
}



extern "C" __global__ void gather_sample_targets_4043906f_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_targets,
    wp::array_t<wp::vec_t<2, wp::int32>> var_sample_xy,
    wp::int32 var_view,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_targets,
    wp::array_t<wp::vec_t<2, wp::int32>> adj_sample_xy,
    wp::int32 adj_view,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sample_gt)
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
        wp::vec_t<2, wp::int32>* var_1;
        wp::vec_t<2, wp::int32> var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<2, wp::int32> adj_1 = {};
        wp::vec_t<2, wp::int32> adj_2 = {};
        wp::vec_t<2, wp::int32> adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        //---------
        // forward
        // def gather_sample_targets(                                                             <L 751>
        // sample = wp.tid()                                                                      <L 759>
        var_0 = builtin_tid1d();
        // xy = sample_xy[sample]                                                                 <L 760>
        var_1 = wp::address(var_sample_xy, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // sample_gt[sample] = targets[view * width * height + xy[1] * width + xy[0]]             <L 761>
        var_4 = wp::mul(var_view, var_width);
        var_5 = wp::mul(var_4, var_height);
        var_7 = wp::extract(var_2, var_6);
        var_8 = wp::mul(var_7, var_width);
        var_9 = wp::add(var_5, var_8);
        var_11 = wp::extract(var_2, var_10);
        var_12 = wp::add(var_9, var_11);
        var_13 = wp::address(var_targets, var_12);
        var_14 = wp::load(var_13);
        // wp::array_store(var_sample_gt, var_0, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_sample_gt, var_0, var_14, adj_sample_gt, adj_0, adj_13);
        wp::adj_address(var_targets, var_12, adj_targets, adj_12, adj_13);
        wp::adj_add(var_9, var_11, adj_9, adj_11, adj_12);
        wp::adj_extract(var_2, var_10, adj_2, adj_10, adj_11);
        wp::adj_add(var_5, var_8, adj_5, adj_8, adj_9);
        wp::adj_mul(var_7, var_width, adj_7, adj_width, adj_8);
        wp::adj_extract(var_2, var_6, adj_2, adj_6, adj_7);
        wp::adj_mul(var_4, var_height, adj_4, adj_height, adj_5);
        wp::adj_mul(var_view, var_width, adj_view, adj_width, adj_4);
        // adj: sample_gt[sample] = targets[view * width * height + xy[1] * width + xy[0]]        <L 761>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_sample_xy, var_0, adj_sample_xy, adj_0, adj_1);
        // adj: xy = sample_xy[sample]                                                            <L 760>
        // adj: sample = wp.tid()                                                                 <L 759>
        // adj: def gather_sample_targets(                                                        <L 751>
        continue;
    }
}



extern "C" __global__ void render_forward_63084a97_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::array_t<wp::vec_t<3, wp::float32>> var_targets,
    wp::array_t<wp::int32> var_view_ids,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_tiles_x,
    wp::int32 var_tiles_y,
    wp::int32 var_tile,
    wp::float32 var_focal,
    wp::int32 var_view_count,
    wp::vec_t<3, wp::float32> var_background,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_image,
    wp::array_t<wp::float32> var_loss)
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
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.5;
        wp::float32 var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.5;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::float32 var_25 = 0.0;
        const wp::float32 var_26 = 0.0;
        const wp::float32 var_27 = 0.0;
        wp::vec_t<3, wp::float32> var_28;
        const wp::float32 var_29 = 1.0;
        wp::float32 var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::uint64* var_39;
        wp::uint64 var_40;
        wp::uint64 var_41;
        wp::uint64 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<4, wp::float32>* var_46;
        wp::float32* var_47;
        wp::mat_t<4, 4, wp::float32>* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<4, wp::float32> var_54;
        wp::float32 var_55;
        wp::mat_t<4, 4, wp::float32> var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::float32 var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        const wp::float32 var_65 = 1.0;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 0.0001;
        bool var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        const wp::int32 var_83 = 3;
        wp::int32 var_84;
        wp::int32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        //---------
        // forward
        // def render_forward(                                                                    <L 449>
        // thread = wp.tid()                                                                      <L 474>
        var_0 = builtin_tid1d();
        // pixels = width * height                                                                <L 475>
        var_1 = wp::mul(var_width, var_height);
        // batch_view = thread // pixels                                                          <L 476>
        var_2 = wp::floordiv(var_0, var_1);
        // view = view_ids[batch_view]                                                            <L 477>
        var_3 = wp::address(var_view_ids, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // pixel = thread - batch_view * pixels                                                   <L 478>
        var_6 = wp::mul(var_2, var_1);
        var_7 = wp::sub(var_0, var_6);
        // px = float(pixel % width) + 0.5                                                        <L 479>
        var_8 = wp::mod(var_7, var_width);
        var_9 = wp::float(var_8);
        var_11 = wp::add(var_9, var_10);
        // py = float(pixel // width) + 0.5                                                       <L 480>
        var_12 = wp::floordiv(var_7, var_width);
        var_13 = wp::float(var_12);
        var_15 = wp::add(var_13, var_14);
        // record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)       <L 481>
        var_16 = wp::mul(var_2, var_tiles_x);
        var_17 = wp::mul(var_16, var_tiles_y);
        var_18 = wp::floordiv(var_7, var_width);
        var_19 = wp::floordiv(var_18, var_tile);
        var_20 = wp::mul(var_19, var_tiles_x);
        var_21 = wp::add(var_17, var_20);
        var_22 = wp::mod(var_7, var_width);
        var_23 = wp::floordiv(var_22, var_tile);
        var_24 = wp::add(var_21, var_23);
        // rgb = wp.vec3(0.0, 0.0, 0.0) # the accumulated color so far                            <L 482>
        var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
        // transmittance = float(1.0) # the light that is left after going through Gaussians before it       <L 483>
        var_30 = wp::float(var_29);
        // for entry in range(offsets[record], offsets[record + 1]):                              <L 484>
        var_31 = wp::address(var_offsets, var_24);
        var_33 = wp::add(var_24, var_32);
        var_34 = wp::address(var_offsets, var_33);
        var_36 = wp::load(var_31);
        var_37 = wp::load(var_34);
        var_35 = wp::range(var_36, var_37);
        start_for_0:;
            if (iter_cmp(var_35) == 0) goto end_for_0;
            var_38 = wp::iter_next(var_35);
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 485>
            var_39 = wp::address(var_pairs, var_38);
            var_40 = 4294967295ull;
            var_42 = wp::load(var_39);
            var_41 = wp::bit_and(var_42, var_40);
            var_43 = wp::int(var_41);
            // alpha = alpha_at_pixel(means[splat], log_scales[splat], quaternions[splat],        <L 486>
            var_44 = wp::address(var_means, var_43);
            var_45 = wp::address(var_log_scales, var_43);
            var_46 = wp::address(var_quaternions, var_43);
            // opacity_logits[splat], cameras[view], px, py,                                      <L 487>
            var_47 = wp::address(var_opacity_logits, var_43);
            var_48 = wp::address(var_cameras, var_4);
            // float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)       <L 488>
            var_49 = wp::float(var_width);
            var_50 = wp::float(var_height);
            var_52 = wp::load(var_44);
            var_53 = wp::load(var_45);
            var_54 = wp::load(var_46);
            var_55 = wp::load(var_47);
            var_56 = wp::load(var_48);
            var_51 = alpha_at_pixel_0(var_52, var_53, var_54, var_55, var_56, var_11, var_15, var_49, var_50, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 491>
            var_58 = (var_51 > var_57);
            if (var_58) {
                // colour = colour_at_view(color[splat])                                          <L 492>
                var_59 = wp::address(var_color, var_43);
                var_61 = wp::load(var_59);
                var_60 = colour_at_view_0(var_61);
                // rgb = rgb + transmittance * alpha * colour                                     <L 493>
                var_62 = wp::mul(var_30, var_51);
                var_63 = wp::mul(var_62, var_60);
                var_64 = wp::add(var_28, var_63);
                // transmittance = transmittance * (1.0 - alpha)                                  <L 494>
                var_66 = wp::sub(var_65, var_51);
                var_67 = wp::mul(var_30, var_66);
                // if transmittance < TRANSMITTANCE_CUTOFF:                                       <L 495>
                var_69 = (var_67 < var_68);
                if (var_69) {
                    // break                                                                      <L 496>
                    wp::assign(var_28, var_64);
                    wp::assign(var_30, var_67);
                    goto end_for_0;
                }
                var_70 = wp::where(var_69, var_28, var_64);
                var_71 = wp::where(var_69, var_30, var_67);
            }
            var_72 = wp::where(var_58, var_70, var_28);
            var_73 = wp::where(var_58, var_71, var_30);
            wp::assign(var_28, var_72);
            wp::assign(var_30, var_73);
            goto start_for_0;
        end_for_0:;
        // rgb = rgb + transmittance * background # the left light is shown as bg                 <L 497>
        var_74 = wp::mul(var_30, var_background);
        var_75 = wp::add(var_28, var_74);
        // image[thread] = rgb                                                                    <L 498>
        wp::array_store(var_image, var_0, var_75);
        // difference = rgb - targets[view * pixels + pixel]                                      <L 499>
        var_76 = wp::mul(var_4, var_1);
        var_77 = wp::add(var_76, var_7);
        var_78 = wp::address(var_targets, var_77);
        var_80 = wp::load(var_78);
        var_79 = wp::sub(var_75, var_80);
        // wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))       <L 500>
        var_82 = wp::dot(var_79, var_79);
        var_84 = wp::mul(var_83, var_1);
        var_85 = wp::mul(var_84, var_view_count);
        var_86 = wp::float(var_85);
        var_87 = wp::div(var_82, var_86);
        var_88 = wp::atomic_add(var_loss, var_81, var_87);
    }
}



extern "C" __global__ void render_forward_63084a97_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::array_t<wp::vec_t<3, wp::float32>> var_targets,
    wp::array_t<wp::int32> var_view_ids,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_tiles_x,
    wp::int32 var_tiles_y,
    wp::int32 var_tile,
    wp::float32 var_focal,
    wp::int32 var_view_count,
    wp::vec_t<3, wp::float32> var_background,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_image,
    wp::array_t<wp::float32> var_loss,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_means,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_quaternions,
    wp::array_t<wp::float32> adj_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> adj_cameras,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_targets,
    wp::array_t<wp::int32> adj_view_ids,
    wp::array_t<wp::int32> adj_offsets,
    wp::array_t<wp::uint64> adj_pairs,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::int32 adj_tiles_x,
    wp::int32 adj_tiles_y,
    wp::int32 adj_tile,
    wp::float32 adj_focal,
    wp::int32 adj_view_count,
    wp::vec_t<3, wp::float32> adj_background,
    wp::int32 adj_compact_enabled,
    wp::float32 adj_compact_beta,
    wp::float32 adj_compact_alpha_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_image,
    wp::array_t<wp::float32> adj_loss)
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
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.5;
        wp::float32 var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        const wp::float32 var_14 = 0.5;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::float32 var_25 = 0.0;
        const wp::float32 var_26 = 0.0;
        const wp::float32 var_27 = 0.0;
        wp::vec_t<3, wp::float32> var_28;
        const wp::float32 var_29 = 1.0;
        wp::float32 var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::uint64* var_39;
        wp::uint64 var_40;
        wp::uint64 var_41;
        wp::uint64 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<4, wp::float32>* var_46;
        wp::float32* var_47;
        wp::mat_t<4, 4, wp::float32>* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<4, wp::float32> var_54;
        wp::float32 var_55;
        wp::mat_t<4, 4, wp::float32> var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::float32 var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        const wp::float32 var_65 = 1.0;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 0.0001;
        bool var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::float32 var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        const wp::int32 var_83 = 3;
        wp::int32 var_84;
        wp::int32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::range_t adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::uint64 adj_39 = {};
        wp::uint64 adj_40 = {};
        wp::uint64 adj_41 = {};
        wp::uint64 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::vec_t<4, wp::float32> adj_46 = {};
        wp::float32 adj_47 = {};
        wp::mat_t<4, 4, wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<4, wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::mat_t<4, 4, wp::float32> adj_56 = {};
        wp::float32 adj_57 = {};
        bool adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::vec_t<3, wp::float32> adj_60 = {};
        wp::vec_t<3, wp::float32> adj_61 = {};
        wp::float32 adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::vec_t<3, wp::float32> adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        bool adj_69 = {};
        wp::vec_t<3, wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::float32 adj_73 = {};
        wp::vec_t<3, wp::float32> adj_74 = {};
        wp::vec_t<3, wp::float32> adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::vec_t<3, wp::float32> adj_78 = {};
        wp::vec_t<3, wp::float32> adj_79 = {};
        wp::vec_t<3, wp::float32> adj_80 = {};
        wp::int32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::float32 adj_88 = {};
        //---------
        // forward
        // def render_forward(                                                                    <L 449>
        // thread = wp.tid()                                                                      <L 474>
        var_0 = builtin_tid1d();
        // pixels = width * height                                                                <L 475>
        var_1 = wp::mul(var_width, var_height);
        // batch_view = thread // pixels                                                          <L 476>
        var_2 = wp::floordiv(var_0, var_1);
        // view = view_ids[batch_view]                                                            <L 477>
        var_3 = wp::address(var_view_ids, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // pixel = thread - batch_view * pixels                                                   <L 478>
        var_6 = wp::mul(var_2, var_1);
        var_7 = wp::sub(var_0, var_6);
        // px = float(pixel % width) + 0.5                                                        <L 479>
        var_8 = wp::mod(var_7, var_width);
        var_9 = wp::float(var_8);
        var_11 = wp::add(var_9, var_10);
        // py = float(pixel // width) + 0.5                                                       <L 480>
        var_12 = wp::floordiv(var_7, var_width);
        var_13 = wp::float(var_12);
        var_15 = wp::add(var_13, var_14);
        // record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)       <L 481>
        var_16 = wp::mul(var_2, var_tiles_x);
        var_17 = wp::mul(var_16, var_tiles_y);
        var_18 = wp::floordiv(var_7, var_width);
        var_19 = wp::floordiv(var_18, var_tile);
        var_20 = wp::mul(var_19, var_tiles_x);
        var_21 = wp::add(var_17, var_20);
        var_22 = wp::mod(var_7, var_width);
        var_23 = wp::floordiv(var_22, var_tile);
        var_24 = wp::add(var_21, var_23);
        // rgb = wp.vec3(0.0, 0.0, 0.0) # the accumulated color so far                            <L 482>
        var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
        // transmittance = float(1.0) # the light that is left after going through Gaussians before it       <L 483>
        var_30 = wp::float(var_29);
        // for entry in range(offsets[record], offsets[record + 1]):                              <L 484>
        var_31 = wp::address(var_offsets, var_24);
        var_33 = wp::add(var_24, var_32);
        var_34 = wp::address(var_offsets, var_33);
        var_36 = wp::load(var_31);
        var_37 = wp::load(var_34);
        var_35 = wp::range(var_36, var_37);
        // rgb = rgb + transmittance * background # the left light is shown as bg                 <L 497>
        var_74 = wp::mul(var_30, var_background);
        var_75 = wp::add(var_28, var_74);
        // image[thread] = rgb                                                                    <L 498>
        // wp::array_store(var_image, var_0, var_75);
        // difference = rgb - targets[view * pixels + pixel]                                      <L 499>
        var_76 = wp::mul(var_4, var_1);
        var_77 = wp::add(var_76, var_7);
        var_78 = wp::address(var_targets, var_77);
        var_80 = wp::load(var_78);
        var_79 = wp::sub(var_75, var_80);
        // wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))       <L 500>
        var_82 = wp::dot(var_79, var_79);
        var_84 = wp::mul(var_83, var_1);
        var_85 = wp::mul(var_84, var_view_count);
        var_86 = wp::float(var_85);
        var_87 = wp::div(var_82, var_86);
        // var_88 = wp::atomic_add(var_loss, var_81, var_87);
        //---------
        // reverse
        wp::adj_atomic_add(var_loss, var_81, var_87, adj_loss, adj_81, adj_87, adj_88);
        wp::adj_div(var_82, var_86, var_87, adj_82, adj_86, adj_87);
        wp::adj_float(var_85, adj_85, adj_86);
        wp::adj_mul(var_84, var_view_count, adj_84, adj_view_count, adj_85);
        wp::adj_mul(var_83, var_1, adj_83, adj_1, adj_84);
        wp::adj_dot(var_79, var_79, adj_79, adj_79, adj_82);
        // adj: wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))  <L 500>
        wp::adj_sub(var_75, var_80, adj_75, adj_78, adj_79);
        wp::adj_address(var_targets, var_77, adj_targets, adj_77, adj_78);
        wp::adj_add(var_76, var_7, adj_76, adj_7, adj_77);
        wp::adj_mul(var_4, var_1, adj_4, adj_1, adj_76);
        // adj: difference = rgb - targets[view * pixels + pixel]                                 <L 499>
        wp::adj_array_store(var_image, var_0, var_75, adj_image, adj_0, adj_75);
        // adj: image[thread] = rgb                                                               <L 498>
        wp::adj_add(var_28, var_74, adj_28, adj_74, adj_75);
        wp::adj_mul(var_30, var_background, adj_30, adj_background, adj_74);
        // adj: rgb = rgb + transmittance * background # the left light is shown as bg            <L 497>
        var_35 = wp::iter_reverse(var_35);
        start_for_0:;
            if (iter_cmp(var_35) == 0) goto end_for_0;
            var_38 = wp::iter_next(var_35);
        	adj_39 = {};
        	adj_40 = {};
        	adj_41 = {};
        	adj_42 = {};
        	adj_43 = {};
        	adj_44 = {};
        	adj_45 = {};
        	adj_46 = {};
        	adj_47 = {};
        	adj_48 = {};
        	adj_49 = {};
        	adj_50 = {};
        	adj_51 = {};
        	adj_52 = {};
        	adj_53 = {};
        	adj_54 = {};
        	adj_55 = {};
        	adj_56 = {};
        	adj_57 = {};
        	adj_58 = {};
        	adj_59 = {};
        	adj_60 = {};
        	adj_61 = {};
        	adj_62 = {};
        	adj_63 = {};
        	adj_64 = {};
        	adj_65 = {};
        	adj_66 = {};
        	adj_67 = {};
        	adj_68 = {};
        	adj_69 = {};
        	adj_70 = {};
        	adj_71 = {};
        	adj_72 = {};
        	adj_73 = {};
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 485>
            var_39 = wp::address(var_pairs, var_38);
            var_40 = 4294967295ull;
            var_42 = wp::load(var_39);
            var_41 = wp::bit_and(var_42, var_40);
            var_43 = wp::int(var_41);
            // alpha = alpha_at_pixel(means[splat], log_scales[splat], quaternions[splat],        <L 486>
            var_44 = wp::address(var_means, var_43);
            var_45 = wp::address(var_log_scales, var_43);
            var_46 = wp::address(var_quaternions, var_43);
            // opacity_logits[splat], cameras[view], px, py,                                      <L 487>
            var_47 = wp::address(var_opacity_logits, var_43);
            var_48 = wp::address(var_cameras, var_4);
            // float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)       <L 488>
            var_49 = wp::float(var_width);
            var_50 = wp::float(var_height);
            var_52 = wp::load(var_44);
            var_53 = wp::load(var_45);
            var_54 = wp::load(var_46);
            var_55 = wp::load(var_47);
            var_56 = wp::load(var_48);
            var_51 = alpha_at_pixel_0(var_52, var_53, var_54, var_55, var_56, var_11, var_15, var_49, var_50, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 491>
            var_58 = (var_51 > var_57);
            if (var_58) {
                // colour = colour_at_view(color[splat])                                          <L 492>
                var_59 = wp::address(var_color, var_43);
                var_61 = wp::load(var_59);
                var_60 = colour_at_view_0(var_61);
                // rgb = rgb + transmittance * alpha * colour                                     <L 493>
                var_62 = wp::mul(var_30, var_51);
                var_63 = wp::mul(var_62, var_60);
                var_64 = wp::add(var_28, var_63);
                // transmittance = transmittance * (1.0 - alpha)                                  <L 494>
                var_66 = wp::sub(var_65, var_51);
                var_67 = wp::mul(var_30, var_66);
                // if transmittance < TRANSMITTANCE_CUTOFF:                                       <L 495>
                var_69 = (var_67 < var_68);
                if (var_69) {
                    // break                                                                      <L 496>
                    wp::assign(var_28, var_64);
                    wp::assign(var_30, var_67);
                    goto end_for_0;
                }
                var_70 = wp::where(var_69, var_28, var_64);
                var_71 = wp::where(var_69, var_30, var_67);
            }
            var_72 = wp::where(var_58, var_70, var_28);
            var_73 = wp::where(var_58, var_71, var_30);
            wp::assign(var_28, var_72);
            wp::assign(var_30, var_73);
            wp::adj_assign(var_30, var_73, adj_30, adj_73);
            wp::adj_assign(var_28, var_72, adj_28, adj_72);
            wp::adj_where(var_58, var_71, var_30, adj_58, adj_71, adj_30, adj_73);
            wp::adj_where(var_58, var_70, var_28, adj_58, adj_70, adj_28, adj_72);
            if (var_58) {
                wp::adj_where(var_69, var_30, var_67, adj_69, adj_30, adj_67, adj_71);
                wp::adj_where(var_69, var_28, var_64, adj_69, adj_28, adj_64, adj_70);
                if (var_69) {
                    wp::adj_assign(var_30, var_67, adj_30, adj_67);
                    wp::adj_assign(var_28, var_64, adj_28, adj_64);
                    // adj: break                                                                 <L 496>
                }
                // adj: if transmittance < TRANSMITTANCE_CUTOFF:                                  <L 495>
                wp::adj_mul(var_30, var_66, adj_30, adj_66, adj_67);
                wp::adj_sub(var_65, var_51, adj_65, adj_51, adj_66);
                // adj: transmittance = transmittance * (1.0 - alpha)                             <L 494>
                wp::adj_add(var_28, var_63, adj_28, adj_63, adj_64);
                wp::adj_mul(var_62, var_60, adj_62, adj_60, adj_63);
                wp::adj_mul(var_30, var_51, adj_30, adj_51, adj_62);
                // adj: rgb = rgb + transmittance * alpha * colour                                <L 493>
                adj_colour_at_view_0(var_61, adj_59, adj_60);
                wp::adj_address(var_color, var_43, adj_color, adj_43, adj_59);
                // adj: colour = colour_at_view(color[splat])                                     <L 492>
            }
            // adj: if alpha > 0.0:                                                               <L 491>
            adj_alpha_at_pixel_0(var_52, var_53, var_54, var_55, var_56, var_11, var_15, var_49, var_50, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min, adj_44, adj_45, adj_46, adj_47, adj_48, adj_11, adj_15, adj_49, adj_50, adj_focal, adj_compact_enabled, adj_compact_beta, adj_compact_alpha_min, adj_51);
            wp::adj_float(var_height, adj_height, adj_50);
            wp::adj_float(var_width, adj_width, adj_49);
            // adj: float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)  <L 488>
            wp::adj_address(var_cameras, var_4, adj_cameras, adj_4, adj_48);
            wp::adj_address(var_opacity_logits, var_43, adj_opacity_logits, adj_43, adj_47);
            // adj: opacity_logits[splat], cameras[view], px, py,                                 <L 487>
            wp::adj_address(var_quaternions, var_43, adj_quaternions, adj_43, adj_46);
            wp::adj_address(var_log_scales, var_43, adj_log_scales, adj_43, adj_45);
            wp::adj_address(var_means, var_43, adj_means, adj_43, adj_44);
            // adj: alpha = alpha_at_pixel(means[splat], log_scales[splat], quaternions[splat],   <L 486>
            wp::adj_address(var_pairs, var_38, adj_pairs, adj_38, adj_39);
            // adj: splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                             <L 485>
        	goto start_for_0;
        end_for_0:;
        wp::adj_address(var_offsets, var_33, adj_offsets, adj_33, adj_34);
        wp::adj_add(var_24, var_32, adj_24, adj_32, adj_33);
        wp::adj_address(var_offsets, var_24, adj_offsets, adj_24, adj_31);
        // adj: for entry in range(offsets[record], offsets[record + 1]):                         <L 484>
        wp::adj_float(var_29, adj_29, adj_30);
        // adj: transmittance = float(1.0) # the light that is left after going through Gaussians before it  <L 483>
        wp::adj_vec_t(var_25, var_26, var_27, adj_25, adj_26, adj_27, adj_28);
        // adj: rgb = wp.vec3(0.0, 0.0, 0.0) # the accumulated color so far                       <L 482>
        wp::adj_add(var_21, var_23, adj_21, adj_23, adj_24);
        wp::adj_mod(var_7, var_width, adj_7, adj_width, adj_22);
        wp::adj_add(var_17, var_20, adj_17, adj_20, adj_21);
        wp::adj_mul(var_19, var_tiles_x, adj_19, adj_tiles_x, adj_20);
        wp::adj_mul(var_16, var_tiles_y, adj_16, adj_tiles_y, adj_17);
        wp::adj_mul(var_2, var_tiles_x, adj_2, adj_tiles_x, adj_16);
        // adj: record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)  <L 481>
        wp::adj_add(var_13, var_14, adj_13, adj_14, adj_15);
        wp::adj_float(var_12, adj_12, adj_13);
        // adj: py = float(pixel // width) + 0.5                                                  <L 480>
        wp::adj_add(var_9, var_10, adj_9, adj_10, adj_11);
        wp::adj_float(var_8, adj_8, adj_9);
        wp::adj_mod(var_7, var_width, adj_7, adj_width, adj_8);
        // adj: px = float(pixel % width) + 0.5                                                   <L 479>
        wp::adj_sub(var_0, var_6, adj_0, adj_6, adj_7);
        wp::adj_mul(var_2, var_1, adj_2, adj_1, adj_6);
        // adj: pixel = thread - batch_view * pixels                                              <L 478>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_view_ids, var_2, adj_view_ids, adj_2, adj_3);
        // adj: view = view_ids[batch_view]                                                       <L 477>
        // adj: batch_view = thread // pixels                                                     <L 476>
        wp::adj_mul(var_width, var_height, adj_width, adj_height, adj_1);
        // adj: pixels = width * height                                                           <L 475>
        // adj: thread = wp.tid()                                                                 <L 474>
        // adj: def render_forward(                                                               <L 449>
        continue;
    }
}



extern "C" __global__ void render_sparse_backward_a1ba08ca_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_means,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_opacity_logits,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras,
    wp::int32 var_view,
    wp::array_t<wp::vec_t<2, wp::int32>> var_sample_xy,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::uint64> var_pairs,
    wp::int32 var_samples_per_tile,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::float32 var_focal,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rendered_rgb,
    wp::array_t<wp::int32> var_last_contributor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient,
    wp::int32 var_compact_enabled,
    wp::float32 var_compact_beta,
    wp::float32 var_compact_alpha_min,
    wp::array_t<wp::float32> var_mean_grad_flat,
    wp::array_t<wp::float32> var_scale_grad_flat,
    wp::array_t<wp::float32> var_quaternion_grad_flat,
    wp::array_t<wp::float32> var_opacity_grad,
    wp::array_t<wp::float32> var_color_grad_flat)
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
        wp::int32 var_1;
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.5;
        wp::float32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.5;
        wp::float32 var_14;
        wp::mat_t<4, 4, wp::float32>* var_15;
        wp::mat_t<4, 4, wp::float32> var_16;
        wp::mat_t<4, 4, wp::float32> var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::float32 var_24 = 0.0;
        const wp::float32 var_25 = 0.0;
        const wp::float32 var_26 = 0.0;
        wp::vec_t<3, wp::float32> var_27;
        const wp::float32 var_28 = 1.0;
        wp::float32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::range_t var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::uint64* var_39;
        wp::uint64 var_40;
        wp::uint64 var_41;
        wp::uint64 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32>* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<4, wp::float32>* var_50;
        wp::vec_t<4, wp::float32> var_51;
        wp::vec_t<4, wp::float32> var_52;
        wp::float32* var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        wp::vec_t<3, wp::float32>* var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::float32 var_64 = 1.0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::float32 var_72;
        //---------
        // forward
        // def render_sparse_backward(                                                            <L 834>
        // thread = wp.tid()                                                                      <L 861>
        var_0 = builtin_tid1d();
        // record = thread // samples_per_tile                                                    <L 862>
        var_1 = wp::floordiv(var_0, var_samples_per_tile);
        // xy = sample_xy[thread]                                                                 <L 863>
        var_2 = wp::address(var_sample_xy, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // px = float(xy[0]) + 0.5                                                                <L 864>
        var_6 = wp::extract(var_3, var_5);
        var_7 = wp::float(var_6);
        var_9 = wp::add(var_7, var_8);
        // py = float(xy[1]) + 0.5                                                                <L 865>
        var_11 = wp::extract(var_3, var_10);
        var_12 = wp::float(var_11);
        var_14 = wp::add(var_12, var_13);
        // camera = cameras[view]                                                                 <L 866>
        var_15 = wp::address(var_cameras, var_view);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // final_rgb = rendered_rgb[thread]                                                       <L 867>
        var_18 = wp::address(var_rendered_rgb, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // pixel_grad = pixel_gradient[thread]                                                    <L 868>
        var_21 = wp::address(var_pixel_gradient, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // prefix_rgb = wp.vec3(0.0, 0.0, 0.0)                                                    <L 869>
        var_27 = wp::vec_t<3, wp::float32>(var_24, var_25, var_26);
        // transmittance = float(1.0)                                                             <L 870>
        var_29 = wp::float(var_28);
        // stop = last_contributor[thread]                                                        <L 871>
        var_30 = wp::address(var_last_contributor, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // for entry in range(offsets[record], stop + 1):                                         <L 872>
        var_33 = wp::address(var_offsets, var_1);
        var_35 = wp::add(var_31, var_34);
        var_37 = wp::load(var_33);
        var_36 = wp::range(var_37, var_35);
        start_for_0:;
            if (iter_cmp(var_36) == 0) goto end_for_0;
            var_38 = wp::iter_next(var_36);
            // splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))                                  <L 873>
            var_39 = wp::address(var_pairs, var_38);
            var_40 = 4294967295ull;
            var_42 = wp::load(var_39);
            var_41 = wp::bit_and(var_42, var_40);
            var_43 = wp::int(var_41);
            // mean = means[splat]                                                                <L 874>
            var_44 = wp::address(var_means, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // log_scale = log_scales[splat]                                                      <L 875>
            var_47 = wp::address(var_log_scales, var_43);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // quaternion = quaternions[splat]                                                    <L 876>
            var_50 = wp::address(var_quaternions, var_43);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // opacity_logit = opacity_logits[splat]                                              <L 877>
            var_53 = wp::address(var_opacity_logits, var_43);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // alpha = alpha_at_pixel(                                                            <L 878>
            // mean, log_scale, quaternion, opacity_logit, camera,                                <L 879>
            // px, py, float(width), float(height), focal,                                        <L 880>
            var_56 = wp::float(var_width);
            var_57 = wp::float(var_height);
            // compact_enabled, compact_beta, compact_alpha_min,                                  <L 881>
            var_58 = alpha_at_pixel_0(var_45, var_48, var_51, var_54, var_16, var_9, var_14, var_56, var_57, var_focal, var_compact_enabled, var_compact_beta, var_compact_alpha_min);
            // if alpha > 0.0:                                                                    <L 883>
            var_60 = (var_58 > var_59);
            if (var_60) {
                // colour = colour_at_view(color[splat])                                          <L 884>
                var_61 = wp::address(var_color, var_43);
                var_63 = wp::load(var_61);
                var_62 = colour_at_view_0(var_63);
                // next_transmittance = transmittance * (1.0 - alpha)                             <L 885>
                var_65 = wp::sub(var_64, var_58);
                var_66 = wp::mul(var_29, var_65);
                // prefix_rgb = prefix_rgb + transmittance * alpha * colour                       <L 898>
                var_67 = wp::mul(var_29, var_58);
                var_68 = wp::mul(var_67, var_62);
                var_69 = wp::add(var_27, var_68);
                // transmittance = next_transmittance                                             <L 899>
                var_70 = wp::copy(var_66);
            }
            var_71 = wp::where(var_60, var_69, var_27);
            var_72 = wp::where(var_60, var_70, var_29);
            wp::assign(var_27, var_71);
            wp::assign(var_29, var_72);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void constrain_parameters_ad4aef34_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::float32 var_minimum_log_scale,
    wp::float32 var_maximum_log_scale)
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
        const wp::int32 var_4 = 0;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 1.0;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        const wp::float32 var_25 = 1.0;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 1.0;
        wp::float32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        //---------
        // forward
        // def constrain_parameters(                                                              <L 935>
        // i = wp.tid()                                                                           <L 941>
        var_0 = builtin_tid1d();
        // scale = log_scales[i]                                                                  <L 942>
        var_1 = wp::address(var_log_scales, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // log_scales[i] = wp.vec3(                                                               <L 943>
        // wp.clamp(scale[0], minimum_log_scale, maximum_log_scale),                              <L 944>
        var_5 = wp::extract(var_2, var_4);
        var_6 = wp::clamp(var_5, var_minimum_log_scale, var_maximum_log_scale);
        // wp.clamp(scale[1], minimum_log_scale, maximum_log_scale),                              <L 945>
        var_8 = wp::extract(var_2, var_7);
        var_9 = wp::clamp(var_8, var_minimum_log_scale, var_maximum_log_scale);
        // wp.clamp(scale[2], minimum_log_scale, maximum_log_scale),                              <L 946>
        var_11 = wp::extract(var_2, var_10);
        var_12 = wp::clamp(var_11, var_minimum_log_scale, var_maximum_log_scale);
        var_13 = wp::vec_t<3, wp::float32>(var_6, var_9, var_12);
        // log_scales[i] = wp.vec3(                                                               <L 943>
        wp::array_store(var_log_scales, var_0, var_13);
        // colour = color[i]                                                                      <L 948>
        var_14 = wp::address(var_color, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // color[i] = wp.vec3(                                                                    <L 949>
        // wp.clamp(colour[0], 0.0, 1.0),                                                         <L 950>
        var_18 = wp::extract(var_15, var_17);
        var_21 = wp::clamp(var_18, var_19, var_20);
        // wp.clamp(colour[1], 0.0, 1.0),                                                         <L 951>
        var_23 = wp::extract(var_15, var_22);
        var_26 = wp::clamp(var_23, var_24, var_25);
        // wp.clamp(colour[2], 0.0, 1.0),                                                         <L 952>
        var_28 = wp::extract(var_15, var_27);
        var_31 = wp::clamp(var_28, var_29, var_30);
        var_32 = wp::vec_t<3, wp::float32>(var_21, var_26, var_31);
        // color[i] = wp.vec3(                                                                    <L 949>
        wp::array_store(var_color, var_0, var_32);
    }
}



extern "C" __global__ void constrain_parameters_ad4aef34_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales,
    wp::array_t<wp::vec_t<3, wp::float32>> var_color,
    wp::float32 var_minimum_log_scale,
    wp::float32 var_maximum_log_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_log_scales,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_color,
    wp::float32 adj_minimum_log_scale,
    wp::float32 adj_maximum_log_scale)
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
        const wp::int32 var_4 = 0;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 1.0;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        const wp::float32 var_25 = 1.0;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 1.0;
        wp::float32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        //---------
        // forward
        // def constrain_parameters(                                                              <L 935>
        // i = wp.tid()                                                                           <L 941>
        var_0 = builtin_tid1d();
        // scale = log_scales[i]                                                                  <L 942>
        var_1 = wp::address(var_log_scales, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // log_scales[i] = wp.vec3(                                                               <L 943>
        // wp.clamp(scale[0], minimum_log_scale, maximum_log_scale),                              <L 944>
        var_5 = wp::extract(var_2, var_4);
        var_6 = wp::clamp(var_5, var_minimum_log_scale, var_maximum_log_scale);
        // wp.clamp(scale[1], minimum_log_scale, maximum_log_scale),                              <L 945>
        var_8 = wp::extract(var_2, var_7);
        var_9 = wp::clamp(var_8, var_minimum_log_scale, var_maximum_log_scale);
        // wp.clamp(scale[2], minimum_log_scale, maximum_log_scale),                              <L 946>
        var_11 = wp::extract(var_2, var_10);
        var_12 = wp::clamp(var_11, var_minimum_log_scale, var_maximum_log_scale);
        var_13 = wp::vec_t<3, wp::float32>(var_6, var_9, var_12);
        // log_scales[i] = wp.vec3(                                                               <L 943>
        // wp::array_store(var_log_scales, var_0, var_13);
        // colour = color[i]                                                                      <L 948>
        var_14 = wp::address(var_color, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // color[i] = wp.vec3(                                                                    <L 949>
        // wp.clamp(colour[0], 0.0, 1.0),                                                         <L 950>
        var_18 = wp::extract(var_15, var_17);
        var_21 = wp::clamp(var_18, var_19, var_20);
        // wp.clamp(colour[1], 0.0, 1.0),                                                         <L 951>
        var_23 = wp::extract(var_15, var_22);
        var_26 = wp::clamp(var_23, var_24, var_25);
        // wp.clamp(colour[2], 0.0, 1.0),                                                         <L 952>
        var_28 = wp::extract(var_15, var_27);
        var_31 = wp::clamp(var_28, var_29, var_30);
        var_32 = wp::vec_t<3, wp::float32>(var_21, var_26, var_31);
        // color[i] = wp.vec3(                                                                    <L 949>
        // wp::array_store(var_color, var_0, var_32);
        //---------
        // reverse
        wp::adj_array_store(var_color, var_0, var_32, adj_color, adj_0, adj_32);
        // adj: color[i] = wp.vec3(                                                               <L 949>
        wp::adj_vec_t(var_21, var_26, var_31, adj_21, adj_26, adj_31, adj_32);
        wp::adj_clamp(var_28, var_29, var_30, adj_28, adj_29, adj_30, adj_31);
        wp::adj_extract(var_15, var_27, adj_15, adj_27, adj_28);
        // adj: wp.clamp(colour[2], 0.0, 1.0),                                                    <L 952>
        wp::adj_clamp(var_23, var_24, var_25, adj_23, adj_24, adj_25, adj_26);
        wp::adj_extract(var_15, var_22, adj_15, adj_22, adj_23);
        // adj: wp.clamp(colour[1], 0.0, 1.0),                                                    <L 951>
        wp::adj_clamp(var_18, var_19, var_20, adj_18, adj_19, adj_20, adj_21);
        wp::adj_extract(var_15, var_17, adj_15, adj_17, adj_18);
        // adj: wp.clamp(colour[0], 0.0, 1.0),                                                    <L 950>
        // adj: color[i] = wp.vec3(                                                               <L 949>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_color, var_0, adj_color, adj_0, adj_14);
        // adj: colour = color[i]                                                                 <L 948>
        wp::adj_array_store(var_log_scales, var_0, var_13, adj_log_scales, adj_0, adj_13);
        // adj: log_scales[i] = wp.vec3(                                                          <L 943>
        wp::adj_vec_t(var_6, var_9, var_12, adj_6, adj_9, adj_12, adj_13);
        wp::adj_clamp(var_11, var_minimum_log_scale, var_maximum_log_scale, adj_11, adj_minimum_log_scale, adj_maximum_log_scale, adj_12);
        wp::adj_extract(var_2, var_10, adj_2, adj_10, adj_11);
        // adj: wp.clamp(scale[2], minimum_log_scale, maximum_log_scale),                         <L 946>
        wp::adj_clamp(var_8, var_minimum_log_scale, var_maximum_log_scale, adj_8, adj_minimum_log_scale, adj_maximum_log_scale, adj_9);
        wp::adj_extract(var_2, var_7, adj_2, adj_7, adj_8);
        // adj: wp.clamp(scale[1], minimum_log_scale, maximum_log_scale),                         <L 945>
        wp::adj_clamp(var_5, var_minimum_log_scale, var_maximum_log_scale, adj_5, adj_minimum_log_scale, adj_maximum_log_scale, adj_6);
        wp::adj_extract(var_2, var_4, adj_2, adj_4, adj_5);
        // adj: wp.clamp(scale[0], minimum_log_scale, maximum_log_scale),                         <L 944>
        // adj: log_scales[i] = wp.vec3(                                                          <L 943>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_log_scales, var_0, adj_log_scales, adj_0, adj_1);
        // adj: scale = log_scales[i]                                                             <L 942>
        // adj: i = wp.tid()                                                                      <L 941>
        // adj: def constrain_parameters(                                                         <L 935>
        continue;
    }
}



extern "C" __global__ void sparse_mse_pixel_gradient_c7e18c15_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rendered_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt,
    wp::int32 var_num_samples,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient)
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
        const wp::float32 var_1 = 2.0;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        wp::float32 var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // forward
        // def sparse_mse_pixel_gradient(                                                         <L 765>
        // sample = wp.tid()                                                                      <L 771>
        var_0 = builtin_tid1d();
        // pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(       <L 772>
        var_2 = wp::address(var_rendered_rgb, var_0);
        var_3 = wp::address(var_sample_gt, var_0);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::mul(var_1, var_4);
        // 3 * num_samples                                                                        <L 773>
        var_9 = wp::mul(var_8, var_num_samples);
        var_10 = wp::float(var_9);
        var_11 = wp::div(var_7, var_10);
        // pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(       <L 772>
        wp::array_store(var_pixel_gradient, var_0, var_11);
    }
}



extern "C" __global__ void sparse_mse_pixel_gradient_c7e18c15_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rendered_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sample_gt,
    wp::int32 var_num_samples,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_rendered_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sample_gt,
    wp::int32 adj_num_samples,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_pixel_gradient)
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
        const wp::float32 var_1 = 2.0;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        wp::float32 var_10;
        wp::vec_t<3, wp::float32> var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        //---------
        // forward
        // def sparse_mse_pixel_gradient(                                                         <L 765>
        // sample = wp.tid()                                                                      <L 771>
        var_0 = builtin_tid1d();
        // pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(       <L 772>
        var_2 = wp::address(var_rendered_rgb, var_0);
        var_3 = wp::address(var_sample_gt, var_0);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::mul(var_1, var_4);
        // 3 * num_samples                                                                        <L 773>
        var_9 = wp::mul(var_8, var_num_samples);
        var_10 = wp::float(var_9);
        var_11 = wp::div(var_7, var_10);
        // pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(       <L 772>
        // wp::array_store(var_pixel_gradient, var_0, var_11);
        //---------
        // reverse
        wp::adj_array_store(var_pixel_gradient, var_0, var_11, adj_pixel_gradient, adj_0, adj_11);
        // adj: pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(  <L 772>
        wp::adj_div(var_7, var_10, adj_7, adj_10, adj_11);
        wp::adj_float(var_9, adj_9, adj_10);
        wp::adj_mul(var_8, var_num_samples, adj_8, adj_num_samples, adj_9);
        // adj: 3 * num_samples                                                                   <L 773>
        wp::adj_mul(var_1, var_4, adj_1, adj_4, adj_7);
        wp::adj_sub(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_sample_gt, var_0, adj_sample_gt, adj_0, adj_3);
        wp::adj_address(var_rendered_rgb, var_0, adj_rendered_rgb, adj_0, adj_2);
        // adj: pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(  <L 772>
        // adj: sample = wp.tid()                                                                 <L 771>
        // adj: def sparse_mse_pixel_gradient(                                                    <L 765>
        continue;
    }
}



extern "C" __global__ void pack_vec3_gradient_e4e3849b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_packed)
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
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        wp::float32* var_3;
        const wp::int32 var_4 = 3;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::float32* var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def pack_vec3_gradient(flat: wp.array(dtype=wp.float32), packed: wp.array(dtype=wp.vec3)):       <L 903>
        // i = wp.tid()                                                                           <L 904>
        var_0 = builtin_tid1d();
        // packed[i] = wp.vec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])                     <L 905>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_flat, var_2);
        var_5 = wp::mul(var_0, var_4);
        var_7 = wp::add(var_5, var_6);
        var_8 = wp::address(var_flat, var_7);
        var_10 = wp::mul(var_0, var_9);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_flat, var_12);
        var_15 = wp::load(var_3);
        var_16 = wp::load(var_8);
        var_17 = wp::load(var_13);
        var_14 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        wp::array_store(var_packed, var_0, var_14);
    }
}



extern "C" __global__ void pack_vec3_gradient_e4e3849b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_packed,
    wp::array_t<wp::float32> adj_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_packed)
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
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        wp::float32* var_3;
        const wp::int32 var_4 = 3;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::float32* var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        //---------
        // forward
        // def pack_vec3_gradient(flat: wp.array(dtype=wp.float32), packed: wp.array(dtype=wp.vec3)):       <L 903>
        // i = wp.tid()                                                                           <L 904>
        var_0 = builtin_tid1d();
        // packed[i] = wp.vec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])                     <L 905>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_flat, var_2);
        var_5 = wp::mul(var_0, var_4);
        var_7 = wp::add(var_5, var_6);
        var_8 = wp::address(var_flat, var_7);
        var_10 = wp::mul(var_0, var_9);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_flat, var_12);
        var_15 = wp::load(var_3);
        var_16 = wp::load(var_8);
        var_17 = wp::load(var_13);
        var_14 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        // wp::array_store(var_packed, var_0, var_14);
        //---------
        // reverse
        wp::adj_array_store(var_packed, var_0, var_14, adj_packed, adj_0, adj_14);
        wp::adj_vec_t(var_15, var_16, var_17, adj_3, adj_8, adj_13, adj_14);
        wp::adj_address(var_flat, var_12, adj_flat, adj_12, adj_13);
        wp::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
        wp::adj_mul(var_0, var_9, adj_0, adj_9, adj_10);
        wp::adj_address(var_flat, var_7, adj_flat, adj_7, adj_8);
        wp::adj_add(var_5, var_6, adj_5, adj_6, adj_7);
        wp::adj_mul(var_0, var_4, adj_0, adj_4, adj_5);
        wp::adj_address(var_flat, var_2, adj_flat, adj_2, adj_3);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: packed[i] = wp.vec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])                <L 905>
        // adj: i = wp.tid()                                                                      <L 904>
        // adj: def pack_vec3_gradient(flat: wp.array(dtype=wp.float32), packed: wp.array(dtype=wp.vec3)):  <L 903>
        continue;
    }
}



extern "C" __global__ void mse_pixel_gradient_f38ca9fa_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_image,
    wp::array_t<wp::vec_t<3, wp::float32>> var_target,
    wp::array_t<wp::int32> var_view_ids,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_view_count,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient)
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
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::float32 var_14 = 2.0;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::vec_t<3, wp::float32> var_20;
        //---------
        // forward
        // def mse_pixel_gradient(                                                                <L 724>
        // thread = wp.tid()                                                                      <L 738>
        var_0 = builtin_tid1d();
        // pixels = width * height                                                                <L 739>
        var_1 = wp::mul(var_width, var_height);
        // batch_view = thread // pixels                                                          <L 740>
        var_2 = wp::floordiv(var_0, var_1);
        // pixel = thread - batch_view * pixels                                                   <L 741>
        var_3 = wp::mul(var_2, var_1);
        var_4 = wp::sub(var_0, var_3);
        // target_pixel = view_ids[batch_view] * pixels + pixel                                   <L 742>
        var_5 = wp::address(var_view_ids, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::mul(var_7, var_1);
        var_8 = wp::add(var_6, var_4);
        // difference = image[thread] - target[target_pixel]                                      <L 745>
        var_9 = wp::address(var_image, var_0);
        var_10 = wp::address(var_target, var_8);
        var_12 = wp::load(var_9);
        var_13 = wp::load(var_10);
        var_11 = wp::sub(var_12, var_13);
        // pixel_gradient[thread] = difference * (2.0 / float(3 * pixels * view_count))           <L 747>
        var_16 = wp::mul(var_15, var_1);
        var_17 = wp::mul(var_16, var_view_count);
        var_18 = wp::float(var_17);
        var_19 = wp::div(var_14, var_18);
        var_20 = wp::mul(var_11, var_19);
        wp::array_store(var_pixel_gradient, var_0, var_20);
    }
}



extern "C" __global__ void mse_pixel_gradient_f38ca9fa_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_image,
    wp::array_t<wp::vec_t<3, wp::float32>> var_target,
    wp::array_t<wp::int32> var_view_ids,
    wp::int32 var_width,
    wp::int32 var_height,
    wp::int32 var_view_count,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pixel_gradient,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_image,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_target,
    wp::array_t<wp::int32> adj_view_ids,
    wp::int32 adj_width,
    wp::int32 adj_height,
    wp::int32 adj_view_count,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_pixel_gradient)
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
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::float32 var_14 = 2.0;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::vec_t<3, wp::float32> var_20;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        //---------
        // forward
        // def mse_pixel_gradient(                                                                <L 724>
        // thread = wp.tid()                                                                      <L 738>
        var_0 = builtin_tid1d();
        // pixels = width * height                                                                <L 739>
        var_1 = wp::mul(var_width, var_height);
        // batch_view = thread // pixels                                                          <L 740>
        var_2 = wp::floordiv(var_0, var_1);
        // pixel = thread - batch_view * pixels                                                   <L 741>
        var_3 = wp::mul(var_2, var_1);
        var_4 = wp::sub(var_0, var_3);
        // target_pixel = view_ids[batch_view] * pixels + pixel                                   <L 742>
        var_5 = wp::address(var_view_ids, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::mul(var_7, var_1);
        var_8 = wp::add(var_6, var_4);
        // difference = image[thread] - target[target_pixel]                                      <L 745>
        var_9 = wp::address(var_image, var_0);
        var_10 = wp::address(var_target, var_8);
        var_12 = wp::load(var_9);
        var_13 = wp::load(var_10);
        var_11 = wp::sub(var_12, var_13);
        // pixel_gradient[thread] = difference * (2.0 / float(3 * pixels * view_count))           <L 747>
        var_16 = wp::mul(var_15, var_1);
        var_17 = wp::mul(var_16, var_view_count);
        var_18 = wp::float(var_17);
        var_19 = wp::div(var_14, var_18);
        var_20 = wp::mul(var_11, var_19);
        // wp::array_store(var_pixel_gradient, var_0, var_20);
        //---------
        // reverse
        wp::adj_array_store(var_pixel_gradient, var_0, var_20, adj_pixel_gradient, adj_0, adj_20);
        wp::adj_mul(var_11, var_19, adj_11, adj_19, adj_20);
        wp::adj_div(var_14, var_18, var_19, adj_14, adj_18, adj_19);
        wp::adj_float(var_17, adj_17, adj_18);
        wp::adj_mul(var_16, var_view_count, adj_16, adj_view_count, adj_17);
        wp::adj_mul(var_15, var_1, adj_15, adj_1, adj_16);
        // adj: pixel_gradient[thread] = difference * (2.0 / float(3 * pixels * view_count))      <L 747>
        wp::adj_sub(var_12, var_13, adj_9, adj_10, adj_11);
        wp::adj_address(var_target, var_8, adj_target, adj_8, adj_10);
        wp::adj_address(var_image, var_0, adj_image, adj_0, adj_9);
        // adj: difference = image[thread] - target[target_pixel]                                 <L 745>
        wp::adj_add(var_6, var_4, adj_6, adj_4, adj_8);
        wp::adj_mul(var_7, var_1, adj_5, adj_1, adj_6);
        wp::adj_address(var_view_ids, var_2, adj_view_ids, adj_2, adj_5);
        // adj: target_pixel = view_ids[batch_view] * pixels + pixel                              <L 742>
        wp::adj_sub(var_0, var_3, adj_0, adj_3, adj_4);
        wp::adj_mul(var_2, var_1, adj_2, adj_1, adj_3);
        // adj: pixel = thread - batch_view * pixels                                              <L 741>
        // adj: batch_view = thread // pixels                                                     <L 740>
        wp::adj_mul(var_width, var_height, adj_width, adj_height, adj_1);
        // adj: pixels = width * height                                                           <L 739>
        // adj: thread = wp.tid()                                                                 <L 738>
        // adj: def mse_pixel_gradient(                                                           <L 724>
        continue;
    }
}



extern "C" __global__ void adam_quaternion_step_fb5bfef0_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_gradients,
    wp::array_t<wp::vec_t<4, wp::float32>> var_m,
    wp::array_t<wp::vec_t<4, wp::float32>> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps)
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
        const wp::int32 var_1 = 4;
        wp::int32 var_2;
        wp::float32* var_3;
        const wp::int32 var_4 = 4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::float32* var_8;
        const wp::int32 var_9 = 4;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        wp::int32 var_12;
        wp::float32* var_13;
        const wp::int32 var_14 = 4;
        wp::int32 var_15;
        const wp::int32 var_16 = 3;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::vec_t<4, wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<4, wp::float32>* var_24;
        wp::vec_t<4, wp::float32> var_25;
        wp::vec_t<4, wp::float32> var_26;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::vec_t<4, wp::float32> var_29;
        wp::vec_t<4, wp::float32> var_30;
        wp::vec_t<4, wp::float32>* var_31;
        wp::vec_t<4, wp::float32> var_32;
        wp::vec_t<4, wp::float32> var_33;
        const wp::float32 var_34 = 1.0;
        wp::float32 var_35;
        wp::vec_t<4, wp::float32> var_36;
        wp::vec_t<4, wp::float32> var_37;
        wp::vec_t<4, wp::float32> var_38;
        wp::vec_t<4, wp::float32>* var_39;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<4, wp::float32> var_45;
        wp::vec_t<4, wp::float32> var_46;
        wp::vec_t<4, wp::float32>* var_47;
        const wp::float32 var_48 = 1.0;
        const wp::float32 var_49 = 1.0;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::vec_t<4, wp::float32> var_53;
        wp::vec_t<4, wp::float32> var_54;
        const wp::int32 var_55 = 0;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 1;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 2;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 3;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::vec_t<4, wp::float32> var_67;
        wp::vec_t<4, wp::float32> var_68;
        wp::vec_t<4, wp::float32>* var_69;
        wp::vec_t<4, wp::float32> var_70;
        wp::vec_t<4, wp::float32> var_71;
        wp::vec_t<4, wp::float32> var_72;
        wp::vec_t<4, wp::float32> var_73;
        wp::vec_t<4, wp::float32> var_74;
        wp::float32 var_75;
        const wp::float32 var_76 = 1e-08;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<4, wp::float32> var_79;
        //---------
        // forward
        // def adam_quaternion_step(                                                              <L 909>
        // i = wp.tid()                                                                           <L 922>
        var_0 = builtin_tid1d();
        // g = wp.vec4(gradients[i * 4], gradients[i * 4 + 1], gradients[i * 4 + 2], gradients[i * 4 + 3])       <L 923>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_gradients, var_2);
        var_5 = wp::mul(var_0, var_4);
        var_7 = wp::add(var_5, var_6);
        var_8 = wp::address(var_gradients, var_7);
        var_10 = wp::mul(var_0, var_9);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_gradients, var_12);
        var_15 = wp::mul(var_0, var_14);
        var_17 = wp::add(var_15, var_16);
        var_18 = wp::address(var_gradients, var_17);
        var_20 = wp::load(var_3);
        var_21 = wp::load(var_8);
        var_22 = wp::load(var_13);
        var_23 = wp::load(var_18);
        var_19 = wp::vec_t<4, wp::float32>(var_20, var_21, var_22, var_23);
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g                                                <L 924>
        var_24 = wp::address(var_m, var_0);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_beta1, var_26);
        var_28 = wp::sub(var_27, var_beta1);
        var_29 = wp::mul(var_28, var_19);
        var_30 = wp::add(var_25, var_29);
        wp::array_store(var_m, var_0, var_30);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g, g)                                  <L 925>
        var_31 = wp::address(var_v, var_0);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_beta2, var_33);
        var_35 = wp::sub(var_34, var_beta2);
        var_36 = wp::cw_mul(var_19, var_19);
        var_37 = wp::mul(var_35, var_36);
        var_38 = wp::add(var_32, var_37);
        wp::array_store(var_v, var_0, var_38);
        // mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))                                           <L 926>
        var_39 = wp::address(var_m, var_0);
        var_42 = wp::add(var_t, var_41);
        var_43 = wp::pow(var_beta1, var_42);
        var_44 = wp::sub(var_40, var_43);
        var_46 = wp::load(var_39);
        var_45 = wp::div(var_46, var_44);
        // vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))                                           <L 927>
        var_47 = wp::address(var_v, var_0);
        var_50 = wp::add(var_t, var_49);
        var_51 = wp::pow(var_beta2, var_50);
        var_52 = wp::sub(var_48, var_51);
        var_54 = wp::load(var_47);
        var_53 = wp::div(var_54, var_52);
        // sqrt_vhat = wp.vec4(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3]))       <L 928>
        var_56 = wp::extract(var_53, var_55);
        var_57 = wp::sqrt(var_56);
        var_59 = wp::extract(var_53, var_58);
        var_60 = wp::sqrt(var_59);
        var_62 = wp::extract(var_53, var_61);
        var_63 = wp::sqrt(var_62);
        var_65 = wp::extract(var_53, var_64);
        var_66 = wp::sqrt(var_65);
        var_67 = wp::vec_t<4, wp::float32>(var_57, var_60, var_63, var_66);
        // eps_vec4 = wp.vec4(eps, eps, eps, eps)                                                 <L 929>
        var_68 = wp::vec_t<4, wp::float32>(var_eps, var_eps, var_eps, var_eps);
        // q = quaternions[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec4))                      <L 930>
        var_69 = wp::address(var_quaternions, var_0);
        var_70 = wp::add(var_67, var_68);
        var_71 = wp::cw_div(var_45, var_70);
        var_72 = wp::mul(var_lr, var_71);
        var_74 = wp::load(var_69);
        var_73 = wp::sub(var_74, var_72);
        // quaternions[i] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)                                    <L 931>
        var_75 = wp::dot(var_73, var_73);
        var_77 = wp::add(var_75, var_76);
        var_78 = wp::sqrt(var_77);
        var_79 = wp::div(var_73, var_78);
        wp::array_store(var_quaternions, var_0, var_79);
    }
}



extern "C" __global__ void adam_quaternion_step_fb5bfef0_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions,
    wp::array_t<wp::float32> var_gradients,
    wp::array_t<wp::vec_t<4, wp::float32>> var_m,
    wp::array_t<wp::vec_t<4, wp::float32>> var_v,
    wp::float32 var_lr,
    wp::float32 var_beta1,
    wp::float32 var_beta2,
    wp::float32 var_t,
    wp::float32 var_eps,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_quaternions,
    wp::array_t<wp::float32> adj_gradients,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_m,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_v,
    wp::float32 adj_lr,
    wp::float32 adj_beta1,
    wp::float32 adj_beta2,
    wp::float32 adj_t,
    wp::float32 adj_eps)
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
        const wp::int32 var_1 = 4;
        wp::int32 var_2;
        wp::float32* var_3;
        const wp::int32 var_4 = 4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32 var_7;
        wp::float32* var_8;
        const wp::int32 var_9 = 4;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        wp::int32 var_12;
        wp::float32* var_13;
        const wp::int32 var_14 = 4;
        wp::int32 var_15;
        const wp::int32 var_16 = 3;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::vec_t<4, wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<4, wp::float32>* var_24;
        wp::vec_t<4, wp::float32> var_25;
        wp::vec_t<4, wp::float32> var_26;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::vec_t<4, wp::float32> var_29;
        wp::vec_t<4, wp::float32> var_30;
        wp::vec_t<4, wp::float32>* var_31;
        wp::vec_t<4, wp::float32> var_32;
        wp::vec_t<4, wp::float32> var_33;
        const wp::float32 var_34 = 1.0;
        wp::float32 var_35;
        wp::vec_t<4, wp::float32> var_36;
        wp::vec_t<4, wp::float32> var_37;
        wp::vec_t<4, wp::float32> var_38;
        wp::vec_t<4, wp::float32>* var_39;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<4, wp::float32> var_45;
        wp::vec_t<4, wp::float32> var_46;
        wp::vec_t<4, wp::float32>* var_47;
        const wp::float32 var_48 = 1.0;
        const wp::float32 var_49 = 1.0;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::vec_t<4, wp::float32> var_53;
        wp::vec_t<4, wp::float32> var_54;
        const wp::int32 var_55 = 0;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 1;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 2;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 3;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::vec_t<4, wp::float32> var_67;
        wp::vec_t<4, wp::float32> var_68;
        wp::vec_t<4, wp::float32>* var_69;
        wp::vec_t<4, wp::float32> var_70;
        wp::vec_t<4, wp::float32> var_71;
        wp::vec_t<4, wp::float32> var_72;
        wp::vec_t<4, wp::float32> var_73;
        wp::vec_t<4, wp::float32> var_74;
        wp::float32 var_75;
        const wp::float32 var_76 = 1e-08;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<4, wp::float32> var_79;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<4, wp::float32> adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<4, wp::float32> adj_24 = {};
        wp::vec_t<4, wp::float32> adj_25 = {};
        wp::vec_t<4, wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<4, wp::float32> adj_29 = {};
        wp::vec_t<4, wp::float32> adj_30 = {};
        wp::vec_t<4, wp::float32> adj_31 = {};
        wp::vec_t<4, wp::float32> adj_32 = {};
        wp::vec_t<4, wp::float32> adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::vec_t<4, wp::float32> adj_36 = {};
        wp::vec_t<4, wp::float32> adj_37 = {};
        wp::vec_t<4, wp::float32> adj_38 = {};
        wp::vec_t<4, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::vec_t<4, wp::float32> adj_45 = {};
        wp::vec_t<4, wp::float32> adj_46 = {};
        wp::vec_t<4, wp::float32> adj_47 = {};
        wp::float32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::vec_t<4, wp::float32> adj_53 = {};
        wp::vec_t<4, wp::float32> adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::vec_t<4, wp::float32> adj_67 = {};
        wp::vec_t<4, wp::float32> adj_68 = {};
        wp::vec_t<4, wp::float32> adj_69 = {};
        wp::vec_t<4, wp::float32> adj_70 = {};
        wp::vec_t<4, wp::float32> adj_71 = {};
        wp::vec_t<4, wp::float32> adj_72 = {};
        wp::vec_t<4, wp::float32> adj_73 = {};
        wp::vec_t<4, wp::float32> adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<4, wp::float32> adj_79 = {};
        //---------
        // forward
        // def adam_quaternion_step(                                                              <L 909>
        // i = wp.tid()                                                                           <L 922>
        var_0 = builtin_tid1d();
        // g = wp.vec4(gradients[i * 4], gradients[i * 4 + 1], gradients[i * 4 + 2], gradients[i * 4 + 3])       <L 923>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_gradients, var_2);
        var_5 = wp::mul(var_0, var_4);
        var_7 = wp::add(var_5, var_6);
        var_8 = wp::address(var_gradients, var_7);
        var_10 = wp::mul(var_0, var_9);
        var_12 = wp::add(var_10, var_11);
        var_13 = wp::address(var_gradients, var_12);
        var_15 = wp::mul(var_0, var_14);
        var_17 = wp::add(var_15, var_16);
        var_18 = wp::address(var_gradients, var_17);
        var_20 = wp::load(var_3);
        var_21 = wp::load(var_8);
        var_22 = wp::load(var_13);
        var_23 = wp::load(var_18);
        var_19 = wp::vec_t<4, wp::float32>(var_20, var_21, var_22, var_23);
        // m[i] = beta1 * m[i] + (1.0 - beta1) * g                                                <L 924>
        var_24 = wp::address(var_m, var_0);
        var_26 = wp::load(var_24);
        var_25 = wp::mul(var_beta1, var_26);
        var_28 = wp::sub(var_27, var_beta1);
        var_29 = wp::mul(var_28, var_19);
        var_30 = wp::add(var_25, var_29);
        // wp::array_store(var_m, var_0, var_30);
        // v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g, g)                                  <L 925>
        var_31 = wp::address(var_v, var_0);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_beta2, var_33);
        var_35 = wp::sub(var_34, var_beta2);
        var_36 = wp::cw_mul(var_19, var_19);
        var_37 = wp::mul(var_35, var_36);
        var_38 = wp::add(var_32, var_37);
        // wp::array_store(var_v, var_0, var_38);
        // mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))                                           <L 926>
        var_39 = wp::address(var_m, var_0);
        var_42 = wp::add(var_t, var_41);
        var_43 = wp::pow(var_beta1, var_42);
        var_44 = wp::sub(var_40, var_43);
        var_46 = wp::load(var_39);
        var_45 = wp::div(var_46, var_44);
        // vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))                                           <L 927>
        var_47 = wp::address(var_v, var_0);
        var_50 = wp::add(var_t, var_49);
        var_51 = wp::pow(var_beta2, var_50);
        var_52 = wp::sub(var_48, var_51);
        var_54 = wp::load(var_47);
        var_53 = wp::div(var_54, var_52);
        // sqrt_vhat = wp.vec4(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3]))       <L 928>
        var_56 = wp::extract(var_53, var_55);
        var_57 = wp::sqrt(var_56);
        var_59 = wp::extract(var_53, var_58);
        var_60 = wp::sqrt(var_59);
        var_62 = wp::extract(var_53, var_61);
        var_63 = wp::sqrt(var_62);
        var_65 = wp::extract(var_53, var_64);
        var_66 = wp::sqrt(var_65);
        var_67 = wp::vec_t<4, wp::float32>(var_57, var_60, var_63, var_66);
        // eps_vec4 = wp.vec4(eps, eps, eps, eps)                                                 <L 929>
        var_68 = wp::vec_t<4, wp::float32>(var_eps, var_eps, var_eps, var_eps);
        // q = quaternions[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec4))                      <L 930>
        var_69 = wp::address(var_quaternions, var_0);
        var_70 = wp::add(var_67, var_68);
        var_71 = wp::cw_div(var_45, var_70);
        var_72 = wp::mul(var_lr, var_71);
        var_74 = wp::load(var_69);
        var_73 = wp::sub(var_74, var_72);
        // quaternions[i] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)                                    <L 931>
        var_75 = wp::dot(var_73, var_73);
        var_77 = wp::add(var_75, var_76);
        var_78 = wp::sqrt(var_77);
        var_79 = wp::div(var_73, var_78);
        // wp::array_store(var_quaternions, var_0, var_79);
        //---------
        // reverse
        wp::adj_array_store(var_quaternions, var_0, var_79, adj_quaternions, adj_0, adj_79);
        wp::adj_div(var_73, var_78, adj_73, adj_78, adj_79);
        wp::adj_sqrt(var_77, var_78, adj_77, adj_78);
        wp::adj_add(var_75, var_76, adj_75, adj_76, adj_77);
        wp::adj_dot(var_73, var_73, adj_73, adj_73, adj_75);
        // adj: quaternions[i] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)                               <L 931>
        wp::adj_sub(var_74, var_72, adj_69, adj_72, adj_73);
        wp::adj_mul(var_lr, var_71, adj_lr, adj_71, adj_72);
        wp::adj_cw_div(var_45, var_70, var_71, adj_45, adj_70, adj_71);
        wp::adj_add(var_67, var_68, adj_67, adj_68, adj_70);
        wp::adj_address(var_quaternions, var_0, adj_quaternions, adj_0, adj_69);
        // adj: q = quaternions[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec4))                 <L 930>
        wp::adj_vec_t(var_eps, var_eps, var_eps, var_eps, adj_eps, adj_eps, adj_eps, adj_eps, adj_68);
        // adj: eps_vec4 = wp.vec4(eps, eps, eps, eps)                                            <L 929>
        wp::adj_vec_t(var_57, var_60, var_63, var_66, adj_57, adj_60, adj_63, adj_66, adj_67);
        wp::adj_sqrt(var_65, var_66, adj_65, adj_66);
        wp::adj_extract(var_53, var_64, adj_53, adj_64, adj_65);
        wp::adj_sqrt(var_62, var_63, adj_62, adj_63);
        wp::adj_extract(var_53, var_61, adj_53, adj_61, adj_62);
        wp::adj_sqrt(var_59, var_60, adj_59, adj_60);
        wp::adj_extract(var_53, var_58, adj_53, adj_58, adj_59);
        wp::adj_sqrt(var_56, var_57, adj_56, adj_57);
        wp::adj_extract(var_53, var_55, adj_53, adj_55, adj_56);
        // adj: sqrt_vhat = wp.vec4(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3]))  <L 928>
        wp::adj_div(var_54, var_52, adj_47, adj_52, adj_53);
        wp::adj_sub(var_48, var_51, adj_48, adj_51, adj_52);
        wp::adj_pow(var_beta2, var_50, var_51, adj_beta2, adj_50, adj_51);
        wp::adj_add(var_t, var_49, adj_t, adj_49, adj_50);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_47);
        // adj: vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))                                      <L 927>
        wp::adj_div(var_46, var_44, adj_39, adj_44, adj_45);
        wp::adj_sub(var_40, var_43, adj_40, adj_43, adj_44);
        wp::adj_pow(var_beta1, var_42, var_43, adj_beta1, adj_42, adj_43);
        wp::adj_add(var_t, var_41, adj_t, adj_41, adj_42);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_39);
        // adj: mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))                                      <L 926>
        wp::adj_array_store(var_v, var_0, var_38, adj_v, adj_0, adj_38);
        wp::adj_add(var_32, var_37, adj_32, adj_37, adj_38);
        wp::adj_mul(var_35, var_36, adj_35, adj_36, adj_37);
        wp::adj_cw_mul(var_19, var_19, adj_19, adj_19, adj_36);
        wp::adj_sub(var_34, var_beta2, adj_34, adj_beta2, adj_35);
        wp::adj_mul(var_beta2, var_33, adj_beta2, adj_31, adj_32);
        wp::adj_address(var_v, var_0, adj_v, adj_0, adj_31);
        // adj: v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g, g)                             <L 925>
        wp::adj_array_store(var_m, var_0, var_30, adj_m, adj_0, adj_30);
        wp::adj_add(var_25, var_29, adj_25, adj_29, adj_30);
        wp::adj_mul(var_28, var_19, adj_28, adj_19, adj_29);
        wp::adj_sub(var_27, var_beta1, adj_27, adj_beta1, adj_28);
        wp::adj_mul(var_beta1, var_26, adj_beta1, adj_24, adj_25);
        wp::adj_address(var_m, var_0, adj_m, adj_0, adj_24);
        // adj: m[i] = beta1 * m[i] + (1.0 - beta1) * g                                           <L 924>
        wp::adj_vec_t(var_20, var_21, var_22, var_23, adj_3, adj_8, adj_13, adj_18, adj_19);
        wp::adj_address(var_gradients, var_17, adj_gradients, adj_17, adj_18);
        wp::adj_add(var_15, var_16, adj_15, adj_16, adj_17);
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        wp::adj_address(var_gradients, var_12, adj_gradients, adj_12, adj_13);
        wp::adj_add(var_10, var_11, adj_10, adj_11, adj_12);
        wp::adj_mul(var_0, var_9, adj_0, adj_9, adj_10);
        wp::adj_address(var_gradients, var_7, adj_gradients, adj_7, adj_8);
        wp::adj_add(var_5, var_6, adj_5, adj_6, adj_7);
        wp::adj_mul(var_0, var_4, adj_0, adj_4, adj_5);
        wp::adj_address(var_gradients, var_2, adj_gradients, adj_2, adj_3);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: g = wp.vec4(gradients[i * 4], gradients[i * 4 + 1], gradients[i * 4 + 2], gradients[i * 4 + 3])  <L 923>
        // adj: i = wp.tid()                                                                      <L 922>
        // adj: def adam_quaternion_step(                                                         <L 909>
        continue;
    }
}

