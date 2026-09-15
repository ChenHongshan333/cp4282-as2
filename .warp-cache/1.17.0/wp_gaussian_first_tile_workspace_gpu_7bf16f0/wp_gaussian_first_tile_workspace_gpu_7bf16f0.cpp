#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 1
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

#define builtin_block_dim() wp::block_dim()


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:346
static wp::vec_t<3, wp::float32> projected_covariance_0(
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


// D:\NUS CS\Y3 S1\CP4282\cp4282-as2\3dgs_trainer.py:346
static void adj_projected_covariance_0(
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
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}

struct wp_args_project_tile_records_595de004 {
    wp::array_t<wp::vec_t<3, wp::float32>> means;
    wp::array_t<wp::vec_t<3, wp::float32>> log_scales;
    wp::array_t<wp::vec_t<4, wp::float32>> quaternions;
    wp::array_t<wp::float32> opacity_logits;
    wp::array_t<wp::int32> active;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> cameras;
    wp::array_t<wp::int32> view_ids;
    wp::int32 capacity;
    wp::int32 width;
    wp::int32 height;
    wp::float32 focal;
    wp::int32 view_count;
    wp::int32 compact_box_enabled;
    wp::float32 compact_box_beta;
    wp::float32 compact_box_alpha_min;
    wp::array_t<wp::vec_t<2, wp::float32>> centres;
    wp::array_t<wp::vec_t<3, wp::float32>> conics;
    wp::array_t<wp::float32> supports;
    wp::array_t<wp::float32> depths;
    wp::array_t<wp::int32> group_ids;
    wp::array_t<wp::uint32> splat_ids;
};


void project_tile_records_595de004_cpu_kernel_forward(
    wp::launch_bounds_t<1> dim,
    size_t task_index,
    wp_args_project_tile_records_595de004 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_means = _wp_args->means;
    wp::array_t<wp::vec_t<3, wp::float32>> var_log_scales = _wp_args->log_scales;
    wp::array_t<wp::vec_t<4, wp::float32>> var_quaternions = _wp_args->quaternions;
    wp::array_t<wp::float32> var_opacity_logits = _wp_args->opacity_logits;
    wp::array_t<wp::int32> var_active = _wp_args->active;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_cameras = _wp_args->cameras;
    wp::array_t<wp::int32> var_view_ids = _wp_args->view_ids;
    wp::int32 var_capacity = _wp_args->capacity;
    wp::int32 var_width = _wp_args->width;
    wp::int32 var_height = _wp_args->height;
    wp::float32 var_focal = _wp_args->focal;
    wp::int32 var_view_count = _wp_args->view_count;
    wp::int32 var_compact_box_enabled = _wp_args->compact_box_enabled;
    wp::float32 var_compact_box_beta = _wp_args->compact_box_beta;
    wp::float32 var_compact_box_alpha_min = _wp_args->compact_box_alpha_min;
    wp::array_t<wp::vec_t<2, wp::float32>> var_centres = _wp_args->centres;
    wp::array_t<wp::vec_t<3, wp::float32>> var_conics = _wp_args->conics;
    wp::array_t<wp::float32> var_supports = _wp_args->supports;
    wp::array_t<wp::float32> var_depths = _wp_args->depths;
    wp::array_t<wp::int32> var_group_ids = _wp_args->group_ids;
    wp::array_t<wp::uint32> var_splat_ids = _wp_args->splat_ids;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<2, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    wp::uint32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    wp::mat_t<4, 4, wp::float32>* var_21;
    wp::int32 var_22;
    wp::mat_t<4, 4, wp::float32> var_23;
    wp::mat_t<4, 4, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32>* var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32>* var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::float32 var_37 = 1.0;
    wp::vec_t<4, wp::float32> var_38;
    wp::vec_t<4, wp::float32> var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.1;
    bool var_43;
    wp::vec_t<3, wp::float32>* var_44;
    wp::vec_t<4, wp::float32>* var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<4, wp::float32> var_52;
    const wp::float32 var_53 = 9.0;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    bool var_56;
    const wp::float32 var_57 = 1.0;
    const wp::float32 var_58 = 1.0;
    wp::float32* var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.0;
    bool var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    const wp::int32 var_78 = 2;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    bool var_87;
    const wp::float32 var_88 = 0.0;
    bool var_89;
    const wp::float32 var_90 = 1e-12;
    bool var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::int32 var_99 = 0;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::vec_t<3, wp::float32> var_102;
    const wp::int32 var_103 = 0;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    const wp::float32 var_107 = 0.5;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::float32 var_115 = 0.5;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::vec_t<2, wp::float32> var_119;
    bool var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    bool var_123;
    const wp::int32 var_124 = 1;
    wp::float32 var_125;
    bool var_126;
    const wp::int32 var_127 = 0;
    wp::float32 var_128;
    bool var_129;
    const wp::int32 var_130 = 1;
    wp::float32 var_131;
    bool var_132;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    bool var_135;
    //---------
    // forward
    // def project_tile_records(                                                              <L 37>
    // item = wp.tid()                                                                        <L 60>
    var_0 = builtin_tid1d();
    // batch_view = item // capacity                                                          <L 61>
    var_1 = wp::floordiv(var_0, var_capacity);
    // splat = item - batch_view * capacity                                                   <L 62>
    var_2 = wp::mul(var_1, var_capacity);
    var_3 = wp::sub(var_0, var_2);
    // centres[item] = wp.vec2(0.0, 0.0)                                                      <L 63>
    var_6 = wp::vec_t<2, wp::float32>(var_4, var_5);
    wp::array_store(var_centres, var_0, var_6);
    // conics[item] = wp.vec3(0.0, 0.0, 0.0)                                                  <L 64>
    var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
    wp::array_store(var_conics, var_0, var_10);
    // supports[item] = 0.0                                                                   <L 65>
    wp::array_store(var_supports, var_0, var_11);
    // depths[item] = 0.0                                                                     <L 66>
    wp::array_store(var_depths, var_0, var_12);
    // group_ids[item] = batch_view                                                           <L 67>
    wp::array_store(var_group_ids, var_0, var_1);
    // splat_ids[item] = wp.uint32(splat)                                                     <L 68>
    var_13 = wp::uint32(var_3);
    wp::array_store(var_splat_ids, var_0, var_13);
    // if batch_view < view_count and active[splat] != 0:                                     <L 70>
    var_15 = (var_1 < var_view_count);
    var_14 = var_15;
    if (var_14) {
        var_16 = wp::address(var_active, var_3);
        var_19 = wp::load(var_16);
        var_18 = (var_19 != var_17);
        var_14 = var_14 && var_18;
    }
    if (var_14) {
        // camera = cameras[view_ids[batch_view]]                                             <L 71>
        var_20 = wp::address(var_view_ids, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_cameras, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::copy(var_24);
        // point = camera * wp.vec4(means[splat][0], means[splat][1], means[splat][2], 1.0)       <L 72>
        var_25 = wp::address(var_means, var_3);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::address(var_means, var_3);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        var_33 = wp::address(var_means, var_3);
        var_36 = wp::load(var_33);
        var_35 = wp::extract(var_36, var_34);
        var_38 = wp::vec_t<4, wp::float32>(var_27, var_31, var_35, var_37);
        var_39 = wp::mul(var_23, var_38);
        // z = point[2]                                                                       <L 73>
        var_41 = wp::extract(var_39, var_40);
        // if z > NEAR_PLANE:                                                                 <L 74>
        var_43 = (var_41 > var_42);
        if (var_43) {
            // covariance = projected_covariance(                                             <L 75>
            // log_scales[splat], quaternions[splat], camera,                                 <L 76>
            var_44 = wp::address(var_log_scales, var_3);
            var_45 = wp::address(var_quaternions, var_3);
            // point[0], point[1], z, focal,                                                  <L 77>
            var_47 = wp::extract(var_39, var_46);
            var_49 = wp::extract(var_39, var_48);
            var_51 = wp::load(var_44);
            var_52 = wp::load(var_45);
            var_50 = projected_covariance_0(var_51, var_52, var_23, var_47, var_49, var_41, var_focal);
            // support = float(SUPPORT_RADIUS_SQUARED)                                        <L 79>
            var_54 = wp::float(var_53);
            // if compact_box_enabled != 0:                                                   <L 80>
            var_56 = (var_compact_box_enabled != var_55);
            if (var_56) {
                // opacity = 1.0 / (1.0 + wp.exp(-opacity_logits[splat]))                     <L 81>
                var_59 = wp::address(var_opacity_logits, var_3);
                var_61 = wp::load(var_59);
                var_60 = wp::neg(var_61);
                var_62 = wp::exp(var_60);
                var_63 = wp::add(var_58, var_62);
                var_64 = wp::div(var_57, var_63);
                // support = 0.0                                                              <L 82>
                // if opacity > compact_box_alpha_min:                                        <L 83>
                var_66 = (var_64 > var_compact_box_alpha_min);
                if (var_66) {
                    // support = wp.min(                                                      <L 84>
                    // float(SUPPORT_RADIUS_SQUARED),                                         <L 85>
                    var_67 = wp::float(var_53);
                    // compact_box_beta * 2.0 * wp.log(opacity / compact_box_alpha_min),       <L 86>
                    var_69 = wp::mul(var_compact_box_beta, var_68);
                    var_70 = wp::div(var_64, var_compact_box_alpha_min);
                    var_71 = wp::log(var_70);
                    var_72 = wp::mul(var_69, var_71);
                    var_73 = wp::min(var_67, var_72);
                }
                var_74 = wp::where(var_66, var_73, var_65);
            }
            var_75 = wp::where(var_56, var_74, var_54);
            // determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1]       <L 89>
            var_77 = wp::extract(var_50, var_76);
            var_79 = wp::extract(var_50, var_78);
            var_80 = wp::mul(var_77, var_79);
            var_82 = wp::extract(var_50, var_81);
            var_84 = wp::extract(var_50, var_83);
            var_85 = wp::mul(var_82, var_84);
            var_86 = wp::sub(var_80, var_85);
            // if support > 0.0 and determinant > 1.0e-12:                                    <L 90>
            var_89 = (var_75 > var_88);
            var_87 = var_89;
            if (var_87) {
                var_91 = (var_86 > var_90);
                var_87 = var_87 && var_91;
            }
            if (var_87) {
                // conic = wp.vec3(                                                           <L 91>
                // covariance[2] / determinant,                                               <L 92>
                var_93 = wp::extract(var_50, var_92);
                var_94 = wp::div(var_93, var_86);
                // -covariance[1] / determinant,                                              <L 93>
                var_96 = wp::extract(var_50, var_95);
                var_97 = wp::neg(var_96);
                var_98 = wp::div(var_97, var_86);
                // covariance[0] / determinant,                                               <L 94>
                var_100 = wp::extract(var_50, var_99);
                var_101 = wp::div(var_100, var_86);
                var_102 = wp::vec_t<3, wp::float32>(var_94, var_98, var_101);
                // centre = wp.vec2(                                                          <L 96>
                // focal * point[0] / z + 0.5 * float(width),                                 <L 97>
                var_104 = wp::extract(var_39, var_103);
                var_105 = wp::mul(var_focal, var_104);
                var_106 = wp::div(var_105, var_41);
                var_108 = wp::float(var_width);
                var_109 = wp::mul(var_107, var_108);
                var_110 = wp::add(var_106, var_109);
                // focal * point[1] / z + 0.5 * float(height),                                <L 98>
                var_112 = wp::extract(var_39, var_111);
                var_113 = wp::mul(var_focal, var_112);
                var_114 = wp::div(var_113, var_41);
                var_116 = wp::float(var_height);
                var_117 = wp::mul(var_115, var_116);
                var_118 = wp::add(var_114, var_117);
                var_119 = wp::vec_t<2, wp::float32>(var_110, var_118);
                // if (                                                                       <L 100>
                // wp.isfinite(centre[0])                                                     <L 101>
                var_122 = wp::extract(var_119, var_121);
                var_123 = wp::isfinite(var_122);
                var_120 = var_123;
                if (var_120) {
                    // and wp.isfinite(centre[1])                                             <L 102>
                    var_125 = wp::extract(var_119, var_124);
                    var_126 = wp::isfinite(var_125);
                    var_120 = var_120 && var_126;
                }
                if (var_120) {
                    // and wp.isfinite(conic[0])                                              <L 103>
                    var_128 = wp::extract(var_102, var_127);
                    var_129 = wp::isfinite(var_128);
                    var_120 = var_120 && var_129;
                }
                if (var_120) {
                    // and wp.isfinite(conic[1])                                              <L 104>
                    var_131 = wp::extract(var_102, var_130);
                    var_132 = wp::isfinite(var_131);
                    var_120 = var_120 && var_132;
                }
                if (var_120) {
                    // and wp.isfinite(conic[2])                                              <L 105>
                    var_134 = wp::extract(var_102, var_133);
                    var_135 = wp::isfinite(var_134);
                    var_120 = var_120 && var_135;
                }
                if (var_120) {
                    // centres[item] = centre                                                 <L 107>
                    wp::array_store(var_centres, var_0, var_119);
                    // conics[item] = conic                                                   <L 108>
                    wp::array_store(var_conics, var_0, var_102);
                    // supports[item] = support                                               <L 109>
                    wp::array_store(var_supports, var_0, var_75);
                    // depths[item] = z                                                       <L 110>
                    wp::array_store(var_depths, var_0, var_41);
                }
            }
        }
    }
}



extern "C" {

// Python CPU entry points
WP_API void project_tile_records_595de004_cpu_forward(
    wp::launch_bounds_t<1> *dim,
    wp_args_project_tile_records_595de004 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        project_tile_records_595de004_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C

