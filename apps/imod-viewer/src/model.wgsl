// WGSL shaders for IMOD 3D model rendering
// Phong lighting with transparency support and depth cue (fog)

struct Uniforms {
    mvp: mat4x4<f32>,
    model_rot: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    light_dir: vec3<f32>,
    _pad1: f32,
    // Material properties
    ambient_strength: f32,
    diffuse_strength: f32,
    specular_strength: f32,
    shininess: f32,
    // Fog / depth cue
    fog_near: f32,
    fog_far: f32,
    _pad2: f32,
    _pad3: f32,
    bg_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) alpha: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) alpha: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    // Rotate normal by model rotation (upper 3x3)
    let rotated_normal = (u.model_rot * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_normal = normalize(rotated_normal);
    out.world_position = (u.model_rot * vec4<f32>(in.position, 1.0)).xyz;
    out.color = in.color;
    out.alpha = in.alpha;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos - in.world_position);

    // Two-sided lighting: flip normal if facing away from light
    var normal = N;
    if (dot(normal, L) < 0.0) {
        normal = -normal;
    }

    // Ambient
    let ambient = u.ambient_strength;

    // Diffuse (Lambertian)
    let NdotL = max(dot(normal, L), 0.0);
    let diffuse = u.diffuse_strength * NdotL;

    // Specular (Blinn-Phong)
    let H = normalize(L + V);
    let NdotH = max(dot(normal, H), 0.0);
    let specular = u.specular_strength * pow(NdotH, u.shininess);

    let lighting = ambient + diffuse + specular;
    var color = in.color * lighting;
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Depth cue (fog): linear fog based on distance from camera
    let dist = length(in.world_position - u.camera_pos);
    let fog_range = u.fog_far - u.fog_near;
    var fog_factor = 0.0;
    if (fog_range > 0.001) {
        fog_factor = clamp((dist - u.fog_near) / fog_range, 0.0, 1.0);
    }
    color = mix(color, u.bg_color.rgb, fog_factor * 0.5);

    return vec4<f32>(color, in.alpha);
}
