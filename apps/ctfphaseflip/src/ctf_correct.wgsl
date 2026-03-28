// CTF phase-flip correction compute shader.
// Each invocation handles one (fx, fy) frequency pixel in a strip,
// computing the CTF and flipping the phase (negating the complex value)
// where the CTF is negative.

struct Params {
    sw: u32,              // strip width in pixels (frequency domain)
    sh: u32,              // strip height in pixels (frequency domain)
    pixel_a: f32,         // pixel size in Angstroms
    wavelength: f32,      // electron wavelength in Angstroms
    cs_a: f32,            // Cs in Angstroms
    amp_contrast: f32,    // amplitude contrast fraction (w)
    cuton_freq: f32,      // minimum spatial frequency for correction
    defocus1: f32,        // strip defocus1 in Angstroms
    defocus2: f32,        // strip defocus2 in Angstroms
    astig_angle_rad: f32, // astigmatism angle in radians
    has_astigmatism: u32, // 1 if astigmatism present
    plate_phase_rad: f32, // phase plate constant phase shift in radians
    tilt_axis_rad: f32,   // tilt axis angle in radians for frequency rotation
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> strip_re: array<f32>;
@group(0) @binding(1) var<storage, read_write> strip_im: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(16, 16)
fn ctf_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fx_idx = gid.x;
    let fy_idx = gid.y;

    if fx_idx >= params.sw || fy_idx >= params.sh {
        return;
    }

    let sw = f32(params.sw);
    let sh = f32(params.sh);

    // Compute frequency coordinates
    var freq_x: f32;
    if fx_idx <= params.sw / 2u {
        freq_x = f32(fx_idx);
    } else {
        freq_x = f32(fx_idx) - sw;
    }
    var freq_y: f32;
    if fy_idx <= params.sh / 2u {
        freq_y = f32(fy_idx);
    } else {
        freq_y = f32(fy_idx) - sh;
    }

    // Apply tilt-axis rotation to frequency coordinates
    if abs(params.tilt_axis_rad) > 1e-6 {
        let cos_ta = cos(params.tilt_axis_rad);
        let sin_ta = sin(params.tilt_axis_rad);
        let rx = freq_x * cos_ta + freq_y * sin_ta;
        let ry = -freq_x * sin_ta + freq_y * cos_ta;
        freq_x = rx;
        freq_y = ry;
    }

    let sx = freq_x / (sw * params.pixel_a);
    let sy = freq_y / (sh * params.pixel_a);

    let s2 = sx * sx + sy * sy;
    let s = sqrt(s2);

    // Skip correction below cuton frequency
    if s < params.cuton_freq {
        return;
    }

    // Compute effective defocus for this frequency direction
    var def_for_ctf = params.defocus1;
    if params.has_astigmatism == 1u {
        let angle = atan2(sy, sx);
        let da = angle - params.astig_angle_rad;
        let cos2 = cos(da) * cos(da);
        let sin2 = sin(da) * sin(da);
        def_for_ctf = params.defocus1 * cos2 + params.defocus2 * sin2;
    }

    // CTF phase: chi(s) = pi * lambda * s^2 * (defocus - 0.5 * Cs * lambda^2 * s^2)
    let wl = params.wavelength;
    let chi = PI * wl * s2 * (def_for_ctf - 0.5 * params.cs_a * wl * wl * s2);

    // Add phase plate constant phase shift
    let chi_total = chi + params.plate_phase_rad;

    // Full CTF with amplitude contrast:
    // CTF = -sin(chi)*sqrt(1-w^2) + cos(chi)*w
    let w = params.amp_contrast;
    let w_phase = sqrt(1.0 - w * w);
    let ctf = -sin(chi_total) * w_phase + cos(chi_total) * w;

    // Phase-flip: negate Fourier coefficient where CTF < 0
    if ctf < 0.0 {
        let idx = fy_idx * params.sw + fx_idx;
        strip_re[idx] = -strip_re[idx];
        strip_im[idx] = -strip_im[idx];
    }
}
