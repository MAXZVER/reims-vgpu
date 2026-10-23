#version 450
// TEMP (lab, PRESENT_SS, Mish 2026-10-22): the supersampled present's window pass. The upscaler (EASU on the
// moving frames, DLSS on the settled ones) writes `mid` larger than the visible part - 6912x2916 for a 3456x1458
// guest, its exact 2x, on the 5120x2160 panel - and this draw resamples it down to the visible part: a Lanczos-2
// kernel widened by the scale, so every `mid` texel weighs in (no aliasing, ~1.35x supersampling per axis), the
// result clamped to the range of the texels under the kernel's main lobe (no ringing around glyphs).
//
// b963b: the first version (weights in arrays indexed by a runtime tap count) cost ~5 ms a 5K present on the
// RTX 4060 - the arrays went to local memory. The taps now run over a constant 6x6 window (enough for the kernel up
// to 1.5x; wider scales are what PRESENT_SS caps away from this shader's quality range, and still get a normalised
// though truncated kernel), weights computed in registers per axis.
layout(set = 0, binding = 0) uniform sampler2D src;
// origin: the visible part's top-left in the window; size: its extent; sharp: unused (the layout RCAS shares).
layout(push_constant) uniform Push { ivec2 origin; ivec2 size; float sharp; } pc;
layout(location = 0) out vec4 color;

float lanczos2(float x) {
    x = abs(x);
    float px = 3.14159265 * max(x, 1.0e-4);
    return x < 2.0 ? 2.0 * sin(px) * sin(px * 0.5) / (px * px) : 0.0;
}

void main() {
    ivec2 mid = textureSize(src, 0);
    vec2 scale = vec2(mid) / vec2(pc.size);
    vec2 c = (gl_FragCoord.xy - vec2(pc.origin)) * scale;
    // Six texels an axis around c, starting where the kernel's support starts (centre-aligned for any scale <= 1.5).
    ivec2 lo = ivec2(floor(c - 2.5));
    vec3 acc = vec3(0.0);
    float sum = 0.0;
    vec3 mn = vec3(1.0);
    vec3 mx = vec3(0.0);
    for (int j = 0; j < 6; j++) {
        float dy = (float(lo.y + j) + 0.5 - c.y) / scale.y;
        float wy = lanczos2(dy);
        int y = clamp(lo.y + j, 0, mid.y - 1);
        for (int i = 0; i < 6; i++) {
            float dx = (float(lo.x + i) + 0.5 - c.x) / scale.x;
            float w = lanczos2(dx) * wy;
            vec3 t = texelFetch(src, ivec2(clamp(lo.x + i, 0, mid.x - 1), y), 0).rgb;
            acc += w * t;
            sum += w;
            if (abs(dx) < 1.0 && abs(dy) < 1.0) {
                mn = min(mn, t);
                mx = max(mx, t);
            }
        }
    }
    vec3 rgb = acc / max(sum, 1.0e-6);
    color = vec4(clamp(rgb, min(mn, mx), mx), 1.0);
}
