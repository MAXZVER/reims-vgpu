#version 450
// TEMP (lab, PRESENT_SS, Mish 2026-10-22): the supersampled present's window pass. The upscaler (EASU on the
// moving frames, DLSS on the settled ones) writes `mid` larger than the visible part - 6912x2916 for a 3456x1458
// guest, its exact 2x, on the 5120x2160 panel - and this draw resamples it down to the visible part: a Lanczos-2
// kernel widened by the scale, so every `mid` texel weighs in (no aliasing, ~1.35x supersampling per axis), the
// result clamped to the range of the texels under the kernel's main lobe (no ringing around glyphs).
layout(set = 0, binding = 0) uniform sampler2D src;
// origin: the visible part's top-left in the window; size: its extent; sharp: unused (the layout RCAS shares).
layout(push_constant) uniform Push { ivec2 origin; ivec2 size; float sharp; } pc;
layout(location = 0) out vec4 color;

const int MAX_TAPS = 10;

float lanczos2(float x) {
    x = abs(x);
    if (x >= 2.0) {
        return 0.0;
    }
    if (x < 1.0e-4) {
        return 1.0;
    }
    float px = 3.14159265 * x;
    return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}

void main() {
    ivec2 mid = textureSize(src, 0);
    vec2 scale = vec2(mid) / vec2(pc.size);
    // The pixel's centre in `mid` texels (texel i's centre is at i + 0.5).
    vec2 c = (gl_FragCoord.xy - vec2(pc.origin)) * scale;
    // Texels whose centres lie within two kernel units (2 * scale texels) of c.
    ivec2 lo = ivec2(floor(c - 2.0 * scale - 0.5)) + 1;
    ivec2 n = min(ivec2(ceil(c + 2.0 * scale - 0.5)) - lo, ivec2(MAX_TAPS));
    float wx[MAX_TAPS];
    float wy[MAX_TAPS];
    for (int i = 0; i < n.x; i++) {
        wx[i] = lanczos2((float(lo.x + i) + 0.5 - c.x) / scale.x);
    }
    for (int j = 0; j < n.y; j++) {
        wy[j] = lanczos2((float(lo.y + j) + 0.5 - c.y) / scale.y);
    }
    vec3 acc = vec3(0.0);
    float sum = 0.0;
    vec3 mn = vec3(1.0);
    vec3 mx = vec3(0.0);
    for (int j = 0; j < n.y; j++) {
        bool inner_y = abs(float(lo.y + j) + 0.5 - c.y) < scale.y;
        for (int i = 0; i < n.x; i++) {
            vec3 t = texelFetch(src, clamp(lo + ivec2(i, j), ivec2(0), mid - 1), 0).rgb;
            float w = wx[i] * wy[j];
            acc += w * t;
            sum += w;
            if (inner_y && abs(float(lo.x + i) + 0.5 - c.x) < scale.x) {
                mn = min(mn, t);
                mx = max(mx, t);
            }
        }
    }
    vec3 rgb = acc / max(sum, 1.0e-6);
    // The main lobe always holds at least the nearest texel (scale >= 1); a 1:1 draw is that texel.
    color = vec4(clamp(rgb, min(mn, mx), mx), 1.0);
}
