#version 450
// Sharpening after the edge-adaptive upscale: the RCAS pass of AMD FidelityFX
// Super Resolution 1.0 (MIT licence, Copyright (c) 2021 Advanced Micro Devices,
// Inc.), restated in GLSL for this one draw, without its optional denoise.
//
// Mish 2026-10-19: a 4096x1728 guest (everything 25 % larger) on the 5120x2160
// panel, "with every filter there is". EASU restores edge direction but, like
// any resample, leaves glyph stems a little softer than the source; RCAS puts
// the contrast back. It reads the EASU result (an image the size of the
// visible part) in a cross of five pixels and applies a negative lobe to the
// four neighbours, limited per channel so the result never leaves the
// neighbourhood's own range: no ringing, no halos around text.
layout(set = 0, binding = 0) uniform sampler2D src;
// origin: the visible part's top-left in the window; size: its extent (the
// EASU image's size); sharp: exp2(-stops), 1.0 = strongest.
layout(push_constant) uniform Push { ivec2 origin; ivec2 size; float sharp; } pc;
layout(location = 0) out vec4 color;

vec3 fetch(ivec2 p) {
    return texelFetch(src, clamp(p, ivec2(0), pc.size - 1), 0).rgb;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) - pc.origin;
    //   b
    // d e f
    //   h
    vec3 b = fetch(p + ivec2(0, -1));
    vec3 d = fetch(p + ivec2(-1, 0));
    vec3 e = fetch(p);
    vec3 f = fetch(p + ivec2(1, 0));
    vec3 h = fetch(p + ivec2(0, 1));

    vec3 mn4 = min(min(b, d), min(f, h));
    vec3 mx4 = max(max(b, d), max(f, h));
    // The strongest lobe that keeps e + lobe * (b + d + f + h - 4e) inside
    // [0, 1] for each channel, from below and from above.
    vec3 hitMin = mn4 / max(4.0 * mx4, vec3(1.0 / 65536.0));
    vec3 hitMax = (1.0 - mx4) / min(4.0 * mn4 - 4.0, vec3(-1.0 / 65536.0));
    vec3 lobeRGB = max(-hitMin, hitMax);
    // FSR_RCAS_LIMIT = 0.25 - 1/16.
    float lobe = max(-(0.25 - 1.0 / 16.0), min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * pc.sharp;
    vec3 rgb = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
