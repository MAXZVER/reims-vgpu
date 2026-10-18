#version 450
// Edge-adaptive upscale of the guest frame into the host window: the EASU pass of
// AMD FidelityFX Super Resolution 1.0 (MIT licence, Copyright (c) 2021 Advanced
// Micro Devices, Inc.), restated in GLSL for this one draw.
//
// Mish 2026-10-18: a guest mode 15-20 % below the 5K panel (4352x1836,
// 4096x1728) made text and icons step on their slanted edges under the
// Catmull-Rom resample this replaced, which is separable and so cannot follow an
// edge. EASU reads a 12-texel neighbourhood, estimates the local edge direction
// and how strongly it is an edge, and filters with a Lanczos-like kernel
// stretched along that direction; the result is clamped to the 2x2 texels
// around the sample, so it adds no ringing.
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform Push { vec2 size; } pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

vec3 fetch(ivec2 p) {
    ivec2 hi = ivec2(pc.size) - 1;
    return texelFetch(src, clamp(p, ivec2(0), hi), 0).rgb;
}

float luma(vec3 c) {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// One of the four centre texels' votes for the edge direction and length, weighted
// by its bilinear share w. l: the texel (c) and its up (a), left (b), right (d)
// and down (e) neighbours.
void easuSet(inout vec2 dir, inout float len, float w, float la, float lb, float lc, float ld, float le) {
    float dc = ld - lc;
    float cb = lc - lb;
    float lenX = max(abs(dc), abs(cb));
    lenX = 1.0 / max(lenX, 1.0 / 65536.0);
    float dirX = ld - lb;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;

    float ec = le - lc;
    float ca = lc - la;
    float lenY = max(abs(ec), abs(ca));
    lenY = 1.0 / max(lenY, 1.0 / 65536.0);
    float dirY = le - la;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

void easuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len2, float lob, float clp, vec3 c) {
    vec2 v = vec2(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x);
    v *= len2;
    float d2 = min(dot(v, v), clp);
    float wB = 0.4 * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 1.5625 * wB - 0.5625;
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void main() {
    vec2 pp = uv * pc.size - 0.5;
    vec2 fp = floor(pp);
    pp -= fp;
    ivec2 p = ivec2(fp);

    //    b c
    //  e f g h
    //  i j k l
    //    n o
    vec3 b = fetch(p + ivec2(0, -1)), c = fetch(p + ivec2(1, -1));
    vec3 e = fetch(p + ivec2(-1, 0)), f = fetch(p), g = fetch(p + ivec2(1, 0)), h = fetch(p + ivec2(2, 0));
    vec3 i = fetch(p + ivec2(-1, 1)), j = fetch(p + ivec2(0, 1)), k = fetch(p + ivec2(1, 1)), l = fetch(p + ivec2(2, 1));
    vec3 n = fetch(p + ivec2(0, 2)), o = fetch(p + ivec2(1, 2));

    float bL = luma(b), cL = luma(c), eL = luma(e), fL = luma(f), gL = luma(g), hL = luma(h);
    float iL = luma(i), jL = luma(j), kL = luma(k), lL = luma(l), nL = luma(n), oL = luma(o);

    vec2 dir = vec2(0.0);
    float len = 0.0;
    easuSet(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
    easuSet(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
    easuSet(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);
    easuSet(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < 1.0 / 32768.0;
    dirR = inversesqrt(max(dirR, 1.0 / 32768.0));
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    len = len * 0.5;
    len *= len;
    float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = 1.0 / lob;

    vec3 aC = vec3(0.0);
    float aW = 0.0;
    easuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b);
    easuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c);
    easuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i);
    easuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j);
    easuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f);
    easuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e);
    easuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k);
    easuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l);
    easuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h);
    easuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g);
    easuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, o);
    easuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n);

    vec3 mn = min(min(f, g), min(j, k));
    vec3 mx = max(max(f, g), max(j, k));
    vec3 rgb = clamp(aC / max(aW, 1.0 / 65536.0), mn, mx);
    color = vec4(rgb, 1.0);
}
