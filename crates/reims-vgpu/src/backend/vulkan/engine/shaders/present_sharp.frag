#version 450
// Catmull-Rom (bicubic, B=0 C=0.5) resampling of the guest frame into the host
// window, in nine bilinear taps: the 4x4 kernel's middle two weights per axis
// are folded into one linearly filtered fetch between the two texels.
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform Push { vec2 size; } pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    vec2 pos = uv * pc.size;
    vec2 tc = floor(pos - 0.5) + 0.5;
    vec2 f = pos - tc;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 o12 = w2 / w12;
    vec2 inv = 1.0 / pc.size;
    vec2 t0 = (tc - 1.0) * inv;
    vec2 t3 = (tc + 2.0) * inv;
    vec2 t12 = (tc + o12) * inv;
    vec4 c = texture(src, vec2(t0.x, t0.y)) * w0.x * w0.y
           + texture(src, vec2(t12.x, t0.y)) * w12.x * w0.y
           + texture(src, vec2(t3.x, t0.y)) * w3.x * w0.y
           + texture(src, vec2(t0.x, t12.y)) * w0.x * w12.y
           + texture(src, vec2(t12.x, t12.y)) * w12.x * w12.y
           + texture(src, vec2(t3.x, t12.y)) * w3.x * w12.y
           + texture(src, vec2(t0.x, t3.y)) * w0.x * w3.y
           + texture(src, vec2(t12.x, t3.y)) * w12.x * w3.y
           + texture(src, vec2(t3.x, t3.y)) * w3.x * w3.y;
    color = clamp(c, 0.0, 1.0);
}
