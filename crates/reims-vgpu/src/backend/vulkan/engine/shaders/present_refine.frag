#version 450
// TEMP (lab, REIMS_VGPU_LAB_PRESENT_VSR_TILES): the refined (VSR) copy of every cell that still holds it, drawn over
// the window pass's own result; the other cells keep what the pass drew.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

layout(binding = 0) uniform sampler2D cache;
layout(std430, binding = 1) readonly buffer Cells {
    uint cells[];
};

layout(push_constant) uniform Push {
    ivec2 origin;  // the visible part's top-left in the window
    int cell;      // cell side in window pixels
    int cols;      // cells per row
    int count;     // cells in all
} pc;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) - pc.origin;
    ivec2 c = p / pc.cell;
    if (cells[uint(pc.count + c.y * pc.cols + c.x)] == 0u) {
        discard;
    }
    color = vec4(texelFetch(cache, p, 0).rgb, 1.0);
}
