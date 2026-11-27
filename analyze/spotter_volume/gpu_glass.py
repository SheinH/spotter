"""Liquid Glass on the GPU: the maths of glass.py (refraction maps, frost, colour curves,
rim light, Fresnel) as three OpenGL passes.

    backdrop --blur x--> --blur y--> frost   (only when the backdrop changes)
    frost + backdrop + maps --composite--> card pixels   (also per animation frame)

The card is a Qt layered window, so the result is read back (a card is ~0.5 MB) and drawn
by QPainter. Every row is stored top row first in every texture, so gl_FragCoord.y counts down
the card and nothing needs flipping.
"""
from __future__ import annotations

import numpy as np

from .glass import GlassMaps, GlassParams, color_curves

_VERT = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // one triangle covering the target
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_BLUR = """
#version 330
uniform sampler2D src;
uniform vec2 size;
uniform vec2 dir;
uniform float sigma;
out vec4 color;
void main() {
    int r = int(ceil(sigma * 3.0));
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = exp(-0.5 * float(i * i) / (sigma * sigma));
        acc += w * texture(src, (gl_FragCoord.xy + dir * float(i)) / size).rgb;
        wsum += w;
    }
    color = vec4(acc / wsum, 1.0);
}
"""

_COMPOSITE = """
#version 330
uniform sampler2D sharp;     // backdrop, region around the card
uniform sampler2D frost;     // the same, blurred
uniform sampler2D maps;      // per card pixel: x shift, y shift, rim, fresnel (device px)
uniform sampler2D shape;     // card coverage 0..1
uniform vec2 bsize;          // backdrop size
uniform vec2 offset;         // card's top-left inside the backdrop
uniform float amount;        // 0 = not materialized yet, 1 = full glass
uniform float chroma;
uniform float luma_k[9];      // luma curve, knots at L = 0, 1/8 .. 1
uniform float chroma_k[9];    // chroma gain, same knots
uniform float specular;
uniform float fresnel;
out vec4 color;

vec3 pick(vec2 at) {
    vec2 uv = (at + 0.5) / bsize;
    return mix(texture(sharp, uv).rgb, texture(frost, uv).rgb, amount);
}

float curve(float k[9], float x) {
    float t = clamp(x, 0.0, 1.0) * 8.0;
    int i = min(int(t), 7);
    return mix(k[i], k[i + 1], t - float(i));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 m = texelFetch(maps, p, 0);
    float a = texelFetch(shape, p, 0).r;
    vec2 at = offset + vec2(p);
    vec2 shift = m.xy * amount;          // convex glass: rim pixels show backdrop from further in
    vec3 c = vec3(pick(at + shift * (1.0 + chroma)).r,
                  pick(at + shift).g,
                  pick(at + shift * (1.0 - chroma)).b);
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 g = clamp(vec3(curve(luma_k, luma)) + (c - vec3(luma)) * curve(chroma_k, luma), 0.0, 1.0);
    c = mix(c, g, amount);
    float light = (m.w * fresnel + m.z * specular) * amount;
    c = c + (1.0 - c) * light;           // screen-blend white
    color = vec4(c * a, a);
}
"""


class GpuGlass:
    """One renderer per card size. Raises on machines without OpenGL 3.3; the card then
    falls back to the solid material."""

    def __init__(self, maps: GlassMaps, scale: float, params: GlassParams = GlassParams()) -> None:
        import moderngl
        self.ctx = moderngl.create_standalone_context()
        self.p = params
        self.scale = scale
        self.h, self.w = maps.alpha.shape
        self._blur = self.ctx.program(vertex_shader=_VERT, fragment_shader=_BLUR)
        self._comp = self.ctx.program(vertex_shader=_VERT, fragment_shader=_COMPOSITE)
        self._vao_blur = self.ctx.vertex_array(self._blur, [])
        self._vao_comp = self.ctx.vertex_array(self._comp, [])

        packed = np.stack([maps.sx, maps.sy, maps.rim, maps.fres], -1).astype("f4")
        self._maps = self.ctx.texture((self.w, self.h), 4, np.ascontiguousarray(packed).tobytes(), dtype="f4")
        self._shape = self.ctx.texture((self.w, self.h), 1, np.ascontiguousarray(maps.alpha, "f4").tobytes(), dtype="f4")
        for t in (self._maps, self._shape):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._out = self.ctx.simple_framebuffer((self.w, self.h), components=4)
        self._bsize: tuple[int, int] | None = None
        self._offset = (0.0, 0.0)
        self.renders = 0                       # full (blur + composite) renders, for tests/logs

    def _alloc(self, bw: int, bh: int) -> None:
        import moderngl
        if self._bsize == (bw, bh):
            return
        self._sharp = self.ctx.texture((bw, bh), 3, alignment=1)
        self._tmp = self.ctx.texture((bw, bh), 4, dtype="f2")
        self._frost = self.ctx.texture((bw, bh), 4, dtype="f2")
        for t in (self._sharp, self._tmp, self._frost):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False    # clamp to edge, like scipy's mode="nearest"
        self._fb_tmp = self.ctx.framebuffer([self._tmp])
        self._fb_frost = self.ctx.framebuffer([self._frost])
        self._bsize = (bw, bh)

    def set_backdrop(self, rgb: np.ndarray, offset: tuple[float, float]) -> None:
        """rgb: (h, w, 3) uint8 of the card's surroundings; offset: the card's (x, y) in it."""
        bh, bw = rgb.shape[:2]
        self._alloc(bw, bh)
        self._sharp.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
        self._offset = (float(offset[0]), float(offset[1]))
        sigma = max(self.p.blur_px * self.scale, 0.5)
        b = self._blur
        b["size"].value = (bw, bh)
        b["sigma"].value = sigma
        for src, dst, d in ((self._sharp, self._fb_tmp, (1.0, 0.0)), (self._tmp, self._fb_frost, (0.0, 1.0))):
            dst.use()
            src.use(0)
            b["src"].value = 0
            b["dir"].value = d
            self._vao_blur.render(vertices=3)
        self.renders += 1

    def compose(self, amount: float, light: float, adapt: float = 0.5) -> np.ndarray:
        """light: appearance, 0 = dark glass .. 1 = light glass; adapt: mean luma 0..1 of the
        backdrop behind the card. (h, w, 4) uint8 premultiplied RGBA of the card (Qt Format_RGBA8888_Premultiplied)."""
        if self._bsize is None:
            raise RuntimeError("set_backdrop first")
        c = self._comp
        self._out.use()
        for unit, (name, tex) in enumerate((("sharp", self._sharp), ("frost", self._frost),
                                            ("maps", self._maps), ("shape", self._shape))):
            tex.use(unit)
            c[name].value = unit
        c["bsize"].value = self._bsize
        c["offset"].value = self._offset
        c["amount"].value = float(amount)
        c["chroma"].value = self.p.chroma
        luma_k, chroma_k = color_curves(light, adapt, self.p)
        c["luma_k"].value = luma_k
        c["chroma_k"].value = chroma_k
        c["specular"].value = self.p.specular
        c["fresnel"].value = self.p.fresnel
        self._vao_comp.render(vertices=3)
        buf = self._out.read(components=4, alignment=1)
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)

    @property
    def ready(self) -> bool:
        return self._bsize is not None

    def release(self) -> None:
        self.ctx.release()
