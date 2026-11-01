"use client";

/**
 * The cloud. Four additive layers share one set of coordinates, all driven by the local
 * density computed client-side in lib/spatial.ts:
 *
 *  - core      small discs; dense regions whiten and twinkle, sparse ones keep the hue
 *  - halo      wide soft discs whose strength grows with density: the glow of the clusters
 *  - dust      one jittered ghost per point, spread wider where the cloud is sparse: haze
 *  - filaments 1-px lines to each point's nearest neighbours: the web between clusters
 *
 * A bloom pass blurs whatever exceeds the luminance threshold, which with additive blending
 * is exactly the cluster cores. Visibility is a per-vertex attribute rewritten when a
 * country is toggled; geometry is uploaded once. No raycasting, no per-point meshes.
 */
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { PointsPayload } from "@/lib/points";
import { analyse } from "@/lib/spatial";

// --- shaders --------------------------------------------------------------------------------

const POINT_VERT = /* glsl */ `
  attribute float aVisible;
  attribute float aSize;
  attribute float aSeed;
  varying vec3 vColor;
  varying float vFade;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uTime;
  uniform float uTwinkle;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = -mv.z;
    // Far points fade a little: cheap depth cue that reads as volume.
    vFade = aVisible * (1.0 - 0.45 * smoothstep(1.5, 9.0, depth));
    float tw = 1.0 + uTwinkle * sin(uTime * (0.6 + aSeed * 2.4) + aSeed * 40.0);
    float px = uSize * aSize * tw * uPixelRatio * (2.6 / max(depth, 0.35));
    gl_PointSize = aVisible > 0.5 ? min(px, 220.0 * uPixelRatio) : 0.0;
  }
`;

const POINT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vFade;
  uniform float uAlpha;
  uniform float uGain;
  uniform float uSoft;
  void main() {
    if (vFade <= 0.0) discard;
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    // Gaussian disc; uSoft widens the falloff for the haze layers.
    float a = exp(-r2 * uSoft) * (1.0 - r2 * r2) * uAlpha * vFade;
    gl_FragColor = vec4(vColor * uGain * a, a);
  }
`;

const LINE_VERT = /* glsl */ `
  attribute float aVisible;
  varying vec3 vColor;
  varying float vFade;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vFade = aVisible * (1.0 - 0.5 * smoothstep(1.5, 9.0, -mv.z));
  }
`;

const LINE_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vFade;
  uniform float uAlpha;
  void main() {
    if (vFade <= 0.0) discard;
    float a = uAlpha * vFade;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

const NEUTRAL = new THREE.Color("#6a7390").convertSRGBToLinear();
const WHITE = new THREE.Color("#fff7e8").convertSRGBToLinear();

function prng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296);
}

// --- geometry ------------------------------------------------------------------------------

type Built = {
  points: THREE.BufferGeometry;
  dust: THREE.BufferGeometry;
  lines: THREE.BufferGeometry;
  edges: Uint32Array;
  n: number;
};

function buildGeometry(data: PointsPayload): Built {
  const n = data.sample;
  const positions = Float32Array.from(data.xyz);
  const { density, edges, radius } = analyse(positions, n, 2);

  const palette = data.domains.map((d) => new THREE.Color(d.color).convertSRGBToLinear());
  const rnd = prng(1234567);
  const colors = new Float32Array(n * 3);
  const haloColors = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  const seeds = new Float32Array(n);
  const tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const d = density[i];
    const conf = 0.3 + 0.7 * data.confidence[i];
    tmp.copy(NEUTRAL).lerp(palette[data.domain[i]], conf);
    haloColors[i * 3] = tmp.r;
    haloColors[i * 3 + 1] = tmp.g;
    haloColors[i * 3 + 2] = tmp.b;
    // Cores of dense regions burn towards white, like the hot centres in a nebula.
    tmp.lerp(WHITE, d * d * 0.75);
    colors[i * 3] = tmp.r;
    colors[i * 3 + 1] = tmp.g;
    colors[i * 3 + 2] = tmp.b;
    sizes[i] = (0.55 + rnd() * 0.9) * (0.75 + 0.75 * d);
    seeds[i] = rnd();
  }

  const points = new THREE.BufferGeometry();
  points.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  points.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  points.setAttribute("aHalo", new THREE.BufferAttribute(haloColors, 3));
  points.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  points.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
  points.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
  points.computeBoundingSphere();

  // Dust: a Gaussian-jittered ghost of every point, wider where the cloud is thin.
  const dustPos = new Float32Array(n * 3);
  const dustCol = new Float32Array(n * 3);
  const dustSize = new Float32Array(n);
  const gauss = () => {
    const u = Math.max(rnd(), 1e-9), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  for (let i = 0; i < n; i++) {
    const sigma = radius * (0.6 + 1.6 * (1 - density[i]));
    dustPos[i * 3] = positions[i * 3] + gauss() * sigma;
    dustPos[i * 3 + 1] = positions[i * 3 + 1] + gauss() * sigma;
    dustPos[i * 3 + 2] = positions[i * 3 + 2] + gauss() * sigma;
    dustCol[i * 3] = haloColors[i * 3] * 0.6;
    dustCol[i * 3 + 1] = haloColors[i * 3 + 1] * 0.6;
    dustCol[i * 3 + 2] = haloColors[i * 3 + 2] * 0.6;
    dustSize[i] = 0.5 + rnd();
  }
  const dust = new THREE.BufferGeometry();
  dust.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  dust.setAttribute("color", new THREE.BufferAttribute(dustCol, 3));
  dust.setAttribute("aSize", new THREE.BufferAttribute(dustSize, 1));
  dust.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
  dust.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));

  // Filaments: one segment per nearest-neighbour pair, coloured by its endpoints.
  const m = edges.length / 2;
  const linePos = new Float32Array(m * 6);
  const lineCol = new Float32Array(m * 6);
  for (let e = 0; e < m; e++) {
    const i = edges[e * 2], j = edges[e * 2 + 1];
    linePos.set(positions.subarray(i * 3, i * 3 + 3), e * 6);
    linePos.set(positions.subarray(j * 3, j * 3 + 3), e * 6 + 3);
    lineCol.set(haloColors.subarray(i * 3, i * 3 + 3), e * 6);
    lineCol.set(haloColors.subarray(j * 3, j * 3 + 3), e * 6 + 3);
  }
  const lines = new THREE.BufferGeometry();
  lines.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
  lines.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));
  lines.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(m * 2).fill(1), 1));

  return { points, dust, lines, edges, n };
}

/** Rewrite the visibility attributes in place when the country mask changes. */
function applyVisibility(b: Built, visible: Uint8Array) {
  for (const g of [b.points, b.dust]) {
    const attr = g.getAttribute("aVisible") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let i = 0; i < b.n; i++) arr[i] = visible[i];
    attr.needsUpdate = true;
  }
  const attr = b.lines.getAttribute("aVisible") as THREE.BufferAttribute;
  const arr = attr.array as Float32Array;
  const m = b.edges.length / 2;
  for (let e = 0; e < m; e++) {
    const on = visible[b.edges[e * 2]] && visible[b.edges[e * 2 + 1]] ? 1 : 0;
    arr[e * 2] = on;
    arr[e * 2 + 1] = on;
  }
  attr.needsUpdate = true;
}

// --- layers --------------------------------------------------------------------------------

type PointLayerProps = {
  geometry: THREE.BufferGeometry;
  size: number;
  alpha: number;
  gain: number;
  soft?: number;
  twinkle?: number;
  colorAttribute?: "color" | "aHalo";
};

function PointLayer({ geometry, size, alpha, gain, soft = 3.2, twinkle = 0, colorAttribute = "color" }: PointLayerProps) {
  const dpr = useThree((s) => s.viewport.dpr);
  const material = useMemo(() => {
    const m = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      uniforms: {
        uSize: { value: size },
        uAlpha: { value: alpha },
        uGain: { value: gain },
        uSoft: { value: soft },
        uTwinkle: { value: twinkle },
        uPixelRatio: { value: dpr },
        uTime: { value: 0 },
      },
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    if (colorAttribute !== "color") {
      // Reuse the shader with the un-whitened colour for the halo.
      m.vertexShader = POINT_VERT.replace("vColor = color;", `vColor = ${colorAttribute};`)
        .replace("attribute float aVisible;", `attribute vec3 ${colorAttribute};\n  attribute float aVisible;`);
    }
    return m;
  }, [size, alpha, gain, soft, twinkle, dpr, colorAttribute]);
  useEffect(() => () => material.dispose(), [material]);
  useFrame(({ clock }) => {
    material.uniforms.uTime.value = clock.elapsedTime;
  });
  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

function Filaments({ geometry, alpha }: { geometry: THREE.BufferGeometry; alpha: number }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: LINE_VERT,
        fragmentShader: LINE_FRAG,
        uniforms: { uAlpha: { value: alpha } },
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
      }),
    [alpha],
  );
  useEffect(() => () => material.dispose(), [material]);
  return <lineSegments geometry={geometry} material={material} frustumCulled={false} />;
}

/** A sparse static far field so orbiting has a sense of space behind the data. */
function Stars() {
  const geometry = useMemo(() => {
    const n = 1600;
    const p = new Float32Array(n * 3), c = new Float32Array(n * 3), s = new Float32Array(n), seed = new Float32Array(n);
    const rnd = prng(99);
    for (let i = 0; i < n; i++) {
      const u = rnd() * 2 - 1, phi = rnd() * Math.PI * 2, r = 14 + rnd() * 8;
      const q = Math.sqrt(1 - u * u);
      p.set([r * q * Math.cos(phi), r * q * Math.sin(phi), r * u], i * 3);
      const b = 0.2 + rnd() * 0.5;
      c.set([b * 0.85, b * 0.9, b], i * 3);
      s[i] = 0.4 + rnd();
      seed[i] = rnd();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(s, 1));
    g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
    g.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    return g;
  }, []);
  return <PointLayer geometry={geometry} size={7} alpha={0.8} gain={1} twinkle={0.25} />;
}

function Cloud({ data, visible }: { data: PointsPayload; visible: Uint8Array }) {
  const built = useMemo(() => buildGeometry(data), [data]);
  useEffect(
    () => () => {
      built.points.dispose();
      built.dust.dispose();
      built.lines.dispose();
    },
    [built],
  );
  useEffect(() => applyVisibility(built, visible), [built, visible]);

  // A slow drift on top of the orbit so the cloud never looks frozen while idle.
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (group.current) group.current.rotation.y += dt * 0.025;
  });

  return (
    <group ref={group}>
      <Filaments geometry={built.lines} alpha={0.09} />
      <PointLayer geometry={built.dust} size={3.5} alpha={0.22} gain={1} soft={2.2} />
      <PointLayer geometry={built.points} size={34} alpha={0.035} gain={1} soft={2.6} colorAttribute="aHalo" />
      <PointLayer geometry={built.points} size={4.2} alpha={1} gain={1.5} twinkle={0.12} />
    </group>
  );
}

export default function PointCloud({ data, visible }: { data: PointsPayload; visible: Uint8Array }) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [0, 0.5, 3.6], fov: 50, near: 0.01, far: 60 }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
    >
      <Stars />
      <Cloud data={data} visible={visible} />
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.4}
        enableDamping
        dampingFactor={0.06}
        minDistance={0.3}
        maxDistance={12}
        enablePan
        zoomSpeed={0.7}
      />
      <EffectComposer multisampling={0}>
        <Bloom intensity={1.15} luminanceThreshold={0.22} luminanceSmoothing={0.35} mipmapBlur radius={0.75} />
      </EffectComposer>
    </Canvas>
  );
}
