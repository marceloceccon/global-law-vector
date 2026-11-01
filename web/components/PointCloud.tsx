"use client";

/**
 * The cloud: one BufferGeometry, drawn twice with a soft-disc shader under modest additive
 * blending, so density still reads as brightness without burning to white.
 *
 *  - core   small discs in the point's colour
 *  - halo   a wider, very faint disc whose strength follows local density (lib/spatial.ts)
 *
 * Both colour encodings travel as vertex attributes (country and domain); a uniform mixes
 * between them, so switching modes is instant and uploads nothing. Visibility is a per-vertex
 * attribute rewritten when a country is toggled. A faint unit cube marks the extent the
 * coordinates were normalised to (99th-percentile radius = 1) for orientation; the axes
 * themselves carry no meaning in a UMAP layout. No picking, no post-processing.
 */
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { PointsPayload } from "@/lib/points";
import { analyse } from "@/lib/spatial";

export type ColorMode = "country" | "domain";

const VERT = /* glsl */ `
  attribute vec3 aCountry;
  attribute vec3 aDomain;
  attribute float aVisible;
  attribute float aSize;
  attribute float aDensity;
  varying vec3 vColor;
  varying float vFade;
  uniform float uMode;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uDensityWeight;
  void main() {
    vColor = mix(aCountry, aDomain, uMode);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = -mv.z;
    // Far points fade a little: a cheap depth cue. Halo strength follows local density.
    vFade = aVisible * (1.0 - 0.35 * smoothstep(1.5, 8.0, depth)) * mix(1.0, aDensity, uDensityWeight);
    float px = uSize * aSize * uPixelRatio * (2.4 / max(depth, 0.35));
    gl_PointSize = aVisible > 0.5 ? min(px, 96.0 * uPixelRatio) : 0.0;
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vFade;
  uniform float uAlpha;
  void main() {
    if (vFade <= 0.0) discard;
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float a = exp(-r2 * 3.0) * (1.0 - r2 * r2) * uAlpha * vFade;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

const NEUTRAL = new THREE.Color("#7a8294").convertSRGBToLinear();

function prng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296);
}

function buildGeometry(data: PointsPayload): THREE.BufferGeometry {
  const n = data.sample;
  const positions = Float32Array.from(data.xyz);
  const { density } = analyse(positions, n, 1);

  const countryPalette = data.countries.map((c) => new THREE.Color(c.color).convertSRGBToLinear());
  const domainPalette = data.domains.map((d) => new THREE.Color(d.color).convertSRGBToLinear());
  const rnd = prng(1234567);
  const byCountry = new Float32Array(n * 3);
  const byDomain = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  const tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    tmp.copy(countryPalette[data.country[i]]);
    byCountry.set([tmp.r, tmp.g, tmp.b], i * 3);
    // Uncertain domain assignments fade towards neutral; confident ones carry the hue.
    tmp.copy(NEUTRAL).lerp(domainPalette[data.domain[i]], 0.3 + 0.7 * data.confidence[i]);
    byDomain.set([tmp.r, tmp.g, tmp.b], i * 3);
    sizes[i] = 0.75 + rnd() * 0.5;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.setAttribute("aCountry", new THREE.BufferAttribute(byCountry, 3));
  g.setAttribute("aDomain", new THREE.BufferAttribute(byDomain, 3));
  g.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  g.setAttribute("aDensity", new THREE.BufferAttribute(density, 1));
  g.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
  g.computeBoundingSphere();
  return g;
}

function Layer({
  geometry,
  mode,
  size,
  alpha,
  densityWeight,
}: {
  geometry: THREE.BufferGeometry;
  mode: ColorMode;
  size: number;
  alpha: number;
  densityWeight: number;
}) {
  const dpr = useThree((s) => s.viewport.dpr);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          uMode: { value: 0 },
          uSize: { value: size },
          uAlpha: { value: alpha },
          uDensityWeight: { value: densityWeight },
          uPixelRatio: { value: dpr },
        },
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
      }),
    [size, alpha, densityWeight, dpr],
  );
  useEffect(() => () => material.dispose(), [material]);
  // Ease between the two encodings over a third of a second.
  useFrame((_, dt) => {
    const u = material.uniforms.uMode;
    const target = mode === "domain" ? 1 : 0;
    u.value += (target - u.value) * Math.min(1, dt * 8);
  });
  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

/** Faint cube at the normalisation extent, for orientation while orbiting. */
function Frame() {
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), []);
  const material = useMemo(
    () => new THREE.LineBasicMaterial({ color: "#aab2c5", transparent: true, opacity: 0.09, depthWrite: false }),
    [],
  );
  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );
  return <lineSegments geometry={geometry} material={material} />;
}

function Cloud({ data, visible, mode }: { data: PointsPayload; visible: Uint8Array; mode: ColorMode }) {
  const geometry = useMemo(() => buildGeometry(data), [data]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => {
    const attr = geometry.getAttribute("aVisible") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let i = 0; i < visible.length; i++) arr[i] = visible[i];
    attr.needsUpdate = true;
  }, [geometry, visible]);
  return (
    <>
      <Layer geometry={geometry} mode={mode} size={16} alpha={0.05} densityWeight={0.8} />
      <Layer geometry={geometry} mode={mode} size={3.4} alpha={0.85} densityWeight={0} />
    </>
  );
}

export default function PointCloud({
  data,
  visible,
  mode,
}: {
  data: PointsPayload;
  visible: Uint8Array;
  mode: ColorMode;
}) {
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [0, 0.5, 2.9], fov: 45, near: 0.01, far: 60 }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
    >
      <Frame />
      <Cloud data={data} visible={visible} mode={mode} />
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.3}
        enableDamping
        dampingFactor={0.06}
        minDistance={0.3}
        maxDistance={12}
        enablePan
        zoomSpeed={0.7}
      />
    </Canvas>
  );
}
