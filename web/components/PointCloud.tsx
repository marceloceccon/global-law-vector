"use client";

/**
 * The cloud itself. One BufferGeometry holds every sampled point; it is drawn twice with the
 * same additive, soft-disc shader: a small bright core and a large faint halo. Additive
 * blending means overlapping points sum, so dense regions glow on their own, and the halo
 * layer gives the volume a gaseous, "nebula" depth without any post-processing.
 *
 * Visibility is a per-vertex attribute (0/1) rewritten when a country is toggled; nothing is
 * re-uploaded except that one Float32 array. No raycasting, no per-point meshes.
 */
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { PointsPayload } from "@/lib/points";

const VERT = /* glsl */ `
  attribute float aVisible;
  attribute float aSize;
  varying vec3 vColor;
  varying float vVisible;
  uniform float uSize;
  uniform float uPixelRatio;
  void main() {
    vColor = color;
    vVisible = aVisible;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // Perspective-correct size, clamped so near points never become giant quads.
    float px = uSize * aSize * uPixelRatio * (2.6 / max(-mv.z, 0.4));
    gl_PointSize = aVisible > 0.5 ? min(px, 160.0 * uPixelRatio) : 0.0;
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vVisible;
  uniform float uAlpha;
  uniform float uGain;
  void main() {
    if (vVisible < 0.5) discard;
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    // Gaussian core with a smooth edge: reads as a soft luminous disc, not a flat sprite.
    float a = exp(-r2 * 3.2) * (1.0 - r2 * r2) * uAlpha;
    gl_FragColor = vec4(vColor * uGain * a, a);
  }
`;

const NEUTRAL = new THREE.Color("#6a7390");

function useCloudGeometry(data: PointsPayload) {
  return useMemo(() => {
    const n = data.sample;
    const positions = new Float32Array(data.xyz);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const palette = data.domains.map((d) => new THREE.Color(d.color).convertSRGBToLinear());
    const tmp = new THREE.Color();
    // Deterministic per-point jitter so the field looks organic rather than gridded.
    let seed = 1234567;
    const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < n; i++) {
      // Uncertain assignments fade towards neutral; confident ones carry the full hue.
      const t = 0.3 + 0.7 * data.confidence[i];
      tmp.copy(NEUTRAL).convertSRGBToLinear().lerp(palette[data.domain[i]], t);
      colors[i * 3] = tmp.r;
      colors[i * 3 + 1] = tmp.g;
      colors[i * 3 + 2] = tmp.b;
      sizes[i] = 0.65 + rnd() * 0.7;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    g.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    g.computeBoundingSphere();
    return g;
  }, [data]);
}

function Layer({
  geometry,
  size,
  alpha,
  gain,
}: {
  geometry: THREE.BufferGeometry;
  size: number;
  alpha: number;
  gain: number;
}) {
  const dpr = useThree((s) => s.viewport.dpr);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          uSize: { value: size },
          uAlpha: { value: alpha },
          uGain: { value: gain },
          uPixelRatio: { value: dpr },
        },
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
      }),
    [size, alpha, gain, dpr],
  );
  useEffect(() => () => material.dispose(), [material]);
  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

/** A sparse, static far field so orbiting has a sense of space behind the data. */
function Stars() {
  const geometry = useMemo(() => {
    const n = 1400;
    const p = new Float32Array(n * 3);
    const c = new Float32Array(n * 3);
    const s = new Float32Array(n);
    let seed = 99;
    const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < n; i++) {
      // Points on a shell of radius 14–22 around the origin.
      const u = rnd() * 2 - 1, phi = rnd() * Math.PI * 2, r = 14 + rnd() * 8;
      const q = Math.sqrt(1 - u * u);
      p.set([r * q * Math.cos(phi), r * q * Math.sin(phi), r * u], i * 3);
      const b = 0.25 + rnd() * 0.5;
      c.set([b * 0.85, b * 0.9, b], i * 3);
      s[i] = 0.5 + rnd();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(s, 1));
    g.setAttribute("aVisible", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    return g;
  }, []);
  return <Layer geometry={geometry} size={7} alpha={0.9} gain={1} />;
}

function Cloud({ data, visible }: { data: PointsPayload; visible: Uint8Array }) {
  const geometry = useCloudGeometry(data);
  useEffect(() => () => geometry.dispose(), [geometry]);

  // Country toggles: rewrite the visibility attribute in place.
  useEffect(() => {
    const attr = geometry.getAttribute("aVisible") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let i = 0; i < visible.length; i++) arr[i] = visible[i];
    attr.needsUpdate = true;
  }, [geometry, visible]);

  // A slow drift on top of the orbit so the cloud never looks frozen while idle.
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (group.current) group.current.rotation.y += dt * 0.03;
  });

  return (
    <group ref={group}>
      <Layer geometry={geometry} size={5.5} alpha={0.85} gain={1.35} />
      <Layer geometry={geometry} size={30} alpha={0.045} gain={1.0} />
    </group>
  );
}

export default function PointCloud({ data, visible }: { data: PointsPayload; visible: Uint8Array }) {
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [0, 0.4, 3.1], fov: 50, near: 0.01, far: 60 }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
    >
      <Stars />
      <Cloud data={data} visible={visible} />
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.45}
        enableDamping
        dampingFactor={0.06}
        minDistance={0.35}
        maxDistance={12}
        enablePan
        zoomSpeed={0.7}
      />
    </Canvas>
  );
}
