async (standIn) => {
  // The duck's eyes in the Microduck simulator page (three r164 under R3F), added to
  // window.__quackback once sandbox.py's _INSTALL has run:
  //   eye()     a picture from the head camera, as the real duck's camera would take it
  //   tof()     the 8x8 depth sensor in the head, emulated by ray casting
  //   head(cmd) the head command the walking policy tracks: [neck, pitch, yaw, roll]
  // and, with `standIn` ({dataUrl, x, y, height}), a stand-in person: a picture standing in the
  // scene, turned toward the duck. The functions appear only once all of it is in place, so an
  // install cut short is noticed (a function is missing) and runs again.
  // Camera and sensor poses come from the robot's MJCF sites head_camera and tof (+x along the
  // optical axis, +y left, +z up). MJCF (x, y, z) is three.js (x, z, -y).
  const qb = window.__quackback;
  if (qb.eye) return;
  const rl = window.rl;

  // three's classes, and the page's renderer and scene, without a second copy of three: one
  // rendered frame hands them to Object3D.onBeforeRender.
  const proto = (o) => Object.getPrototypeOf(o);
  const O3D = proto(proto(proto(rl.camera))); // PerspectiveCamera > Camera > Object3D
  const { renderer, scene, crtTarget } = await new Promise((resolve, reject) => {
    const original = O3D.onBeforeRender;
    const timer = setTimeout(() => {
      O3D.onBeforeRender = original;
      reject(new Error("the simulator rendered no frame in 5 s (is its window minimized?)"));
    }, 5000);
    O3D.onBeforeRender = function (renderer, scene, camera, renderTarget) {
      if (this.isScene && camera === rl.camera) {
        O3D.onBeforeRender = original;
        clearTimeout(timer);
        resolve({ renderer, scene, crtTarget: renderTarget }); // the CRT pass's target
      }
    };
  });
  const grid = scene.children.find(
    (o) => o.isMesh && o.geometry.type === "PlaneGeometry" && o.geometry.parameters.width === 30,
  );
  const r3f = scene.__r3f.root.getState();
  const T = {
    PerspectiveCamera: rl.camera.constructor,
    Vector3: rl.camera.position.constructor,
    Matrix4: rl.camera.matrixWorld.constructor,
    Raycaster: r3f.raycaster.constructor,
    WebGLRenderTarget: crtTarget.constructor,
    Texture: crtTarget.texture.constructor,
    Mesh: grid.constructor,
    PlaneGeometry: grid.geometry.constructor,
  };
  T.MeshBasicMaterial = new T.Mesh().material.constructor;
  const SRGB = "srgb", DOUBLE_SIDE = 2;

  const m = rl.mujoco, SITE = m.mjtObj.mjOBJ_SITE.value;
  function site(name) {
    const id = m.mj_name2id(rl.model, SITE, name); // again each time: a reset swaps the model
    const p = rl.data.site_xpos, R = rl.data.site_xmat; // fresh views, row-major rotation
    const o = new T.Vector3(p[id * 3], p[id * 3 + 2], -p[id * 3 + 1]);
    const axis = (c) => new T.Vector3(R[id * 9 + c], R[id * 9 + 6 + c], -R[id * 9 + 3 + c]);
    const fwd = axis(0);
    return {
      o, fwd, left: axis(1), up: axis(2),
      heading: Math.atan2(-fwd.z, fwd.x), // in the MuJoCo ground frame, counter-clockwise
      pitch: Math.asin(Math.max(-1, Math.min(1, fwd.y))),
      roll: Math.atan2(axis(1).y, axis(2).y), // left side up is positive
    };
  }

  // The head camera: the twin's MJCF camera turned upright, portrait 360 x 640, 72.7 degrees.
  const W = 360, H = 640, VFOV = 72.7;
  const cam = new T.PerspectiveCamera(VFOV, W / H, 0.01, 30);
  const target = new T.WebGLRenderTarget(W, H, { samples: 4 });
  // Like the CRT pass's target: tone mapped and sRGB encoded, so it looks like the main view.
  target.isXRRenderTarget = true;
  target.texture.colorSpace = SRGB;
  target.texture.internalFormat = "RGBA8";
  const pixels = new Uint8Array(W * H * 4);
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  const picture = ctx.createImageData(W, H);
  const basis = new T.Matrix4();
  function eye() {
    if (person) person.onBeforeRender();
    const s = site("head_camera");
    basis.makeBasis(s.left.clone().negate(), s.up, s.fwd.clone().negate()); // looks along -Z
    cam.position.copy(s.o);
    cam.quaternion.setFromRotationMatrix(basis);
    cam.updateMatrixWorld(true);
    const previous = renderer.getRenderTarget(); // put back, so the page's CRT pass is untouched
    renderer.setRenderTarget(target);
    renderer.render(scene, cam);
    renderer.setRenderTarget(previous);
    renderer.readRenderTargetPixels(target, 0, 0, W, H, pixels);
    const row = W * 4;
    for (let y = 0; y < H; y++) {
      picture.data.set(pixels.subarray((H - 1 - y) * row, (H - y) * row), y * row); // GL is bottom-up
    }
    ctx.putImageData(picture, 0, 0);
    const url = canvas.toDataURL("image/jpeg", 0.85);
    return {
      jpeg: url.slice(url.indexOf(",") + 1), vfov: VFOV, heading: s.heading, pitch: s.pitch, roll: s.roll,
    };
  }

  // The 8x8 depth sensor (laid out like microduck_rl's sim/tof.py): row 0 at the top, column 0
  // on the left, slant ranges in mm, status 5 for a valid zone and 255 for none, 4 m at most.
  // It sees the walls, the props, the ball, the floor and the person; not the duck itself.
  const MAX_M = 4.0, VALID = 5, NO_TARGET = 255, BALL_R = 0.05;
  const half = (45 * Math.PI) / 180 / 2;
  const centre = (i) => -half + (i + 0.5) * ((2 * half) / 8);
  const zones = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const el = -centre(r), az = -centre(c);
      zones.push([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]);
    }
  }
  const raycaster = new T.Raycaster();
  raycaster.far = MAX_M;
  function targets() {
    const out = [];
    for (const o of scene.children) {
      if (o.isMesh && o.geometry.type === "PlaneGeometry" && o.geometry.parameters.height === 0.25) {
        out.push(o); // the four walls
      }
    }
    for (const list of Object.values(rl.props || {})) {
      for (const g of list) g.traverse((x) => { if (x.isMesh && !x.userData.fxOverlay) out.push(x); });
    }
    if (person) out.push(person);
    return out;
  }
  const opaque = (hit) => { // ray casting ignores alpha: the person's see-through parts must not echo
    const mask = hit.object.userData.alphaMask;
    if (!mask || !hit.uv) return true;
    const x = Math.min(mask.w - 1, Math.floor(hit.uv.x * mask.w));
    const y = Math.min(mask.h - 1, Math.floor((1 - hit.uv.y) * mask.h));
    return mask.a[y * mask.w + x] >= 128;
  };
  function tof() {
    if (person) person.onBeforeRender();
    const s = site("tof");
    const hits = targets(), dir = new T.Vector3();
    const q = rl.data.qpos, b = rl.ballQposAdr; // the ball from physics: its mesh may lag
    const ball = rl.ballActive ? new T.Vector3(q[b], q[b + 2], -q[b + 1]) : null;
    const distance_mm = new Array(64).fill(0), status = new Array(64).fill(NO_TARGET);
    for (let z = 0; z < 64; z++) {
      const [a, l, u] = zones[z];
      dir.set(0, 0, 0).addScaledVector(s.fwd, a).addScaledVector(s.left, l).addScaledVector(s.up, u).normalize();
      raycaster.set(s.o, dir);
      const hit = raycaster.intersectObjects(hits, false).find(opaque);
      let d = hit ? hit.distance : Infinity;
      if (dir.y < -1e-6) d = Math.min(d, -s.o.y / dir.y); // the floor, y = 0
      if (ball) {
        const ox = s.o.x - ball.x, oy = s.o.y - ball.y, oz = s.o.z - ball.z;
        const bq = ox * dir.x + oy * dir.y + oz * dir.z;
        const disc = bq * bq - (ox * ox + oy * oy + oz * oz - BALL_R * BALL_R);
        if (disc >= 0) {
          const t = -bq - Math.sqrt(disc);
          if (t > 0) d = Math.min(d, t);
        }
      }
      if (d > MAX_M) continue;
      distance_mm[z] = Math.round(d * 1000);
      status[z] = VALID;
    }
    return { distance_mm, status, heading: s.heading, pitch: s.pitch };
  }

  // The head: rl.headTarget holds commands the walking policy tracks, not joint angles.
  const HEAD_MAX = 2.5;
  function head(command) {
    command.forEach((v, i) => { rl.headTarget[i] = Math.max(-HEAD_MAX, Math.min(HEAD_MAX, +v || 0)); });
  }

  // The stand-in person: an unlit picture with its feet on the floor at MuJoCo (x, y), turning
  // (in yaw only) to face the duck before every frame and every depth reading.
  let person = null;
  async function addPerson({ dataUrl, x, y, height }) {
    const img = new Image();
    img.src = dataUrl;
    await img.decode();
    const texture = new T.Texture(img);
    texture.colorSpace = SRGB;
    texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
    texture.needsUpdate = true;
    const width = (height * img.naturalWidth) / img.naturalHeight;
    const geometry = new T.PlaneGeometry(width, height);
    geometry.translate(0, height / 2, 0);
    const material = new T.MeshBasicMaterial({
      map: texture, side: DOUBLE_SIDE, alphaTest: 0.5, toneMapped: false,
    });
    const mesh = new T.Mesh(geometry, material);
    mesh.name = "quackback_person";
    const k = Math.min(1, 256 / Math.max(img.naturalWidth, img.naturalHeight));
    const mw = Math.max(1, Math.round(img.naturalWidth * k));
    const mh = Math.max(1, Math.round(img.naturalHeight * k));
    const mc = document.createElement("canvas");
    mc.width = mw;
    mc.height = mh;
    const g = mc.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0, mw, mh);
    const rgba = g.getImageData(0, 0, mw, mh).data, a = new Uint8Array(mw * mh);
    for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4 + 3];
    mesh.userData.alphaMask = { w: mw, h: mh, a };
    mesh.position.set(x, 0, -y);
    mesh.onBeforeRender = function () {
      const p = rl.data.qpos;
      this.rotation.set(0, Math.atan2(p[0] - this.position.x, -p[1] - this.position.z), 0);
      this.updateMatrixWorld();
    };
    mesh.onBeforeRender();
    scene.add(mesh);
    person = mesh;
  }

  if (standIn) await addPerson(standIn);
  Object.assign(qb, { eye, tof, head });
}
