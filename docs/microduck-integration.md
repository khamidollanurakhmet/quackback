# quackback: Microduck robot integration brief

> quackback drives the **simulator website** today (see the last section); everything before it
> is about the real robot runtime, for the day quackback talks to a real duck.

**Sources.** I read two upstream repos and used them as read-only data:
- microduck at `main` 942e58e. That commit carries the tags `daemon-v0.17.8` and `daemon-dev-main`. Workspace version 0.17.8, `API_VERSION = 44`.
- microduck_rl at `develop` b02fdea, which is its default branch.

Where the code and the docs disagree, I went with the code and noted the doc. Everything in sections 2–7 marked "exists" is in the stable 0.17.8 tag. A real duck may be running an older release; see section 8, Q15.

---

## 1. System in one paragraph

A Microduck is a Radxa board (RK3566, 4× A55) running seven daemons:
- **robotd** runs the 50 Hz control loop, the ONNX policies, safety, odometry and the voice.
- **configd** handles identity and network.
- **updaterd** handles releases.
- **btd** is BLE.
- **padd** or **netpadd** drives from a Bluetooth pad or a UDP pad. Only one of the two runs at a time.
- **mediad** handles the camera, WebRTC and HTTP.
- **tofd** reads the 8×8 depth sensor.

Each daemon owns exactly one unix socket and speaks **JSON-RPC 2.0 as NDJSON**: one object per line, framed by a single `\n`. There is no broker (`microduck/docs/design/architecture.md:190-192`, `microduck/duck-ipc-proto/src/lib.rs:9-10`). The whole contract lives in the `duck-ipc-proto` crate. Default socket paths are listed at `microduck/duck-ipc-proto/src/lib.rs:513-537`:
- robotd: `/run/robotd.sock`
- tofd: `/run/tofd/tof.sock`
- mediad: `/run/mediad/media.sock`
- configd: `/run/configd.sock`
- updaterd: `/run/updaterd.sock`
- padd: `/run/padd/pad.sock`

The sockets are reachable only on the robot, by design ("The wrong-interface bug class stops existing", `architecture.md:233-236`). robotd's socket is mode 0660, `root:robot` (`microduck/robotd/src/main.rs:66-67`, `robotd/systemd/robotd.service:48-49`). robotd checks no per-call credentials; only configd and updaterd do.

Intents such as `robot.move` only store a value that the loop reads on the next tick. Continuous intents are notifications; discrete ones are requests (`lib.rs:656-664`).

The only network-facing daemon for PC clients is **mediad**. It serves:
- a WebRTC signalling server on `0.0.0.0:8443`, carrying H.264 video and a robot-created `control` datachannel that relays the same NDJSON JSON-RPC to the sockets;
- HTTP on `0.0.0.0:8080`: the console page and `GET /frame` (a PNG snapshot);
- an optional Hugging Face rendezvous relay.

The simulator runs the **real robotd** with `--sim host:port`, against a MuJoCo "duck-body" process from microduck_rl (`microduck/docs/robot/simulation.md:12-17`).

---

## 2. How a PC program can talk to a duck

### 2.1 Every real PC↔duck path

| # | Transport | Drive / sound / video / depth / pose | Auth and setup | PC OS | Works with the sim? | Status | Evidence |
|---|---|---|---|---|---|---|---|
| A | **SSH-forwarded unix socket.** `ssh -L /tmp/robotd.sock:/run/robotd.sock microduck@<ip>`, then NDJSON JSON-RPC on the local end. The same works for `/run/tofd/tof.sock`. | Drive yes, sound yes, depth yes (tofd socket), pose yes (`robot.subscribe`). Video: only `media.frame` (raw UYVY, about 1.8 MiB, on `/run/mediad/media.sock`). | SSH key access. The SSH user must be in the `robot` group: provisioning adds `$SUDO_USER` (`microduck/scripts/provision.sh:391-402`). The Armbian user is chosen at flash time (`docs/robot/install-dev.md:20-27`). | Only the socket-to-socket form is documented, on a Linux laptop. On Windows, the TCP-to-remote-socket form `-L 127.0.0.1:PORT:/run/robotd.sock` is standard OpenSSH but is **not documented upstream and untested**. | In the sim the socket is local: `~/.cache/duck-sim/duck-a.sock` (row H). | Stable; documented as a dev trick. | `microduck/docs/design/robotd-design.md:977-978` "`ssh -L /tmp/robotd.sock:/run/robotd.sock` gives pad-on-laptop, robot-on-board with no code."; `docs/robot/cheatsheet-dev.md:127-137` |
| B | **LAN WebRTC (mediad).** Signalling over `ws://<ip>:8443` using the gst-plugins-rs 0.15.3 webrtcsink JSON. One H.264 video track (constrained baseline). The robot creates a reliable, ordered `control` datachannel carrying NDJSON JSON-RPC. | Drive yes, sound yes, video yes (H.264, **pixels not rotated**; call `media.video` for `rotate` and intrinsics), depth yes (`tof.stream`), pose yes (`robot.subscribe`). **No audio track.** | **No authentication on the LAN.** Plain ws, no TLS, no STUN or TURN on the LAN. The client must accept the robot's channel in `ondatachannel`. Every client call needs its own timeout, because the reply queue drops lines when full. | **No Python client in the repo.** The aiortc consumer was removed, and a "DTLS cipher patch" was once needed. A Python port is unverified on every OS. | Only for ducks listed in `DUCK_SIM_CAMERAS`, at port 8443+i. | Stable. `teleop` channel and audio are design-only. | `microduck/mediad/src/main.rs:9-10` "**It does not authenticate.** Anyone who reaches the signalling port can drive the robot and"; `mediad/src/route.rs:63`; `mediad/src/pipeline.rs:2494-2499`; `spaces/policy-playground/README.md:64` |
| C | **mediad HTTP**, `GET http://<ip>:8080/frame`. | Video only: one upright full-resolution PNG per request. | None. At most 4 encodes at once, otherwise 503. Each request waits for the *next* capture (3 s deadline). The PNG is encoded on the board, at a CPU cost nobody has measured. | Any. | Yes, only for camera ducks, at port 8080+i. | Stable. | `mediad/src/web.rs:92-134`; `mediad/src/frame.rs:9-14` "**It asks for a frame rather than taking the last one.**" |
| D | **`media.stream`**, requested over B or E. The robot **dials out** to a `ws(s)://` URL on the PC and pushes a text hello followed by one binary frame per message. | Video only. The default is **H.264** (VPU, fixed 5 fps / 640 px, **sideways on Zero 3W** even though the hello says `rotate:0`). `encoding:"jpeg"` is rotated on the board and costs robot CPU. | Needs B or E to ask for it. **The robot sends its own HF account token as `Authorization: Bearer`** to whatever URL it is given, plain ws included. The stream outlives the session until `{url:null}`. | Any. The PC must accept inbound connections (Windows firewall rule; my inference). | Only for camera ducks. | Stable. | `mediad/src/session.rs:257-266,297-305`; `mediad/src/stream.rs:253-262`; `mediad/src/main.rs:616-618` |
| E | **HF rendezvous control lane.** `GET /events` (SSE) first, then `POST /send {type:"peer", sessionId, rpc:{…}}`. A full remote WebRTC session can also be bridged through it. | Drive is possible but rate-limited. Sound, depth and pose are possible. Video only through a bridged WebRTC session. | The robot must be signed in with `account.login`. The PC needs its **own** HF token. Limits: **1200 req per 60 s per peer** and 400 notifications per 60 s; one consumer per robot. Set an explicit User-Agent. | Any. Pure-Python client at `microduck/spaces/shared/wire.py`, `control.py` and `rendezvous.py`. | Yes, after `scripts/duck-sim ctl account login`; the duck is listed as `simulated`. | Stable, but older mediads silently drop `rpc` envelopes. | `spaces/shared/wire.py:24-27` "**It is not a teleop lane.** The rendezvous allows 1200 requests per 60 s per peer, and going over"; `mediad/src/relay.rs:1017-1025` |
| F | **UDP pad (netpadd)**, `<ip>:4210/udp`. 28-byte `DKPD` v1 datagrams carrying pad state; `DKST` and `DKRB` come back. | Drive only through pad emulation: sticks give velocity, Start gives init/enable. Mouth via the triggers. No video, depth or odometry. | **Off by default.** Turning it on needs `robotctl configure` on the robot plus restarting padd and netpadd. While it is on, **the Bluetooth pad is disabled**. Unauthenticated; the first sender holds a lock. After about 2 s of silence the duck plays `inquire` and **sits**. | Any. A Python sender is documented at `docs/robot/udp-pad.md:224-243`. | **No.** duck-sim never runs netpadd. | Stable. Docs say it is "not the way to control a robot in general". | `robotd-params/src/lib.rs:206-215`; `docs/robot/udp-pad.md:10-15`; `pad-map/src/loss.rs:17-25` |
| G | **BLE (btd), client `duckctl`.** A Rust CLI distributed as a maturin `bin` wheel; there is **no Python API**. | **Teleop refused**: move, head, look, pose, mouth, stop, relax, subscribe and `tof.stream` all get error 14. Allowed: `robot.sound`, `robot.enable`, `robot.init`, `robot.do`, `robot.policies`, `robot.health`, `system.*`, `net.*`. The advertisement carries the robot's IPv4. | PIN via `system.authenticate` (default 000000, 3 tries). The link is **unencrypted by default**. One central at a time. | Win/macOS/Linux via btleplug. The Windows binary ships only on the `daemon-dev-main` prerelease. | **No** (no BLE in the sim). | Stable. | `btd/src/route.rs:237`; `duckctl/pyproject.toml:3-6`; `duck-ble/src/adv.rs:53-79`; `btd/src/main.rs:123-130` |
| H | **Sim only: local robotd socket**, `~/.cache/duck-sim/duck-a.sock` (plus `duck-a-tof.sock`, `-config.sock`, `-updater.sock`, `-frame.sock`). | Same as A. | File ownership only; the daemons run as your user. | The Linux or macOS host that runs duck-sim. A Windows process cannot open a WSL2 unix socket. | Yes. | Stable. | `docs/robot/simulation.md:80-82` |
| I | **Sim only: MuJoCo body**, TCP `127.0.0.1:7801+i`, NDJSON (`hello` protocol 1, `read`, `tof`, …). | **Ground truth**: world `trunk [x,y,z]` and `imu.quat`, plus `sim_time` and `tof` frames. | None. **Read-only use only**: never send `write`, `gain` or `torque`. Polls take the physics lock, so keep them slow (duck-ether polls every 250 ms). | Host or WSL. | Sim only. | Stable. | `microduck_rl/src/mjlab_microduck/sim/body_server.py:323-333,434-436`; `microduck/duck-ether/src/main.rs:389-406` |
| J | **Sim only: raw camera**, TCP `127.0.0.1:(7901 + duck index)`. Each frame is a `[u32 LE length]` followed by 640×360 UYVY, with no handshake. | Video: **sideways**, about 16.7 unique frames per second even when 30 fps is sent. | None. Several readers are allowed. | Host or WSL. | Sim only. | Stable, on the develop branch. | `microduck_rl/src/mjlab_microduck/sim/camera.py:6-9,28-33`; `body_server.py:470,569` |

**What does not exist:**
- No TCP or HTTP JSON-RPC endpoint on robotd.
- No remote mode for `robotctl`. It is built on `std::os::unix::net::UnixStream` (`microduck/robotctl/src/main.rs:34`) and has no move, stop, head or mouth commands (`main.rs:508-613`).
- No mDNS service. The docs call mDNS "unreliable" (`docs/robot/install-dev.md:32-33`).
- No authentication on any network path except ssh, the rendezvous and the BLE PIN.

**Doc–code conflict.** `docs/robot/udp-pad.md:13` calls WebRTC "authenticated and encrypted". The code says the opposite: `mediad/src/main.rs:9-10` and `mediad/src/route.rs:303` ("there is no gate here — a LAN peer may drive the robot"). DTLS encrypts the session, but on the LAN no peer is authenticated.

### 2.2 RECOMMENDATION (needs your confirmation)

**(a) Simulator phase**
- Run `scripts/duck-sim` (`up` mode) on Linux. On this Windows 11 PC that means **WSL2 Ubuntu**, which upstream has not tested.
- Have `MicroduckBackend` speak NDJSON JSON-RPC to robotd's socket (row H). Build the transport as a generic "NDJSON over a byte stream" with two implementations: AF_UNIX and TCP.
- Ways for a Windows-side quackback to reach the socket: (1) run quackback inside WSL; (2) a TCP bridge inside WSL (for example `socat`, which is not upstream); or (3) `sshd` in WSL, so that the sim and a real duck use the identical `ssh -L` path.
- Video: use row J only for sim tests; it does not exist on real hardware. Use `/frame` (row C) if mediad can be built in WSL, because `/frame` behaves the same on the sim and on a real duck.

Reasons: this is exactly what upstream's own sim clients do (`scripts/duck-sim:544-553,721-734`), and the methods are the same as on a real duck.

**(b) Real duck**
- Control, telemetry and depth: path **A**, an ssh-forwarded `/run/robotd.sock`, plus `/run/tofd/tof.sock` if a ToF sensor is fitted.
- Camera: path **C**, polling `/frame`.

Reasons:
- One wire, so one backend for sim and real.
- It is authenticated by ssh.
- No WebRTC stack is needed.
- It is the documented upstream pattern.

Costs:
- It needs ssh access and membership of the `robot` group.
- `/frame` encodes a full-resolution PNG on the board for every request, so the poll rate has to be modest; no measurement exists.
- Camera intrinsics are **not available** on this path. `media.video` is answered only on the datachannel or the rendezvous lane (`mediad/src/session.rs:116-119,169`), so use constants (IMX219 is about 62° HFOV) or calibrate.

Alternative: path **B** (aiortc). A single session carries control, H.264, ToF and intrinsics, with no ssh. But it is unauthenticated, has no reference Python client, and its datachannel exists only while mediad's camera pipeline runs (`docs/design/remote-webrtc.md:97-99`).

Not recommended:
- E, because of the rate limits and because it is "not a teleop lane".
- F, because it is off by default, disables the Bluetooth pad, sits after about 2 s of silence, and does not exist in the sim.
- G, because teleop is refused over BLE. G is still useful for discovering the IP (`duckctl ip`).

---

## 3. Method catalog (robotd unless noted; API 44)

**Envelope rules. Getting any of these wrong fails silently.**
- `params` must be a JSON object, even `{}`. An absent or null `params` is INVALID_PARAMS for any method that takes params (`duck-ipc-proto/src/lib.rs:1676-1679`).
- Every params struct is `deny_unknown_fields`, so an extra key is −32602, and an unknown method is −32601 (`lib.rs:66-69`).
- **Notifications are dropped silently** unless they are one of robot.move, head, pose, mouth, do or sound (`robotd/src/main.rs:4440-4446,4489-4535`). Send stop, enable, look, init, relax and subscribe as **requests**.
- `id` must be a u64 or a string. An unparseable line gets PARSE_ERROR with `id:null`, so readers must tolerate unsolicited error lines (`main.rs:4426-4436`).
- Requests are capped at 64 KiB (`main.rs:69`).
- Each connection is served one request at a time (`lib.rs:1237`).
- `hello {api_version}` returns `{api_version, daemon_version, revision}`. It is informational only: "**No daemon refuses a call because this number differs.**" (`lib.rs:57`). `lib.rs:39` still says the opposite and is stale.

| Need | Method, kind, params | Result, limits and caveats | Status | Evidence |
|---|---|---|---|---|
| Walk velocity | `robot.move` (notification) `{vx m/s fwd, vy m/s left, vyaw rad/s +left}`. Missing fields default to 0. | No reply as a notification. As a request it answers `{accepted:true}`. Last writer wins. Resend more often than every 500 ms; upstream clients use 10 Hz and the protocol says 20–50 Hz. **No robot-side clamp.** Reference values: pad vx ±0.4, vy ±0.3, vyaw ±1.0; console 0.3 m/s and 1.5 rad/s; sim `drive` default 0.15. EMA smoothing with α=0.2 per tick. **Ignored while** sitting (sit latch), while `robot.pose` is active, while picked up, or in limp-fall. Needs the policy enabled. | exists | `lib.rs:2261-2270` "/// Yaw rate, rad/s, positive turns left."; `robotd-params/src/lib.rs:146-156,2407`; `robotd/src/control.rs:630-641`; `robotd/src/main.rs:3124-3131` |
| Turn in place | `robot.move {vx:0, vy:0, vyaw:≠0}` | There is no separate method. With a standing network loaded (the sim loads `alpha_stand.onnx`), a command norm ≤ 0.05 selects the standing net, so a very slow yaw does not turn. | exists | `duck-control/src/policy.rs:27-28,312-316`; `scripts/duck-sim:243` |
| Stop | `robot.stop` (**request**) `{}` | `IntentResult`. Zeroes the twist; the policy stays on and the robot stands. As a notification it does nothing. Simply going quiet also stops the robot within about 500 ms. | exists | `lib.rs:673-674` "/// Stop moving — zero the velocity. Not \"go limp\"."; `robotd/src/main.rs:5463-5466` |
| Emergency stop | No dedicated method. Choices: `robot.stop` (stand); `robot.enable {on:false}` ("disabled — returning to the home pose"); `robot.relax` (torque off, **collapses**, clears enable); `robot.rest` (sit, torque off, servo reboot); `robot.shutdown` (sit, then **power off the board**). | The design doc's "maintenance namespace" e-stop is not implemented. mediad: "It is still not a physical e-stop." | **absent** (only partial substitutes) | `docs/design/robotd-design.md:869-870`; `mediad/src/route.rs:89-95`; `robotd/src/main.rs:5485-5494`; `robotd/src/intents.rs:487-493` |
| Enable / stand | `robot.enable {on:bool, toggle?:bool}` (request) | Always accepted, never refused for being fallen; ramps up from limp. Starts **disabled** at boot. **Use this in the sim, not `robot.init`**: init's 2 s ramp tips the sim duck over. The `robot.policies.enabled` field is the config flag, *not* the live driving state; the live state shows as `robot.state.policy == "held"` or in enable's `reason`. | exists | `robotd/src/main.rs:5471-5498`; `scripts/duck-sim:534-541`; `lib.rs:2591-2593` |
| Posture | `robot.do {skill}` (for example `"sit_toggle"`; request). `robot.pose {z, roll, pitch, active}` (notification). `robot.policies` returns `{mode, enabled, slots, skills, homed?, sitting?}`. `robot.init`, `robot.rest`. | `do` is refused unless enabled and homed. The refusal text "press Start on the pad" is stale, because robotctl now has `robot enable`. Pose trained ranges: z −0.025..+0.010 m, roll and pitch ±0.26 rad, **not clamped**. **Pose is a mode switch that never expires and zeroes the twist**, so always send `{active:false}`. A seated duck needs `sit_toggle`, not init. | exists | `robotd/src/main.rs:5145-5153`; `lib.rs:2502-2503,2621-2629`; `robotd/src/intents.rs:142-143` |
| Head | `robot.head {neck_pitch, head_pitch, head_yaw, head_roll}` rad (notification). `robot.look {x,y,z, neck_pitch}` m in the trunk frame (request) returns `{head, clamped}`. | look is never refused. `clamped` means travel limits or the gimbal near ±90° yaw, so turn the body instead. z is measured from the trunk; the floor is about −0.12 m. Head targets are not deadman-expired, but **idle glances take over after 2.0 s** without a head command while the robot is still. Resend more often than every 2 s. | exists | `lib.rs:2277-2311` "/// Above it. NOTE: trunk frame, not floor — the floor is about 0.12 m below."; `robotd-params/src/lib.rs:935-936`; `robotd/src/main.rs:3144-3150,5108-5131` |
| Built-in sound | `robot.sound {tag: alarm\|greet\|inquire\|peck\|chirp\|coo\|wheee, hold?:bool}` (request or notification) | A random variant from a per-robot synthesised duck-vocalisation bank, 48 kHz. `inquire` is a "Rising question". Refused if audio is disabled or there is no bank. `wheee` + `hold:true` must keep arriving. | exists | `lib.rs:2313-2332,738-741`; `robotd/src/main.rs:5222-5232` |
| Voice / TTS / arbitrary audio | — | No method takes audio data, a file path or text. robotd spawns one `aplay` per bank wav. | **absent** | `robotd/src/sound.rs:1-8`; `lib.rs:2349-2359` |
| Mouth | `robot.mouth {open 0..1}` (notification) | **Never expires** ("a mouth left open by a dead client"), so send 0 when done. The servo gets max(intent, idle-yawn mouth); the theremin and chorale override it. **No physical effect in the sim** (the model has no mouth joint). | exists | `lib.rs:729-731`; `robotd/src/intents.rs:144-146`; `robotd/src/main.rs:3791-3798`; `body_server.py:33-36` |
| LEDs | `robot.flashlight {on, toggle, color}` (request) | **Refused on the Zero 3W and in the sim.** Status, battery and network LEDs have no wire method. | exists only on the beta board | `lib.rs:743-744` |
| Status | `robot.health {}` returns `{healthy, degraded, reason, battery{volts,percent}, motors, cpu_temp_c, cpu_throttle, control_loop, bus, imu}`. `robot.mode`. `system.info` (configd; has a `simulated` flag). | Prefer polling these for status; they cost the 50 Hz loop nothing. | exists | `lib.rs:3640-3670`; `robotctl/src/main.rs:3368-3369` |
| Telemetry / odometry | `robot.subscribe {hz?}` (request) turns the connection into a stream of `robot.state` notifications. | Wire keys are `"move"` (`requested`, `applied` (EMA-smoothed), `limited_by` ∈ {`deadman`, `joint_range`, `not_finite`}) and `"loop"` (`hz`, `missed`), plus `policy`, `safety{fallen, limp, gravity, gain, picked_up}`, `head[4]`, `odom{position[3] m, yaw rad}` (contact odometry in the **boot frame**, no reset method, drifts), `imu{gyro, quat}`, `frames{camera, tof}` (poses in the trunk frame) and `t_ns`. **Any subscription makes the 50 Hz thread build a full state every tick**, so subscribe only while it is needed. Buffer 256, drops on lag. | exists | `lib.rs:3964-3971,4156-4166,4040-4044`; `robotd/src/main.rs:3831-3837,99-103` |
| Geometry | `robot.model {}` returns `{asset, trunk_height_m, joint_names, head_joints, tof_beams[64], tof_fov_deg, frames_at_zero}`. | No camera intrinsics. | exists | `lib.rs:4100-4116` |
| Depth | tofd: `tof.stream {}` (request) returns `{accepted, sensor?, unavailable?, rows, cols, hz}`, then `tof.frame {seq, at_us, t_ns, rows, cols, distance_mm[64] i16, status[64] u8}`. | 8×8 zones, 45°×45°, about 15 Hz, slant range in the sensor frame. Valid statuses are 5 and 9; 255 means no target. Trusting only 5 and 9 "**stops seeing a hand at about 30 cm**". Readings under 0.10 m are treated as noise. **Most ducks have no sensor fitted**; the stream is then accepted with `sensor:null`. Allowed over WebRTC, refused over BLE. | exists (hardware often absent) | `lib.rs:5021-5048`; `tof/src/main.rs:8-10,118-121`; `docs/robot/cheatsheet.md:860-862`; `kinematics/src/tof.rs:32-34,99-101` |
| Camera | Local: `media.frame` on `/run/mediad/media.sock` returns a JSON header `{width, height, format:"UYVY", bytes, captured_at_unix_us, rotate}`, then raw bytes; one connection per frame. Datachannel and lane only: `media.video` → `{width, height, rotate, mono_ns, real_ns, intrinsics?}`, and `media.stream`. HTTP: `/frame`. | `media.frame` is deliberately not a Call and never travels over WebRTC. | exists | `lib.rs:615-618`; `robotctl/src/frame.rs:49-68`; `mediad/src/session.rs:97-113` |
| Go to point / navigation / obstacle avoidance | — | Mapping, localisation and planning (maploc) are "unowned". The ToF Reprojector is a library only. | **absent** (designed only) | `docs/design/robotd-design.md:1061`; `tof/src/main.rs:13-15` |
| Person detection | — | The on-robot detector finds only Microducks: one class, 2 Hz, off by default, results only as `media.detections` to datachannel peers. "person detected" is a design example only. | **absent** | `duck-detect/src/lib.rs:1-4`; `docs/design/architecture.md:287-289` |
| Microphone | — | Only robotd's internal pet-detect worker reads it (off by default). No client API, no WebRTC audio track. | **absent** | `robotd/src/main.rs:2624-2628`; `mediad/src/main.rs:75-78` |

---

## 4. Safety behaviour already in the duck

- **Speed limits: none on the robot.** `Limit` is `{Deadman, Range, NotFinite}` (`duck-control/src/safety.rs:84-91`). `[pad_drive]` limits apply only to what the pad asks for: "This is what the *pad* asks for, not a limit on the robot." (`robotd-params/src/lib.rs:124-126`). The design doc's `limited_by:["max_velocity"]` example (`robotd-design.md:721`) is not implemented. **quackback must clamp vx, vy and vyaw itself.**
- **Smoothing.** Twist and head go through an EMA with cmd_alpha = head_alpha = 0.2 per 20 ms tick (`deploy/robotd.toml:58-63`). Stops and turns ramp over about 100–200 ms rather than snapping. Account for this in a stop-at-distance controller.
- **Deadman.** A twist older than `safety.deadman_ms` (500 ms) is zeroed, and the robot **stands; it does not go limp** (`safety.rs:17-20,74`). Only the twist expires:
  - the head target persists, but idle glances start after 2 s;
  - `robot.pose` never expires, and while it is active it keeps the twist zeroed;
  - the mouth never expires.

  The twist starts maximally stale at boot (`robotd/src/intents.rs:269-275`).
- **Connection loss.** Over ssh or a socket, losing the link means the deadman stops the robot within about 0.5 s. mediad sends no stop when a session drops (`docs/design/remote-webrtc.md:378-379`). A running `media.stream` keeps redialling until `{url:null}`.
- **Falls.** `safety.fallen` (gravity z > −0.5 for 200 ms) **gates nothing**; a fallen robot can still be enabled and driven (`docs/design/robotd-design.md:566-569`). `limp_fall` is on by default (`robotd-params/src/lib.rs:2292`). It drops gain while falling and stands back up. Pick-up detection pauses the policy (`docs/robot/cheatsheet.md:481-483`). `battery_empty_shutdown` (default true) sits the robot down and powers it off at 6.6 V (`robotd-params/src/lib.rs:2211,2291`).
- **Authority arbitration: none.** Each slot is last-writer-wins. Two drivers "interleaving into one slot" means the robot obeys neither (`remote-webrtc.md:466-469`). The single-writer token is an M6 to-do, and `robot.enable` is not even classified as mutating (`docs/project/roadmap.md:191-193`). A connected pad (padd resends at 50 Hz, `lib.rs:948-950`), a browser console, or a second LAN WebRTC peer will interleave with quackback. Nothing refuses a second LAN consumer (`mediad/webclient/index.html:1076-1077`).
- **Auth.** robotd is protected by socket permissions only. WebRTC on the LAN and the UDP pad are open. The BLE PIN travels in clear by default.

---

## 5. Audio

- **The duck cannot play a PC-generated TTS wav through any API.** `robot.sound` accepts only `{tag, hold}` from a closed enum (`lib.rs:2349-2359`). The bank is a seeded duck-vocalisation synth, not speech (`sounds/src/lib.rs:1-10`).
- mediad has no audio track, even though its description says "camera, mic, WebRTC" and the design lists a two-way mic track (`docs/design/remote-webrtc.md:157-163`). No audio code exists in `mediad/src`.
- Unsupported routes, none documented: ssh plus `aplay` would contend with robotd's per-sound `aplay` on the codec; replacing wavs in `/var/lib/robot/sounds/<tag>/` would only be picked at random. Board I2S is fixed to the 48 kHz family, so 44.1 kHz audio plays about 9% off pitch (`sounds/src/synth.rs:3-5`). The speaker rolls off below about 300 Hz (`robotd/src/sound.rs:65-70`).
- **Microphone.** The Zero 3W has a TLV320AIC3104 codec with a head mic (Mic3R) (`deploy/audio/aic3104-init.sh:13-36`), described as "practically a contact mic for head scratches" (`pet-detect/src/worker.rs:51-53`). The beta board has a face-board mic. **No API streams it.** Only robotd's pet-detect worker reads it, that worker is off by default, and its events go only to logs.
- **What this means for "the duck asks out loud and listens":**
  - The spoken question has to come from a PC-side (or other non-duck) speaker.
  - The duck can supply a cue: `robot.sound {"tag":"inquire"}` (rising question) or `greet`. It can also animate with `robot.mouth`, keeping in mind that it never expires and has no effect in the sim.
  - The answer has to be captured by a **PC-side microphone**, so the conversation range is limited by where the PC's speaker and mic are, not where the duck is.
- In the sim, robotd plays bank sounds through host ALSA `default` (`scripts/duck-sim:255-262`). Under WSL2 that depends on WSLg audio, which is unverified. The sim has no microphone.

---

## 6. Perception and navigation

**Camera**
- Zero 3W: IMX219, about 62° HFOV, mounted a quarter turn off (`rotate=90`). Beta: GC2093, `rotate=0` (`mediad/src/camera.rs:10-16`, `mediad/src/main.rs:111-121`).
- Quality presets: 1080p30, 720p30 (default), 720p15, 360p30.
- Ways off the robot: `/frame` PNG (upright), WebRTC H.264 (not rotated; use `media.video.rotate` and intrinsics), or `media.stream`.
- **Person detection must run on the PC.**
- The sim camera has a 45° vertical FOV, about 72.7° horizontal over 640×360 (`mediad/src/camera.rs:329-330`). That differs from the real camera, so use per-target intrinsics.

**Depth**
- See the depth row in section 3: often absent, 15 Hz, statuses 5/9/255.
- The ToF is mounted in the head and "looks where the head looks" (`tof/src/lib.rs:4-5`). Its live pose is in `robot.state.frames.tof` and its beams come from `robot.model`, so the PC can reproject the zones into the trunk frame itself.
- **The sim ToF only ever emits status 5 or 255** (`microduck_rl/.../sim/tof.py:75-98`), so a stop rule tested only in the sim will look more reliable than it is on hardware.

**Odometry**
- `robot.state.odom` is leg forward kinematics plus integrated IMU yaw, in the boot frame, with no magnetometer and no reset method (`odometry/src/lib.rs:19-21`; `lib.rs:4156-4166`).
- Drift has been measured only in the sim: 9 mm over 3.2 m (`odometry/src/lib.rs:27-31`). Real-robot and yaw drift are unmeasured.
- quackback must record the base pose itself at session start.

**Navigation**
- There is no go-to-point. **Walking to a point or a person must be closed-loop on the PC**: detection, bearing, `robot.move` at 10–20 Hz, odometry and ToF or vision distance, with the speed clamp and the stop distance enforced client-side.
- Use `robot.look` or `robot.head` to aim the camera, resent more often than every 2 s. If `clamped`, turn the body with `vyaw`.
- `robot.subscribe` makes the control thread do extra work every tick, so use it only while navigating.

---

## 7. Running the simulator

**Requirements** (`docs/robot/simulation.md:30-41`; `scripts/duck-sim:198,213-229`):
- microduck plus Rust ≥ 1.89 (`Cargo.toml:94`). The script runs `cargo build -p robotd -p robotctl -p tof -p sounds -p configd -p updater`.
- `python3`.
- microduck_rl at `$DUCK_SIM_RL` (default `~/Pollen/microduck_rl`), on `develop`, after `uv sync`. That gives Python 3.12 (<3.13), mjlab 1.3.0, torch 2.9.1 and onnxruntime, and it provides `libonnxruntime`.
- The runtime is CPU MuJoCo; CUDA is needed only for training (`microduck_rl/README.md:24`).
- Policies are fetched from HF `pollen-robotics/microduck-policies` on first run.

**Commands** (`simulation.md:62-71`):

```sh
scripts/duck-sim                 # up: duck-body + tofd --sim + robotd --sim + configd + updaterd, then robot.enable
DUCK_SIM_VIEWER=0 scripts/duck-sim                 # headless
DUCK_SIM_CAMERAS=a scripts/duck-sim                # + mediad: console/frame http://127.0.0.1:8080, ws :8443, raw 7901
scripts/duck-sim status | drive [vx vyaw] | ctl <robotctl args> | monitor | realtime | down
```

The stack can also be started by hand, as in `simulation.md:197-215`. `boot N` (the systemd-nspawn container mode) is Linux + sudo only, stages only robotd, robotctl and tofd, and is not needed for quackback.

**Real-time factor is critical.** Below 1.0× the policies cannot balance, and below 45 Hz the ducks report unhealthy. Each camera adds about 12 ms of render on the physics thread (`simulation.md:174-179`; `body_server.py:495-498`).

**Windows, WSL2 and Docker**
- Native Windows: not possible. robotd imports tokio `UnixListener` unconditionally (`robotd/src/main.rs:51`), duck-sim is POSIX sh, and sound uses `aplay`.
- Docker: no sim material uses it; the Dockerfiles in `scripts/` are for other purposes.
- WSL2 Ubuntu in `up` mode is the only plausible route, and it is **untested upstream**. Risks: OpenGL rendering through WSLg (`MUJOCO_GL` is never set), holding 1.0× real time, mediad needing GStreamer ≥ 1.22 plus webrtcsink/nice/x264 (upstream used a local `/opt/gst-plugins-rs`, `scripts/duck-sim:413`), WSL2 localhost forwarding for 127.0.0.1-bound ports, and audio.
- A split design (body on Windows, daemons in WSL over TCP) is anticipated in the design (`docs/design/simulation.md:104-106`) but untested, and duck-sim does not support it.

**What the sim provides**
- The real robotd: policies, safety, odometry and the full IPC.
- ToF: 8×8, 45°, 4 m, 15 Hz, statuses 5 and 255 only.
- Camera: opt-in, 640×360 UYVY, sideways on the raw port, about 16.7 unique fps. The raw port is **7901 + duck index**; `simulation.md:170`'s "+1 per camera" is wrong (`body_server.py:569`).
- `/frame` and WebRTC only through mediad.
- Bank sounds on the host speakers.
- Absent: microphone, flashlight (refused), mouth joint, head IMU, BLE, padd/netpadd (`--fake-pads`), and any person model.
- Ground truth through body `read` (row I).

**Connecting quackback.**
- Robotd: AF_UNIX `~/.cache/duck-sim/duck-a.sock`. Send `robot.enable {on:true}` as a request, then `robot.move` notifications at 10 Hz, exactly as `duck-sim:544-553,721-734` does.
- Depth: `duck-a-tof.sock`.

**Start pose and placing a person**
- The start pose is hard-coded at x=0, y=0.5·i, zero yaw (`body_server.py:261-266`). Only the joint keyframe (`DUCK_SIM_KEYFRAME`: SIT, HOME, STAND or FOLD) is selectable.
- No runtime API adds or moves objects.
- Ways to do it without editing upstream:
  - Pass a custom MJCF with `DUCK_SIM_SCENE=/path/scene.xml`. It must keep the robot naming and matching keyframes. `HOME` reads no scene keyframe (`body_server.py:140-141`), so a scene with extra freejoints can work.
  - Write a quackback-owned launcher that imports `body_server` as a library, as upstream does (`microduck_rl/scripts/pickup_e2e_robotd.py:25-44`). It can build the scene with MjSpec and set the trunk qpos after `place()`.
- Two warnings: `scene_apartment` may fail to compile because of a keyframe/nq mismatch with its 7 freejoints (unverified), and it has a stair hole 0.70 m south of the spawn point (`apartment.xml:95-96`).

---

## 8. Gaps and ambiguities (questions for you)

1. **Control transport for a real duck.** Options: (a) ssh-forwarded `/run/robotd.sock` (recommended); (b) LAN WebRTC datachannel via aiortc; (c) both, behind one NDJSON interface; (d) rendezvous lane (remote, rate-limited). Which one?
2. **SSH prerequisites.** Do you have key-based ssh to the duck, with that user in the `robot` group? What is the username? Do you accept that Windows OpenSSH `-L 127.0.0.1:PORT:/run/robotd.sock` (TCP to a remote unix socket) is undocumented upstream and has to be tested first?
3. **Duck discovery.** Should the IP be (a) configured manually, (b) read via `duckctl ip` (BLE; the Windows build is only on the dev prerelease), or (c) taken from the router? mDNS is unreliable.
4. **Sim host.** Should the simulator run as duck-sim `up` in WSL2 Ubuntu (untested upstream)? If so, should quackback (a) run inside WSL, (b) reach the socket through a WSL-side TCP bridge (socat), or (c) use `sshd` in WSL so the sim path matches the real-duck ssh path?
5. **Sim camera.** In the simulator phase, should quackback (a) build mediad with GStreamer/webrtcsink in WSL and use `/frame` (same as on a real duck), (b) read the sim-only raw UYVY port 7901+i, or (c) skip vision and mock detections?
6. **Speaking the question.** The duck has no TTS or wav path. Options: (a) PC speaker plus `robot.sound inquire` and `robot.mouth` as cues; (b) a separate speaker you carry or place; (c) an on-duck change to microduck, which is out of scope unless you say otherwise.
7. **Hearing the answer.** No duck mic API exists. Should it be (a) the PC microphone, or (b) a separate wireless or USB mic? Is the limited range acceptable?
8. **Stop distance of 40 cm or more.** Does your duck have the head ToF? If yes, which statuses count as valid: (a) 5 and 9 only, which loses moving targets near 30 cm, or (b) the theremin set 4/5/6/9/10/12/13? If no ToF, is a camera-only estimate (bounding box plus assumed intrinsics) acceptable?
9. **Speed caps enforced by quackback.** (a) Pad defaults 0.4 / 0.3 / 1.0; (b) something conservative such as 0.15 m/s and 0.5 rad/s; (c) other values?
10. **Authority.** Must quackback (a) require that no pad or console is driving and refuse to start otherwise, (b) only warn, or (c) detect a driver, for example via `pad.status` or `system.services`? There is no arbitration on the robot.
11. **Lifecycle calls.** May quackback send `robot.enable {on:true}` itself, `robot.do sit_toggle` when the duck is `sitting`, and sit or disable it at the end? Or must a human always start the policy?
12. **Abort and "e-stop" semantics.** On abort, should quackback (a) send `robot.stop` (stand), (b) send `robot.enable {on:false}` (home pose), or (c) send `robot.relax` (collapse)?
13. **Return to base.** Odometry has no reset, drift on hardware is unmeasured, and yaw is integrated IMU. Is odometry alone acceptable, or should quackback add a visual marker for the base, or accept a human repositioning the duck?
14. **Idle head.** Should quackback re-send `robot.look`/`robot.head` more often than every 2 s, or will you change `[idle]` on the robot? That needs `robotctl configure` on the robot.
15. **Version target.** Should quackback target microduck 0.17.x / API 44? Which release does your duck run? Older releases may lack `/frame`, `robot.look`, `robot.model` or the rendezvous `rpc` lane, and the `hello` handshake will not catch that.
16. **Sim ground truth.** May tests read the sim body's `trunk` and `quat` for assertions only, never for navigation?
17. **Person in the sim.** Should quackback (a) ship its own MJCF with a person proxy (a real detector will likely not fire on primitives), (b) inject mock detections, or (c) both?
18. **`media.stream`.** It sends the robot owner's HF token to the receiver URL. Should quackback avoid it entirely? I recommend that.
19. **WebRTC video orientation and intrinsics.** If path B is used: should quackback rotate frames using `media.video.rotate`? Accept that H.264 `media.stream` frames are sideways despite `rotate:0`?
20. **Documentation drift to keep in mind.** Each of these disagrees with the code:
    - `udp-pad.md` says WebRTC is "authenticated".
    - The roadmap says API v14.
    - `robotd-design.md` lists look, pose and do as deferred.
    - The robot.do refusal text says to press Start on the pad.
    - `simulation.md` says the frame port is "+1 per camera".
    - duck-sim's comment says there is "no robotctl enable".

    Do you agree the code at API 44 is authoritative?

---

## 9. Evidence index

| Topic | Citation | Quote |
|---|---|---|
| Wire format | microduck/duck-ipc-proto/src/lib.rs:9-10 | "**Wire format: JSON-RPC 2.0, one object per line (NDJSON), over a unix socket.**" |
| robotd socket path | microduck/duck-ipc-proto/src/lib.rs:519 | "pub const ROBOT: &str = \"/run/robotd.sock\";" |
| Socket mode | microduck/robotd/src/main.rs:66-67 | "const SOCKET_MODE: u32 = 0o660;" |
| Off-robot by design | microduck/docs/design/architecture.md:233-236 | "**The wrong-interface bug class stops existing.** Binding `0.0.0.0` by typo, by" |
| SSH forward | microduck/docs/design/robotd-design.md:977-978 | "`ssh -L /tmp/robotd.sock:/run/robotd.sock` gives pad-on-laptop, robot-on-board with no code." |
| Version is not a gate | microduck/duck-ipc-proto/src/lib.rs:57 | "**No daemon refuses a call because this number differs.**" |
| Continuous vs discrete intents | microduck/duck-ipc-proto/src/lib.rs:656-664 | "**Continuous** — `move`, `head`. Sent as *notifications* (no `id`, no reply)," |
| Notifications without an id dropped | microduck/robotd/src/main.rs:4440-4446 | "let Some(id) = request.id.clone() else {" |
| Params must be an object | microduck/duck-ipc-proto/src/lib.rs:1676-1679 | "serde_json::from_value(params.cloned().unwrap_or(Value::Null))" |
| MoveParams | microduck/duck-ipc-proto/src/lib.rs:2261-2270 | "/// Yaw rate, rad/s, positive turns left." |
| Deadman | microduck/duck-control/src/safety.rs:17-20 | "Plus a deadman on the command itself: if intents stop arriving, the velocity goes to" |
| Deadman default | microduck/duck-control/src/safety.rs:74 | "deadman: Duration::from_millis(500)," |
| No velocity limit | microduck/duck-control/src/safety.rs:84-91 | "pub enum Limit {" |
| Pad limits are not robot limits | microduck/robotd-params/src/lib.rs:124-126 | "This is what the *pad* asks for, not a limit on the robot." |
| Stop | microduck/duck-ipc-proto/src/lib.rs:673-674 | "/// Stop moving — zero the velocity. Not \"go limp\"." |
| Enable | microduck/robotd/src/main.rs:5475 | "let on = if p.toggle { !intents.enabled() } else { p.on };" |
| Enable, not init, in sim | microduck/scripts/duck-sim:534 | "# **`robot.enable`, not `robot init`.**" |
| Sit latch ignores move | microduck/robotd/src/control.rs:630-632 | "c.twist = [1.0, 0.0, 0.0];" |
| Look is trunk frame | microduck/duck-ipc-proto/src/lib.rs:2298 | "/// Above it. NOTE: trunk frame, not floor — the floor is about 0.12 m below." |
| Idle head timing | microduck/robotd-params/src/lib.rs:935-936 | "after_s: 2.0," |
| Sound tags | microduck/duck-ipc-proto/src/lib.rs:2322 | "/// Rising question." |
| No arbitrary audio | microduck/robotd/src/sound.rs:1 | "//! The robot's voice, at play time: pick a wav from the bank, feed it to `aplay`." |
| No mediad mic | microduck/mediad/src/main.rs:78 | "server, and a certificate, on the day a microphone or a browser gamepad is wanted." |
| Mouth persists | microduck/robotd/src/intents.rs:144-146 | "a mouth left open by a dead client is exactly what the prototype does." |
| Flashlight refused | microduck/duck-ipc-proto/src/lib.rs:743-744 | "Refused, with a reason, by a board without one (the Zero 3W, the simulator)." |
| Odometry frame | microduck/duck-ipc-proto/src/lib.rs:4156-4166 | "/// at boot. There is no magnetometer and no absolute reference — this frame is" |
| Subscription load | microduck/robotd/src/main.rs:3831-3837 | "Only assemble a frame when somebody is subscribed. On a robot nobody usually is," |
| ToF frame shape | microduck/duck-ipc-proto/src/lib.rs:5033-5048 | "pub distance_mm: Vec<i16>," |
| ToF often absent | microduck/tof/src/main.rs:8-10 | "//! (the common case on a duck without the head module) must be a daemon logging" |
| ToF status pitfall | microduck/docs/robot/cheatsheet.md:860-862 | "**stops seeing a hand at about 30 cm**" |
| media.frame local only | microduck/duck-ipc-proto/src/lib.rs:617-618 | "/// Deliberately not a `Call`: its binary tail must never enter Service/Lane routing" |
| WebRTC has no auth | microduck/mediad/src/main.rs:9-10 | "**It does not authenticate.** Anyone who reaches the signalling port can drive the robot and" |
| WebRTC allowlist | microduck/mediad/src/route.rs:63 | "RobotMove(_) \| RobotHead(_) \| RobotLook(_) \| RobotPose(_) \| RobotMouth(_) => true," |
| Robot creates the channel | microduck/mediad/webclient/index.html:1192-1196 | "// **The robot creates the datachannel, so we receive it.**" |
| /frame waits for next capture | microduck/mediad/src/frame.rs:9-14 | "**It asks for a frame rather than taking the last one.**" |
| media.stream token | microduck/mediad/src/stream.rs:253-262 | "// **The robot's own credential, as the handshake header.**" |
| Rendezvous limits | microduck/spaces/shared/wire.py:24-27 | "**It is not a teleop lane.** The rendezvous allows 1200 requests per 60 s per peer" |
| No Python WebRTC client | microduck/spaces/policy-playground/README.md:64 | "**`aiortc`, `av`, a DTLS cipher patch and `PyGObject`** existed to give Python a WebRTC stack." |
| UDP pad off by default | microduck/robotd-params/src/lib.rs:209 | "enabled: false," |
| BLE refuses teleop | microduck/btd/src/route.rs:237 | "RobotMove(_) \| RobotHead(_) \| RobotLook(_) \| RobotPose(_) \| RobotMouth(_) => false," |
| No authority arbitration | microduck/docs/design/remote-webrtc.md:466-469 | "two writers at 50 Hz interleaving into one slot," |
| Pad keeps sending | microduck/duck-ipc-proto/src/lib.rs:948-950 | "known stick value and keeps sending it at 50 Hz" |
| Sim sockets | microduck/docs/robot/simulation.md:80-82 | "the sockets are `~/.cache/duck-sim/duck-a.sock` (robotd; `duck.sock`" |
| Sim drive client | microduck/scripts/duck-sim:717-720 | "Resent at 10 Hz because `robotd` expires an intent 500 ms after" |
| Sim mediad only with cameras | microduck/scripts/duck-sim:465,481 | "--port $((8443 + i)) --web-port $((8080 + i)) \\" |
| Sim raw camera port | microduck_rl/src/mjlab_microduck/sim/body_server.py:569 | "frames = FrameServer((args.host, args.frame_port + index), FrameHandler)" |
| Sim unique frame rate | microduck_rl/src/mjlab_microduck/sim/body_server.py:470 | "passes_per_eye = max(1, round((1.0 / CAMERA_FPS) / period))" |
| Sim start pose hard-coded | microduck_rl/src/mjlab_microduck/sim/body_server.py:263-266 | "data.qpos[self.trunk + 3 : self.trunk + 7] = [1.0, 0.0, 0.0, 0.0]" |
| Sim ground truth | microduck_rl/src/mjlab_microduck/sim/body_server.py:323-333 | "# Where this duck is in the room. No robot knows that either" |
| Sim as a library | microduck_rl/scripts/pickup_e2e_robotd.py:25 | "class HandWorld(bs.World):" |
| Not native Windows | microduck/robotd/src/main.rs:51 | "use tokio::net::{UnixListener, UnixStream};" |
| No navigation planner | microduck/docs/design/robotd-design.md:1061 | "\| maploc — mapping, MCL, planning \| unowned \| — \|" |

---

# Second pass: corrections and additions


I spot-checked the brief's key claims against the code, and most of them hold. Both clones are shallow (one commit each), so I could not check anything against git history.

## Claims I confirmed (no change needed)
- **Wire format and socket paths:** NDJSON JSON-RPC, socket paths at `microduck/duck-ipc-proto/src/lib.rs:519-537`, and `API_VERSION = 44` at `lib.rs:473` "pub const API_VERSION: u32 = 44;". HEAD 942e58e carries the tags `daemon-v0.17.8` and `daemon-dev-main`.
- **Notification whitelist:** `microduck/robotd/src/main.rs:4489-4535`. It also includes `chorale.heard`, which doesn't matter for quackback.
- **Decoding `params`:** `lib.rs:1676-1679`.
- **stop / enable / look / init / relax dispatch:** `robotd/src/main.rs:5100-5131,5463-5520`.
- **No clamp on the twist:** `robotd/src/intents.rs:306-311`.
- **Deadman of 500 ms:** `duck-control/src/safety.rs:74`.
- **Sim ports:** body `7801+i`, raw camera `7901+index` (`microduck_rl/.../sim/body_server.py:569`), mediad `8080+i` / `8443+i` (`scripts/duck-sim:481`).
- **Sim camera rate:** the module uses `CAMERA_FPS=15`, which gives about 16.7 unique frames per second (`camera.py:33`, `body_server.py:470`).
- **Sim ToF statuses:** only 5 and 255 (`tof.py:25-26`).
- **H.264 `media.stream` frames are not rotated:** `mediad/src/main.rs:616` "The H.264 branch turns nothing".
- **BLE refusals:** `btd/src/route.rs:237,492,498,514`.
- **Socket mode:** 0660 `root:robot` (`robotd/systemd/robotd.service:48-49`).
- **Rendezvous limits:** `mediad/src/relay.rs:1019-1025`.

## Corrections
1. **An idle pad still writes to the robot, 10 times a second.** The brief says "padd resends at 50 Hz". That rate applies only while a stick asks for motion. An idle, connected pad resends a zero frame every 100 ms, and that frame carries velocity, head and pose. It will keep overwriting quackback's twist and head targets.
   - `microduck/pad-map/src/continuous.rs:22` "pub const HEARTBEAT: Duration = Duration::from_millis(100);"
   - `continuous.rs:12` "Ten a second says it as well."
   - `continuous.rs:1` "The continuous intents — velocity, head, pose"
   - Upstream's own fix is to stop padd: `docs/robot/cheatsheet-dev.md:121` "Stop the one on the robot first, or two processes fight over the sticks".
2. **The "pad lost, so sit down" rule applies to the Bluetooth pad too, not only to netpadd.** One second after a driving pad drops out, the duck sits. After that, `robot.move` is ignored until quackback sends `robot.do {"skill":"sit_toggle"}`. Check `robot.policies.sitting` (`lib.rs:2629`).
   - `pad-map/src/loss.rs:8` "Shared by `padd` (the Bluetooth pad dropping out) and `netpadd`"
   - `loss.rs:20` "pub const PAD_LOST_SIT: Duration = Duration::from_secs(1);"
3. **`/frame` timeout.** A capture must arrive within 500 ms, inside an overall 3 s bound; otherwise the request gets 503.
   - `mediad/src/pipeline.rs:316` "FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500)"
   - `mediad/src/snapshot.rs:9` "from_secs(3)"
4. **"Camera intrinsics are not available on path A/C" is wrong.** A Zero 3W IMX219 calibration is compiled in and used by mediad (`mediad/src/sensor.rs:99-100` "family: Some(robotd_params::CameraIntrinsics::alpha)").
   - Values: fx 1061.81, fy 1062.19, cx 596.78, cy 474.52, plus 5 distortion coefficients (`robotd-params/src/lib.rs:699-714`).
   - It applies to the 1280×720 frame before rotation (`lib.rs:692` "on the 1280×720 frame as `mediad` sends it (unrotated)").
   - A per-robot override can live in `[media.intrinsics]` in `/etc/robot/robotd.toml` (`robotd-params/src/lib.rs:662-668`).
   - The `/frame` PNG is rotated upright, so the matrix has to be rotated to match.
5. **Exact sim intrinsics and rotation direction.**
   - Sim: fx = fy ≈ 434.57, cx = 320, cy = 180 on the 640×360 frame before rotation (`mediad/src/camera.rs:242-256`, test at `:331-338`).
   - `rotate=90` means turn the image a quarter turn clockwise to make it upright (`uyvy/src/lib.rs:38` "A quarter turn clockwise: what this robot's camera mount needs.").
   - The sim uses the same mount (`mediad/src/main.rs:247` "a twin declares no board, so a frame from a duck in MuJoCo needs the same turn"). Raw port J frames therefore need that clockwise turn.
6. **duckctl on Windows is not dev-prerelease only.**
   - `docs/robot/duckctl.md:17` "On any machine with uv — Linux, macOS or Windows"
   - CI builds a Windows target: `.github/workflows/_build-duckctl.yml:49-51` "target: x86_64-pc-windows-msvc".
   - Whether PyPI actually has a 0.17.8 Windows wheel cannot be checked from the repo.
7. **Over BLE, not all of `system.*` is allowed.**
   - `btd/src/route.rs:210` "SystemPairingPin | SystemSetPairingPin(_) => false"
   - `robot.shutdown` and `robot.mode` are also refused (`route.rs:298,303`).
8. **The sim note "padd/netpadd (`--fake-pads`)" is misleading.**
   - `--fake-pads` is configd's flag for faking BlueZ (`configd/src/main.rs:68-74`, `scripts/duck-sim:326` "--fake-net --fake-pads").
   - padd and netpadd are simply never started by `up`.
9. **The split design is not anticipated for Windows specifically.** `docs/design/simulation.md:105` mentions "a Linux VM with MuJoCo on the host on macOS".
10. **`robot.state.policy` has more values than "walk", "stand" and "held".** The doc lists only those three (`lib.rs:3967`), but the code also emits:
    - `"sit"` and `"rise"` (`robotd/src/control.rs:632,636`);
    - `"limp_fall"` (`robotd/src/main.rs:3552`) and `"rest"` (`main.rs:3571`);
    - `"limp_pose"`, `"ground_pick"` and skill names.

    Use `robot.policies.homed` / `sitting` and `safety.picked_up` instead of matching on "held".
11. **mediad's full pipeline must run, not just build.** If `pipeline::start` fails (for example, webrtcsink missing in WSL), the whole daemon exits. That takes `/frame` and `media.frame` down with it, and the frame socket is bound only after the pipeline starts.
    - `mediad/src/main.rs:458` "mediad cannot start"
    - `main.rs:469` "cannot bind media.frame; refusing a partial start"
    - On Linux there is no feature switch: `mediad/Cargo.toml:7` "GStreamer is assumed on Linux".
12. **Body server concurrency.** Its docstring says one connection at a time: `body_server.py:392` "One connection at a time, which is the real relationship too". But it is a `ThreadingTCPServer` (`:434`), and duck-sim and duck-ether open extra read connections while robotd is connected (`duck-ether/src/main.rs:389-399`). Concurrent read-only polls do work.
13. **More doc/code conflicts for Q20.**
    - `mediad/webclient/index.html:532` "The robot clamps what it must" conflicts with the absence of any clamp (`intents.rs:306-311`).
    - `spaces/vision-demo/README.md:67` "Frames arrive upright" and the hello comment at `stream.rs:575-577` both conflict with `main.rs:303,616` (no rotation by default). The hello also carries a `"mount_rotate"` field (`stream.rs:579`).
14. **Q15 (version) cannot be answered from these clones.** Both are shallow. Only API-version notes exist:
    - v11 `tof.stream` (`lib.rs:121`)
    - v12 `robot.look` (`lib.rs:127`)
    - v24 `robot.model`, `frames`, `imu`, `t_ns` (`lib.rs:277-292`)
    - v30 `homed` (`lib.rs:340`)
    - v31 `sitting` (`lib.rs:352`)

    Feature-detect by sending requests: an unknown method returns −32601 naming the method (`lib.rs:66-69`).

## Missing for phases 0–4
- **A notification with bad params is silently dropped.** `robot/src/main.rs:4444` "if let Ok(call) = call {". A typo in a `robot.move` notification means the robot never moves and no error comes back. Send moves as requests in tests (they answer `{accepted:true}`, `main.rs:5100-5105`).
- **`hello` and `id: null` details.**
  - `hello` requires `api_version` (`lib.rs:2238-2240` "pub api_version: u32"), so `hello {}` gets −32602.
  - `"id": null` is decoded as no id, so the request is treated as a notification and gets no reply (`lib.rs:1986-1988`).
- **`robot.stop` does not latch.** `robotd/src/intents.rs:321-322` "self.set_twist([0.0; 3]);" is the same as `move {0,0,0}`, so any later move resumes walking. The abort path must kill quackback's own 10 Hz sender first.
- **A subscribed connection still serves requests, so replies must be demultiplexed.** Read replies by `id` and notifications by `method`.
  - `robotd/src/main.rs:4321` "Once set, the connection is both a request channel and a state stream"
  - Connections are concurrent: one task is spawned per connection (`main.rs:4296-4308`).
  - `hz` only thins what is sent (`main.rs:4388`); the per-tick build cost stays.
- **In the sim, heading from odometry is ground truth, so return-to-base will look better than on hardware.**
  - The body sends the MuJoCo trunk orientation as `imu.quat` (`body_server.py:304,337`).
  - It is passed through unchanged (`duck-control/src/sim.rs:283` "quat: frame.imu.quat") and fed to odometry (`robotd/src/main.rs:2414`).
  - On hardware, yaw is integrated IMU yaw (`odometry/src/lib.rs:19-21`).
- **The sim ToF sees the duck itself.** `microduck_rl/.../sim/tof.py:67` "Self-hits are reported, not filtered."; the ray cast excludes no body (`:88` "None, 1, -1, geom"). Drop returns under 0.10 m (`kinematics/src/tof.rs:99-101`).
- **Upstream reference for picking the nearest ToF return.** `kinematics/src/hand.rs:63-75`: near 0.10 m, far 0.70 m, at least 2 zones, statuses `[4,5,6,9,10,12,13]`, 250 ms hold. It is a good model for the PC-side 40 cm stop rule.
- **Waiting for the duck to stand.** Upstream waits 6 s after enable (`scripts/duck-sim:512` "sleep 6") and checks body `trunk_z > 0.095` (`:578`). The robot-side signal is `robot.policies.homed` (`lib.rs:2619`) together with the policy label.
- **Speed caps for Q9.** The microduck_rl velocity task trains vx ±0.4, vy ±0.3, vyaw ±1.0 (`microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:648-650`), and robotd-params says commands beyond that are "asking, not getting" (`robotd-params/src/lib.rs:124-126`). Whether the HF-deployed policy came from this task cannot be proven from the repos.
- **The sim mediad listens on all interfaces with no auth.** `mediad/src/main.rs:32` `default_value = "0.0.0.0"`, and duck-sim passes no `--host`. The body server binds `127.0.0.1`.
- **`scripts/duck-sim ctl frame` fails as shipped.**
  - `ctl` passes no `--media-socket` (`scripts/duck-sim:91-96`).
  - robotctl defaults to `/run/mediad/media.sock` (`robotctl/src/main.rs:117-118`).
  - Pass `--media-socket ~/.cache/duck-sim/duck-a-frame.sock` explicitly.
- **`media.frame` socket details.**
  - `hello` is optional (`mediad/src/frame.rs:163-178`).
  - Each connection gets one frame and then closes (`:223-226`).
  - At most 16 clients at once and a 5 s lifetime (`:100,110-112`).
- **`uv sync` in microduck_rl is heavy, though the body server needs little.**
  - It pins `"torch==2.9.1"` (`pyproject.toml:37`), whose x86_64 wheel pulls CUDA dependencies (`:99-100`), plus a git dependency on Rhoban/bam (`:89`).
  - `body_server` imports only mujoco and numpy (`body_server.py:42-53`).
  - duck-sim still requires the full `.venv` and its onnxruntime (`scripts/duck-sim:222-228`).
- **A movable person proxy in the sim.**
  - A launcher can use a mocap body in a custom World, as upstream does: `microduck_rl/scripts/pickup_e2e_robotd.py:25` "class HandWorld(bs.World):" and `:50` "m.body_mocapid".
  - `DUCK_SIM_SCENE=vslam` is an open room of about 11×11 m (`vslam_room.xml:1-4`).
- **Precedent for PC-side audio with duck mouth animation.**
  - `duckctl/src/udp_pad.rs:12-13` "The quack and the wheee play from this laptop".
  - The `SENDER_SOUNDS` flag tells the duck to stay quiet but keep the mouth (`pad-map/src/wire.rs:27-30`).

## Paths and APIs that exist but are absent from the brief
- **`robotd --fake`**, a robot made of nothing, with no physics. It allows fast CI of the backend without MuJoCo, still on Linux or macOS only.
  - `robotd/src/main.rs:248-251` "Run against a robot made of nothing."
  - `docs/robot/simulation.md:185-187`
- **`spaces/vision-demo`, a Python receiver for `media.stream`** (FastAPI WebSocket, PyAV H.264 decode, OpenCV). It is the reference client for path D.
  - `README.md:92` "DUCK_RECEIVER=ws://192.168.1.50:7860/frames uv run app.py"
  - `receiver.py:1-12`
  - It asks for the stream over the rendezvous lane, and the robot sends its HF token to it.
- **`head_imu.stream`** (`lib.rs:992`) and `robot.state.frames.head_imu`.
- **`duckctl ssh`, `scp` and `open`.** They find the duck's IP from the BLE advertisement and then run ssh, scp or the browser. The default user is `microduck` or `DUCK_BOARD_USER` (`duckctl/src/main.rs:982,993,1036`). Useful for the launch docs.
- **`robotctl frame --output`** saves one raw UYVY frame plus its geometry (`docs/robot/cheatsheet.md:985`).
- **mediad `--no-remote`** (`mediad/src/main.rs:71`) keeps a signed-in sim or robot off the HF robot listing.
- **`media.video` pushed at channel open** as a best-effort notification (`mediad/src/session.rs:82`).

---

# The simulator website (what quackback drives today)

https://huggingface.co/spaces/pollen-robotics/microduck-simulator ("Microduck Sandbox", Apache-2.0,
source read at Space revision ad9d41a): MuJoCo compiled to WebAssembly and the real trained ONNX
policies (onnxruntime-web), at 50 Hz, entirely in the browser. It is **not** the robot runtime:
no robotd, no JSON-RPC, no sockets.

What a PC program can use, all in `app/src/game/game.js` and `app/src/store.js`:

- `window.rl`, "deterministic hooks for automated verification": `data` (MuJoCo state;
  `qpos[0..6]` is the duck's free joint `[x, y, z, qw, qx, qy, qz]`), `controller`, `mode`,
  `inputLocked`, `resetSim`, `camera`, `controls`, `chaseCam`, triggers for kicks and the like.
- `window.__store` / `window.__gameApi`, "debug handles for the console / automated QA".
- The input `Controller` (`controls/controller.js`) takes pluggable sources in priority order
  (`addSource`). The first `isActive()` source owns the `[vx, vy, wz]` twist (m/s, rad/s, +wz
  turns left); actions such as `quack` go through the source's `onAction`.
- `?boot=1` skips the title screen; the twist is gated until the entrance ceremony ends
  (`rl.inputLocked` turns false), about 25 s after the page opens.

quackback (`robot/sandbox.py`) adds itself as the last-priority source with a 500 ms deadman,
reads the pose from `qpos`, and refuses the page's WebSockets, which are only its multiplayer
relays (public Nostr servers).

Measured with the page's walking policy (`BEST_alpha_walking.onnx`), 4 s per command, headless
Edge:

| Command | Result |
|---|---|
| vx 0.15 or 0.20 m/s | stands still |
| vx 0.25 m/s (the page's own keyboard speed) | walks at about 0.09-0.10 m/s |
| wz 0.3-1.0 rad/s, vx 0 | barely turns (under 4 deg/s) |
| vx 0.25 + wz 0.5 / 1.0 | arcs at 15 / 33 deg/s |
| vx -0.2 | does not walk back |

So in the simulator the duck walks only at 0.25 m/s and steers in arcs (a turning radius of
about 0.17 m: 33 deg/s at about 0.1 m/s); `GAIT` in `robot/sandbox.py` says exactly that, and MicroduckBackend steers
accordingly. The arena is 3 x 3 m (walls at +-1.5 m); the duck spawns at (-0.6, 0) facing +x.

## The duck's eyes in the simulator page

The page has no camera feed and no depth sensor for a program to read, and no people in it.
quackback adds them in the page itself (`robot/sandbox_eyes.js`), from the robot model's own
MJCF sites, so that the PC-side code sees what it would see on a real duck:

- **Camera:** the scene rendered from the `head_camera` site into an offscreen target, portrait
  360 x 640 with a 72.7 degree vertical field of view (the twin's camera, upright), JPEG over
  Playwright. The site, not MuJoCo's `<camera>`: that one faces backwards in this model.
  About 20-30 ms a picture; quackback takes at most ten a second.
- **Depth sensor:** 64 rays from the `tof` site over 45 x 45 degrees, laid out like
  microduck_rl's `sim/tof.py` (row 0 at the top, column 0 on the left, slant range in mm,
  status 5 or 255, 4 m at most). They hit the walls, props, the ball, the floor and the person,
  never the duck itself. It sits about 8 cm ahead of the body's centre.
- **Head:** `rl.headTarget` takes head commands that the walking policy tracks while walking;
  no head mode is needed. Measured camera pitch for `head_pitch` -0.6 / -0.9 / -1.2 / -2.5:
  +12 / +23 / +30 / +47 degrees (-11 at rest). `head_yaw` 0.5 turns the camera 28 degrees at
  rest pitch, but about twice as far with the head up, so quackback aims the head by where the
  camera really points.
- **Person:** a picture standing on the floor, turned toward the duck before every frame and
  every depth reading; its transparent parts let the depth rays through.

What the duck does with them is in `robot/microduck.py`. It looks around with its head while
standing, and turns its body (in an arc here) only toward a side the depth sensor has just seen
clear. The detector (MediaPipe EfficientDet-Lite0, person class only) gives a bearing, and the
depth zones in that direction give the range, once the head has put the person in the middle of
the picture. Every walk stops for anything the depth sensor sees within 50 cm in the duck's way.
The walk up to the person ends at 55 cm by the sensor, plus the turning radius for a duck that
turns only in arcs (72 cm here, about 65 cm once it has coasted to a stop): walking away again,
its first half circle takes it up to about 15 cm closer to them. So before it turns round, it
looks about with its depth sensor and walks only an arc that keeps its body 50 cm from
everything it saw, turning the other way, or staying put, when the shorter way is too close. Measured in the page: the walk
coasts 5-8 cm after a stop, and the body keeps turning a few degrees for about a second.
