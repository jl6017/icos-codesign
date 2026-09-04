"""Locomotion frames of a trained S2R policy on a robot dir (flat or rough), rendered with the Fig.-2 camera.
  python -u icos_v2/analysis/render_gait_frames.py --robot_dir <dir> --checkpoint <best dir> --steps 150,350 --out prefix
"""
import argparse, glob, os, re, sys
import numpy as np
import jax, jax.numpy as jp
import mujoco
from PIL import Image
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())
from icos_v2.learning import icos_constants as consts


def load_policy(ckpt, env):
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    import brax.training.checkpoint as brax_checkpoint
    p = os.path.abspath(ckpt)
    subs = sorted([d for d in glob.glob(os.path.join(p, "*")) if os.path.isdir(d) and os.path.basename(d).isdigit()], key=lambda d: int(os.path.basename(d)))
    if subs: p = subs[-1]
    params = brax_checkpoint.load(p)
    net = ppo_networks.make_ppo_networks(observation_size=env.observation_size, action_size=env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize,
                                         policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
                                         policy_obs_key="state", value_obs_key="privileged_state")
    return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--robot_dir", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", default="150,350"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distance", type=float, default=1.15); ap.add_argument("--elevation", type=float, default=-22); ap.add_argument("--azimuth", type=float, default=135)
    a = ap.parse_args()
    rdir = os.path.abspath(a.robot_dir); scene = glob.glob(os.path.join(rdir, "scene_mjx_*_flat_terrain.xml"))[0]
    name = re.search(r"scene_mjx_(.+)_flat_terrain\.xml$", os.path.basename(scene)).group(1)
    sensor = glob.glob(os.path.join(rdir, f"sensor_{name}_*feet.xml"))
    feet = sorted({int(f) for f in re.findall(r'foot_(\d+)_floor_found', open(sensor[0]).read())}) if sensor else []
    consts.MJCF_DIR = rdir; consts.CONFIGS[name] = feet
    from icos_v2.learning.velocity_omni_s2r import VelocityOmniS2R
    env = VelocityOmniS2R(config_name=name)
    policy = load_policy(a.checkpoint, env)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(a.seed); state = reset(rng)
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640); opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
    want = {int(s) for s in a.steps.split(",")}
    for t in range(max(want) + 1):
        if t in want:
            d.qpos[:] = np.asarray(state.data.qpos); d.qvel[:] = np.asarray(state.data.qvel); mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera(); cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.10]; cam.distance = a.distance; cam.elevation = a.elevation; cam.azimuth = a.azimuth
            r.update_scene(d, cam, opt); Image.fromarray(r.render()).save(f"{a.out}_t{t}.png"); print(f"→ {a.out}_t{t}.png  xy=({d.qpos[0]:.2f},{d.qpos[1]:.2f})")
        rng, k = jax.random.split(rng); act, _ = policy(state.obs, k); state = step(state, act)
    r.close()


if __name__ == "__main__":
    main()
