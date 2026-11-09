"""``nerve`` command line: the whole pipeline, one verb each."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _spec(path: str):
    from .spec import load_spec
    return load_spec(path)


def cmd_check(a):
    from .spec import Oracle
    sp = _spec(a.spec); o = Oracle(sp)
    probs = o.self_check()
    print(json.dumps({"spec": sp.name, "states": sp.state_space_size(), "actions": sp.action_width(),
                      "finite": sp.finite, "holes": list(sp.holes), "problems": len(probs)}, indent=1))
    for p in probs[:20]:
        print("  !", p)
    return 1 if probs else 0


def cmd_compile(a):
    from .spec.circuit import compile_circuit
    from .spec.encode import Encoder
    sp = _spec(a.spec); c = compile_circuit(sp, exclude_actions={h.action for h in sp.holes.values()})
    enc = Encoder(sp)
    print(json.dumps({"spec": sp.name, "hidden_units": int(c.ws.shape[1]), "state_dim": enc.state_dim,
                      "action_dim": enc.action_dim, "params_computed": int(c.ws.numel() + c.wa.numel() + c.wo.numel())}))
    if a.out:
        torch.save(c.state_dict(), a.out)


def cmd_rollout(a):
    from .data import EpisodeLog, rollout_dataset
    sp = _spec(a.spec)
    log = EpisodeLog(a.log)
    n = 0
    for pol in a.policy:
        n += rollout_dataset(sp, log, episodes=a.episodes, horizon=a.horizon, policy=pol, negatives=a.negatives, seed=a.seed)
    print(json.dumps({"spec": sp.name, "log": a.log, "positives_added": n}))


def cmd_train(a):
    from .data import EpisodeLog
    from .train import TrainConfig, train
    sp = _spec(a.spec)
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, d=a.d, hidden=a.hidden, holdout=a.holdout, seed=a.seed,
                      compile_circuit=not a.no_circuit, init=a.init, device=a.device, weight_decay=a.weight_decay, ord_weight=a.ord_weight, fact_weight=a.fact_weight,
                      log_every=a.log_every, holdout_salt=a.holdout_salt)
    if a.attach:
        from .train.remote import wait_and_fetch
        j = json.loads(Path(a.attach).read_text()) if Path(a.attach).exists() else {"job": a.attach.split(":")[0], "run": a.attach.split(":")[1]}
        print(json.dumps(wait_and_fetch(j["job"], j["run"], a.out, j.get("repo", "AlexWelcing/nerve-runs")))); return 0
    if a.remote:
        from .train.remote import train_remote
        print(json.dumps(train_remote(a.spec, a.log, a.out, cfg, flavor=a.flavor))); return 0
    r = train(sp, EpisodeLog(a.log), a.out, cfg)
    print(json.dumps({"score": r["score"], "all": r["all"]["exact"], "heldout": r["heldout"]["exact"],
                      "learned_only": r["learned_only"]["exact"], "violations": r["all"]["violation_rate"],
                      "holes": r.get("hole_examples"), "params": r["params"], "seconds": r["seconds"]}))


def cmd_eval(a):
    from .model import NerveEnergy
    from .train import evaluate
    sp = _spec(a.spec); m = NerveEnergy.load(sp, a.model)
    print(json.dumps(evaluate(sp, m.step, a.holdout, learned_only_fn=m.step_learned_only, salt=a.holdout_salt), indent=1))


def cmd_patch(a):
    from .sync import encode_patch
    prev = torch.load(a.prev, map_location="cpu")["jepa"]; new = torch.load(a.new, map_location="cpu")["jepa"]
    p = encode_patch(prev, new, level=a.level)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); p.save(a.out)
    full = sum(v.numel() * 2 for v in new.values() if v.is_floating_point())
    print(json.dumps({"patch": a.out, "bytes": len(p.data), "full_bf16_bytes": full, "ratio": round(full / len(p.data), 2),
                      "sparsity": round(p.sparsity, 4), "digest": p.digest}))


def cmd_serve(a):
    from .serve import NerveRuntime
    sp = _spec(a.spec)
    rt = NerveRuntime(sp, a.model, a.patches)
    rt.serve(a.host, a.port)
    print(json.dumps({"serving": f"http://{a.host}:{a.port}", "spec": sp.name, "patches": rt.applied}), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def cmd_step(a):
    from .serve import NerveRuntime
    sp = _spec(a.spec); rt = NerveRuntime(sp, a.model, a.patches)
    s = json.loads(a.state) if a.state else dict(sp.initial)
    for act in a.action:
        name, _, params = act.partition(":")
        out = rt.step(s, (name, json.loads(params) if params else {}))
        print(json.dumps({"action": act, **out}))
        s = out["state"]


def cmd_export(a):
    from .model import NerveEnergy
    from .serve.export import export_edge
    sp = _spec(a.spec); m = NerveEnergy.load(sp, a.model)
    print(json.dumps(export_edge(sp, m, a.out)))


def cmd_edge(a):
    """Export to edge/export/<spec>/, bundle, and serve the Worker on workerd locally."""
    from .model import NerveEnergy
    from .serve.export import export_edge, write_patch_index
    from .serve import workerd as wd
    sp = _spec(a.spec); m = NerveEnergy.load(sp, a.model)
    out = wd.export_dir(sp.name); out.mkdir(parents=True, exist_ok=True)
    export_edge(sp, m, out / "edge.json")
    if a.patches:
        import shutil as _sh
        (out / "patches").mkdir(exist_ok=True)
        for p in Path(a.patches).glob("*.nrvp"):
            _sh.copy(p, out / "patches" / p.name)
        write_patch_index(out / "patches")
    if not wd.available():
        print(json.dumps({"exported": str(out), "workerd": "not installed (cd edge && npm install)"})); return 0
    if a.no_serve:
        print(json.dumps({"exported": str(out), "bundle": str(wd.bundle())})); return 0
    with wd.Workerd(a.port) as w:
        print(json.dumps({"serving": w.base, "spec": sp.name, "try": f"curl -X POST {w.base}/{sp.name}/session/me -d '{{\"action\":\"{next(iter(sp.actions))}\"}}'"}), flush=True)
        try:
            while w.proc and w.proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def cmd_render(a):
    """render-build | render-train | render-sheet for the neural field renderer."""
    from .render import RenderSet
    sp = _spec(a.spec)
    if a.rcmd == "build":
        from .render.dataset import build
        from .render.templates import TEMPLATES
        print(json.dumps(build(sp, TEMPLATES[sp.name], a.out, a.width, a.height, a.scale, a.holdout)))
    elif a.rcmd == "train":
        from .render.train import RenderTrainConfig, train_render
        cfg = RenderTrainConfig(steps=a.steps, hidden=a.hidden, layers=a.layers, octaves=a.octaves, lr=a.lr, device=a.device)
        if a.remote:
            from .render.remote import render_remote
            print(json.dumps(render_remote(a.spec, a.data, a.out, cfg, flavor=a.flavor))); return 0
        r = train_render(sp, RenderSet(a.data), a.out, cfg)
        print(json.dumps({k: r[k] for k in ("train", "heldout", "seconds", "params")}))
    elif a.rcmd == "sheet":
        from .render.sheet import contact_sheet
        from .render.train import load_field
        print(json.dumps(contact_sheet(sp, RenderSet(a.data), load_field(a.model), a.out, upscale=a.upscale)))


def cmd_shader(a):
    """shader-train: train the engine's shader from a `nerve rasters` dump, locally or on a Hugging Face GPU."""
    from .render.shader import Rasters, ShaderTrain, train_shader
    t = ShaderTrain(steps=a.steps, pixels=a.pixels, batch_states=a.batch, lr=a.lr, seed=a.seed, log_every=a.log_every, device=a.device)
    arch = {"cond": a.cond, "hidden": a.hidden, "layers": a.layers, "octaves": a.octaves}
    if a.remote:
        from .render.shader_remote import shader_remote
        print(json.dumps(shader_remote(a.spec_name, a.rasters, a.out, t, arch, flavor=a.flavor))); return 0
    rs = Rasters(a.rasters)
    print(json.dumps(train_shader(a.spec_name, rs, rs.config(**arch), t, a.out)))


def cmd_evolve(a):
    from .evolve import Supervisor
    sup = Supervisor(a.spec, a.workdir, budget_usd=a.budget, iterations=a.iterations, mode=a.mode,
                     model=a.supervisor_model, role_model=a.role_model, remote=a.remote, flavor=a.flavor)
    print(json.dumps(sup.run(), indent=1))


def cmd_drag(a):
    """Run a gesture along a pointer path: one energy minimisation per frame over spec-derived futures."""
    from .spec.drag import DragSession
    sp = _spec(a.spec)
    s = json.loads(a.state) if a.state else dict(sp.initial)
    energy = None
    if a.model:
        from .model import NerveEnergy
        m = NerveEnergy.load(sp, a.model); enc = m.enc
        def energy(start, action, cand):   # the learned term breaks ties where the spec is silent
            if action is None or not sp.finite:
                return 0.0
            E = m.energies(start, action); return float(E[enc.state_id(cand)] - E.min())
    sess = DragSession(sp, sp.futures[a.gesture], s, energy)
    for pt in a.path:
        x, y = (float(v) for v in pt.split(","))
        fr = sess.frame((x, y))
        print(json.dumps({"pointer": [x, y], "branch": fr.branch, "gap": round(fr.gap, 2), "far": fr.far,
                          "preview": fr.preview, **({"weights": fr.weights} if fr.weights else {})}))
    print(json.dumps({"drop": sess.release()}))


def cmd_intake(a):
    from .intake import intake
    out = intake(a.source, kind=a.kind, model=a.model, out=a.out)
    print(json.dumps(out, indent=1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nerve", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="type/invariant check a spec against its own oracle"); s.add_argument("spec"); s.set_defaults(f=cmd_check)
    s = sub.add_parser("compile", help="compile the spec's explicit transitions to exact weights"); s.add_argument("spec"); s.add_argument("--out"); s.set_defaults(f=cmd_compile)
    s = sub.add_parser("rollout", help="generate trajectories + spec-derived negatives into the log")
    s.add_argument("spec"); s.add_argument("--log", required=True); s.add_argument("--episodes", type=int, default=50)
    s.add_argument("--horizon", type=int, default=20); s.add_argument("--negatives", type=int, default=3)
    s.add_argument("--policy", nargs="+", default=["exhaustive", "coverage", "random", "adversary"]); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_rollout)
    s = sub.add_parser("train", help="train the JEPA energy from the log")
    s.add_argument("spec"); s.add_argument("--log", required=True); s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int, default=60); s.add_argument("--lr", type=float, default=3e-3); s.add_argument("--d", type=int, default=64)
    s.add_argument("--hidden", type=int, default=128); s.add_argument("--holdout", type=float, default=0.15); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-circuit", action="store_true"); s.add_argument("--init", help="warm-start from checkpoint")
    s.add_argument("--device", default="auto", help="auto|cpu|cuda|mps"); s.add_argument("--weight-decay", type=float, default=1e-4); s.add_argument("--ord-weight", type=float, default=1.0); s.add_argument("--fact-weight", type=float, default=10.0)
    s.add_argument("--log-every", type=int, default=10, help="evaluate every N epochs (the eval dominates CPU time on larger specs; raise it)")
    s.add_argument("--holdout-salt", default="nerve", help="salt of the hold-out hash: a different salt holds out different cells")
    s.add_argument("--remote", choices=["hf"], help="run the training as a Hugging Face Job on a GPU (needs HF_TOKEN)")
    s.add_argument("--flavor", default="a10g-small", help="HF Jobs hardware flavor for --remote hf")
    s.add_argument("--attach", help="attach to a running HF job: path to job.json or <job_id>:<run>"); s.set_defaults(f=cmd_train)
    s = sub.add_parser("eval", help="official conformance eval"); s.add_argument("spec"); s.add_argument("--model", required=True); s.add_argument("--holdout", type=float, default=0.15); s.add_argument("--holdout-salt", default="nerve"); s.set_defaults(f=cmd_eval)
    s = sub.add_parser("patch", help="PULSE sparse patch between two checkpoints"); s.add_argument("--prev", required=True); s.add_argument("--new", required=True); s.add_argument("--out", required=True); s.add_argument("--level", type=int, default=1); s.set_defaults(f=cmd_patch)
    s = sub.add_parser("serve", help="run an inference worker"); s.add_argument("spec"); s.add_argument("--model", required=True); s.add_argument("--patches"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8787); s.set_defaults(f=cmd_serve)
    s = sub.add_parser("step", help="drive the network from the shell"); s.add_argument("spec"); s.add_argument("--model", required=True); s.add_argument("--patches"); s.add_argument("--state"); s.add_argument("action", nargs="+", help="name or name:{json params}"); s.set_defaults(f=cmd_step)
    s = sub.add_parser("export", help="export weights for the Cloudflare edge worker"); s.add_argument("spec"); s.add_argument("--model", required=True); s.add_argument("--out", required=True); s.set_defaults(f=cmd_export)
    s = sub.add_parser("edge", help="export + run the Cloudflare Worker locally on workerd"); s.add_argument("spec"); s.add_argument("--model", required=True)
    s.add_argument("--patches"); s.add_argument("--port", type=int, default=8788); s.add_argument("--no-serve", action="store_true"); s.set_defaults(f=cmd_edge)
    for rc in ("build", "train", "sheet"):
        s = sub.add_parser(f"render-{rc}", help=f"neural field renderer: {rc}"); s.add_argument("spec"); s.set_defaults(f=cmd_render, rcmd=rc)
        s.add_argument("--out", required=True)
        if rc == "build":
            s.add_argument("--width", type=int, default=560); s.add_argument("--height", type=int, default=360); s.add_argument("--scale", type=int, default=1); s.add_argument("--holdout", type=float, default=0.2)
        else:
            s.add_argument("--data", required=True)
        if rc == "train":
            s.add_argument("--steps", type=int, default=4000); s.add_argument("--hidden", type=int, default=256); s.add_argument("--layers", type=int, default=6)
            s.add_argument("--octaves", type=int, default=10); s.add_argument("--lr", type=float, default=1e-3); s.add_argument("--device", default="auto")
            s.add_argument("--remote", choices=["hf"]); s.add_argument("--flavor", default="a10g-small")
        if rc == "sheet":
            s.add_argument("--model", required=True); s.add_argument("--upscale", type=int, default=2)
    s = sub.add_parser("shader-train", help="train the engine's shader (GPU twin of `nerve train-shader`) from a `nerve rasters` dump")
    s.add_argument("spec_name"); s.add_argument("--rasters", required=True); s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=20000); s.add_argument("--pixels", type=int, default=4096); s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3); s.add_argument("--seed", type=int, default=0); s.add_argument("--log-every", type=int, default=1000); s.add_argument("--device", default="auto")
    s.add_argument("--cond", type=int, default=128); s.add_argument("--hidden", type=int, default=128); s.add_argument("--layers", type=int, default=4); s.add_argument("--octaves", type=int, default=8)
    s.add_argument("--remote", choices=["hf"]); s.add_argument("--flavor", default="a10g-small"); s.set_defaults(f=cmd_shader)
    s = sub.add_parser("evolve", help="EAT-style supervisor loop over team configs"); s.add_argument("spec"); s.add_argument("--workdir", required=True)
    s.add_argument("--budget", type=float, default=5.0); s.add_argument("--iterations", type=int, default=4); s.add_argument("--mode", choices=["offline", "claude"], default="offline")
    s.add_argument("--supervisor-model", default="claude-fable-5-1"); s.add_argument("--role-model", default="claude-sonnet-5-5")
    s.add_argument("--remote", choices=["hf"], help="train each iteration as a Hugging Face GPU Job"); s.add_argument("--flavor", default="a10g-small"); s.set_defaults(f=cmd_evolve)
    s = sub.add_parser("drag", help="simulate a gesture: pointer path -> per-frame preview, gap, branch; drop on release")
    s.add_argument("spec"); s.add_argument("--gesture", required=True); s.add_argument("--state"); s.add_argument("--model", help="add a trained network's learned energies as tie-breaker")
    s.add_argument("path", nargs="+", help="pointer positions x,y in render coordinates"); s.set_defaults(f=cmd_drag)
    s = sub.add_parser("intake", help="code | screenshot | prose -> behavior spec (via a harness agent)"); s.add_argument("source"); s.add_argument("--kind", choices=["auto", "code", "image", "prose"], default="auto")
    s.add_argument("--model", default="claude"); s.add_argument("--out", required=True); s.set_defaults(f=cmd_intake)

    a = p.parse_args(argv)
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
