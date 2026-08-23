#!/usr/bin/env python3
"""v7 审计的变异测试：对源码打入单点缺陷，看测试套件是否捕获。"""
import subprocess, sys
from pathlib import Path

# Derived from this file's own location, never hard-coded. Until v17.1 it
# was an absolute path to the build directory of the round that created
# the script, and it was not re-pointed when the tree was copied forward:
# the sweep shipped inside v17.0 mutated caide_pkg16 and ran that tree's
# tests, then printed "0 escaped" and "source tree verified". A run
# against the wrong subject reads exactly like a run against the right
# one -- the same failure as a partial run reading like a complete one,
# one level worse.
PKG = Path(__file__).resolve().parent.parent
SRC = PKG / "src" / "caide"

MUTANTS = [
    # (编号, 说明, 文件, 原文, 变异)
    ("M1", "roofline 取 min", "roofline.py",
     "step = max(memory_time, compute_time)",
     "step = min(memory_time, compute_time)"),
    ("M2", "KV 漏乘 batch", "roofline.py",
     "kv_bytes = model.kv_bytes_per_token * context_length * batch",
     "kv_bytes = model.kv_bytes_per_token * context_length"),
    ("M3", "GEMM 漏乘 batch", "roofline.py",
     "gemm_flops = 2.0 * model.active_params * batch * verified_tokens",
     "gemm_flops = 2.0 * model.active_params * verified_tokens"),
    ("M4", "v7 验证因子移除", "roofline.py",
     "verified_tokens = cfg.speculative_gamma + 1.0 if speculating else 1.0",
     "verified_tokens = 1.0"),
    ("M5", "v7 验证MFU按batch行数", "roofline.py",
     "flops = _achievable_flops(hw, cfg, _decode_mfu(cfg, batch * verified_tokens))",
     "flops = _achievable_flops(hw, cfg, _decode_mfu(cfg, batch))"),
    ("M6", "v7 草稿显存重新免费", "roofline.py",
     "headroom = total_memory - model.weight_bytes - _draft_weight_bytes(model, cfg)",
     "headroom = total_memory - model.weight_bytes"),
    ("M7", "v7 空闲按满载功率", "costing.py",
     "idle_joules = (state.hardware.resolved_idle_power\n                   * accel_seconds * (1.0 / duty - 1.0))",
     "idle_joules = (state.hardware.power_watts\n                   * accel_seconds * (1.0 / duty - 1.0))"),
    ("M8", "v7 能量阶梯跳过", "costing.py",
     "    replicas = max(math.ceil(capacity_units - 1e-9),\n                   state.serving.min_replicas)",
     "    replicas = max(capacity_units,\n                   state.serving.min_replicas)"),
    ("M9", "v7 掉队者回到整基数乘法", "roofline.py",
     "memory_time += (expert_traffic / bandwidth) * stretch",
     "memory_time *= (1.0 + stretch)"),
    ("M10", "v7 不均衡下限退化 peak=mean", "roofline.py",
     "peak = max(mean + math.sqrt(2.0 * mean * math.log(n_experts)), 1.0)",
     "peak = max(mean, 1.0)"),
    ("M11", "v7 预填充再次被排除", "calibration.py",
     "cycle = perf.prefill_seconds + perf.decode_seconds",
     "cycle = perf.decode_seconds"),
    ("M12", "验证步 all-reduce 与 token 脱钩", "roofline.py",
     "collective = _collective_seconds(model, hw, cfg, batch * verified_tokens)",
     "collective = _collective_seconds(model, hw, cfg, batch)"),
    ("M13", "副本取整用 round", "costing.py",
     "    replicas = max(math.ceil(capacity_units - 1e-9), state.serving.min_replicas)",
     "    replicas = max(round(capacity_units - 1e-9), state.serving.min_replicas)"),
    ("M14", "占空比在美元中被忽略", "costing.py",
     "raw_compute = accel_seconds / SECONDS_PER_HOUR * hourly / duty",
     "raw_compute = accel_seconds / SECONDS_PER_HOUR * hourly"),
    ("M16", "v8 路由token数退回 batch", "roofline.py",
     "    weight_bytes = model.decode_weight_bytes(routed_tokens)",
     "    weight_bytes = model.decode_weight_bytes(batch)"),
    ("M17", "v8 需求响应退回 token 价", "scaling.py",
     "                effective = _effective_unit_cost(cost, assumptions, volume)",
     "                effective = cost"),
    ("M18", "v8 价不敏项从支出中丢失", "scaling.py",
     "            inelastic_spend=(assumptions.price_inelastic_per_query * volume),",
     "            inelastic_spend=0.0,"),
    ("M19", "v8 层弹性用前向差分且不取对数", "costing.py",
     "            out[k] = (math.log(b) - math.log(a)) / dlnv",
     "            out[k] = (b - a) / a"),
    ("M20", "v8 饱和不再上报", "perturb.py",
     "    if rr != 1.0 and any(w.review_rate * rr > 1.0 for w in scenario.workloads):",
     "    if False:"),
    ("M21", "v8 草稿KV项被静默丢弃", "roofline.py",
     "        if cfg.draft_kv_ratio > 0:",
     "        if False:"),
    ("M22", "v9 打平带回到首末点span", "breakeven.py",
     "                if len(current) >= 2:\n                    runs.append((current[0], current[-1]))\n                current = []",
     "                current = current\n                pass"),
    ("M23", "v9 max_share 再次被忽略", "routing.py",
     "        feasible = [t for t in tiers if t.serves(w)\n                    and used[t.name] + w.share <= t.max_share + 1e-12]",
     "        feasible = [t for t in tiers if t.serves(w)]"),
    ("M24", "v9 非可分离层退回边际计价", "routing.py",
     "    if tier.annual_cost_fn is not None:\n        return tier.annual_cost_fn(served, annual_volume)",
     "    if False:\n        return tier.annual_cost_fn(served, annual_volume)"),
    ("M25", "v9 复核时长扰动被丢弃", "perturb.py",
     "                    review_minutes=w.review_minutes * rm)",
     "                    review_minutes=w.review_minutes)"),
    ("M26", "v9 复核工资扰动被丢弃", "perturb.py",
     "            reviewer_hourly_cost=assurance.reviewer_hourly_cost * wage)",
     "            reviewer_hourly_cost=assurance.reviewer_hourly_cost)"),
    ("M27", "v9 margin 退回均值分母", "breakeven.py",
     "            denom = max(min(abs(cost_a(x)), abs(cost_b(x))), 1e-12)",
     "            denom = max(abs(0.5 * (cost_a(x) + cost_b(x))), 1e-12)"),
    ("M28", "v10 质量下限不再检查", "costing.py",
     "    quality_violations = [w.name for w in workloads\n                          if w.quality_floor > quality + 1e-12]",
     "    quality_violations = []"),
    ("M29", "v10 摘要漏掉复核工资", "report.py",
     "                      a.incident_response_annual, a.reviewer_hourly_cost,",
     "                      a.incident_response_annual,"),
    ("M30", "v10 量化重新统一施加", "efficiency.py",
     "        head = max(QUANTISATION_HEAD_BYTES, bytes_per_param)",
     "        head = bytes_per_param"),
    ("M31", "v10 输入嵌入重新计入流式", "specs.py",
     "        return max(streamed - gathered * self.head_bytes, 0.0)",
     "        return streamed"),
    ("M32", "v10 分布参数不再记录", "uncertainty.py",
     "                        {\"median\": median, \"sigma\": sigma})",
     "                        {})"),
    ("M33", "v10 饱和不再进入叙述", "scaling.py",
     "        if self.saturated_from is not None:",
     "        if False:"),
    ("M34", "v10 cheapest 退回全体最小", "report.py",
     "        feasible = {k: v for k, v in self.tco.items() if v.feasible}\n        pool = feasible or self.tco",
     "        pool = self.tco"),
    ("M35", "v11 apply_stack 不再施加质量变化", "efficiency.py",
     "    delta = stack_quality_delta(keys)\n    if delta:",
     "    delta = stack_quality_delta(keys)\n    if False:"),
    ("M36", "v11 蒸馏质量常数复活（双计）", "efficiency.py",
     "            apply=_distil(0.5),",
     "            apply=_distil(0.5), quality_delta=-0.05,"),
    ("M37", "v11 API 的 SLO 退回无条件通过", "costing.py",
     "        slo_met=None,          # not modelled for a commercial endpoint",
     "        slo_met=True,"),
    ("M38", "v11 latency_sensitive 再次被忽略", "costing.py",
     "            if not qc.slo_met and w.latency_sensitive:",
     "            if not qc.slo_met:"),
    ("M39", "v11 供应商能耗退回硬编码", "costing.py",
     "                provider_energy_wh_per_ktok=provider_energy_wh_per_ktok)",
     "                provider_energy_wh_per_ktok=0.30)"),
    ("M40", "v11 残差吸收器标记被抹去", "calibration.py",
     "    \"residual_absorber\": True,",
     "    \"residual_absorber\": False,"),
    ("M41", "v12 失败关联不再计算", "uncertainty.py",
     "        fail_rho = (_spearman(values, failed.astype(float))\n                    if any_failed else math.nan)",
     "        fail_rho = math.nan"),
    ("M42", "v12 regime 退回点估计", "scaling.py",
     "    if not math.isfinite(half) or (ci_low < 1.0 < ci_high):\n        regime = \"undetermined\"",
     "    if False:\n        regime = \"undetermined\""),
    ("M43", "v12 可行比例谎报为 1", "uncertainty.py",
     "        return (self.valid.size / n) if n else math.nan",
     "        return 1.0"),
    ("M44", "v12 解释力返回归一化后的 1", "uncertainty.py",
     "        return sum(e.spearman ** 2 for e in sensitivity(self)\n                   if math.isfinite(e.spearman))",
     "        return 1.0"),
    ("M45", "v12 质量来源声明被抹去", "efficiency.py",
     "    quality_basis: str = \"\"",
     "    quality_basis: str = \"Quoted constant\""),
    ("M46", "v13 CostLayer 校验被移除", "costing.py",
     "        if (self.step_cost > 0) != (self.step_size > 0):",
     "        if False:"),
    ("M47", "v13 CostLayer 负值校验被移除", "costing.py",
     "            if value < 0:",
     "            if False:"),
    ("M48", "v13 阶梯层向下取整", "costing.py",
     "            units = math.ceil(volume / self.step_size) if volume > 0 else 0",
     "            units = int(volume / self.step_size) if volume > 0 else 0"),
    ("M49", "v14 CLI 退回全体最小", "cli.py",
     "    admissible = [(n, r) for n, r in ordered if r.feasible]\n    best_name, best = (admissible or ordered)[0]",
     "    admissible = ordered\n    best_name, best = ordered[0]"),
    ("M50", "v14 CLI 弹性投影退回混合单价", "cli.py",
     "    return (split[\"declining_per_query\"], result.annual_volume,",
     "    return (result.effective_per_query, result.annual_volume,"),
    ("M51", "v14 质量差距不再记录", "costing.py",
     "    quality_shortfall = {w.name: (w.quality_floor - quality) / w.quality_floor",
     "    quality_shortfall = {}"),
    ("M52", "v15 CSV 退回只写 values()", "report.py",
     "    rows = []\n    for name, r in bundle.tco.items():",
     "    rows = []\n    for name, r in [(None, x) for x in bundle.tco.values()]:"),
    ("M53", "v15 写出器不再列出被排除者", "report.py",
     "            ruled_out = bundle.infeasible()\n            if ruled_out:\n                parts.append(",
     "            ruled_out = {}\n            if ruled_out:\n                parts.append("),
    ("M54", "v15 报告退回单一 band", "report.py",
     "            windows = be.tie_bands(0.05)\n            if len(be.crossings) > 4 and windows:\n                parts.append(",
     "            windows = [be.tie_band(0.05)] if be.tie_band(0.05) else []\n            if len(be.crossings) > 4 and windows:\n                parts.append("),
    ("M55", "v16 primary 重新汇总阶梯", "breakeven.py",
     "        if len(self.crossings) > 1:\n            raise ValueError(",
     "        if False:\n            raise ValueError("),
    ("M56", "v16 any_feasible 不再进入报告", "report.py",
     "            if not bundle.any_feasible:\n                parts.append(",
     "            if False:\n                parts.append("),
    ("M15", "约定归一化失效", "calibration.py",
     "        if self.convention == \"per_request\":\n            return self.measured_output_tps * self.batch",
     "        if self.convention == \"per_request\":\n            return self.measured_output_tps"),
]

# --- interruption safety -------------------------------------------------
# A run of this script that is killed part-way leaves a mutated source file
# on disk. That happened during the v14 audit: a timeout stopped the loop
# mid-mutant and a defect from the v9 round sat in breakeven.py until the
# next full test run caught it. The tree is checksummed before and after,
# and a dirty tree refuses to start.
import hashlib

_SNAPSHOT = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(SRC.glob("*.py"))}
_STAMP = PKG / ".mutation_in_progress"
if _STAMP.exists():
    raise SystemExit(
        f"{_STAMP} exists: a previous run was interrupted and the tree may "
        "still be mutated. Restore it from version control, delete the "
        "stamp, and re-run."
    )
_STAMP.write_text("running\n")

# --- sharding -------------------------------------------------------------
# The full set against the full suite is about 23 minutes, which does not
# fit one working session. Until v17 that fact was handled by running the
# new mutants only and calling it a sweep -- a partial run whose output
# reads exactly like a complete one. Sharding makes the whole set
# affordable instead: --shard k/n runs every n-th mutant, and three
# consecutive rounds cover it. The shard is printed and recorded, so a
# partial run cannot be mistaken for a full one by anyone, including the
# person who ran it.
import argparse as _argparse

_parser = _argparse.ArgumentParser(description="mutation sweep")
_parser.add_argument("--shard", default="1/1",
                     help="k/n: run every n-th mutant starting at k")
_args, _ = _parser.parse_known_args()
_k, _n = (int(x) for x in _args.shard.split("/"))
if not 1 <= _k <= _n:
    raise SystemExit(f"bad shard {_args.shard}")
MUTANTS = [m for i, m in enumerate(MUTANTS) if i % _n == _k - 1]
print(f"shard {_k}/{_n}: {len(MUTANTS)} mutant(s)\n")

# A mutant is only informative if the suite is green without it. Until
# v17.1 nothing checked that: a tree that was already failing would have
# reported every mutant as "captured", by the failure that was there all
# along. The first-catcher column is read by a person, and it has to be
# worth reading.
_baseline = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
     "-p", "no:cacheprovider"],
    cwd=PKG, capture_output=True, text=True, timeout=1800)
if _baseline.returncode != 0:
    _STAMP.unlink(missing_ok=True)
    raise SystemExit(
        "baseline suite is not green; every mutant would report as caught "
        "by the failure already present:\n" + _baseline.stdout[-1500:])
print(f"baseline green against {PKG.name}\n")

results = []
for mid, desc, fname, old, new in MUTANTS:
    path = SRC / fname
    backup = path.read_text()
    if backup.count(old) != 1:
        results.append((mid, desc, f"PATCH-FAIL(count={backup.count(old)})", ""))
        continue
    path.write_text(backup.replace(old, new))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-x", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=PKG, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            results.append((mid, desc, "逃逸", ""))
        else:
            first = ""
            for line in proc.stdout.splitlines():
                if line.startswith("FAILED"):
                    first = line.split("FAILED ")[1].split(" -")[0]
                    break
            results.append((mid, desc, "捕获", first))
    finally:
        path.write_text(backup)

print(f"{'编号':<5} {'结果':<5} {'说明':<28} 首个捕获测试")
escaped = 0
for mid, desc, verdict, test in results:
    if verdict == "逃逸":
        escaped += 1
    print(f"{mid:<5} {verdict:<5} {desc:<28} {test}")
print(f"\n逃逸: {escaped} / {len(results)}")

_dirty = [p.name for p, digest in _SNAPSHOT.items()
          if hashlib.sha256(p.read_bytes()).hexdigest() != digest]
_STAMP.unlink(missing_ok=True)
if _dirty:
    raise SystemExit(f"source tree left modified: {_dirty}")
print("源码树校验通过：与运行前逐字节一致")
