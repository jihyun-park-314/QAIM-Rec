"""Pilot scripts (plan.md §4 Stage P1-P4, M6).

Each pilotN module re-uses M0-M5 functions with small configs (plan.md §1
design principle: "Pilot은 동일 모듈을 소규모 config로 재사용 — 별도 구현
금지"). These modules are skeletons only: function signatures + dataclass
schemas, no implementation logic.
"""
