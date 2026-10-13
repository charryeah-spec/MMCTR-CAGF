"""Task 1 嵌入消融实验预设（A0–A5）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = ROOT / "data/MicroLens_1M_x1/ablation"
OFFICIAL_ITEM_INFO = ROOT / "data/MicroLens_1M_x1/item_info.parquet"


@dataclass(frozen=True)
class AblationPreset:
    code: str
    name: str
    purpose: str
    use_rq: bool
    use_behavior: bool
    use_mm: bool
    equal_weights: bool
    use_full_weights: bool = False
    use_gated: bool = False

    @property
    def item_info_path(self) -> Path:
        if self.code == "A0":
            return OFFICIAL_ITEM_INFO
        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        return ABLATION_DIR / f"item_info_{self.code}.parquet"

    @property
    def needs_build(self) -> bool:
        return self.code != "A0"


PRESETS: dict[str, AblationPreset] = {
    "A0": AblationPreset(
        code="A0",
        name="官方基线",
        purpose="官方嵌入复现",
        use_rq=False,
        use_behavior=False,
        use_mm=False,
        equal_weights=False,
    ),
    "A1": AblationPreset(
        code="A1",
        name="官方 + RQ",
        purpose="验证结构平滑",
        use_rq=True,
        use_behavior=False,
        use_mm=False,
        equal_weights=True,
    ),
    "A2": AblationPreset(
        code="A2",
        name="官方 + 行为 SVD",
        purpose="验证协同信号",
        use_rq=False,
        use_behavior=True,
        use_mm=False,
        equal_weights=True,
    ),
    "A3": AblationPreset(
        code="A3",
        name="官方 + 多版本均值",
        purpose="验证集成降方差",
        use_rq=False,
        use_behavior=False,
        use_mm=True,
        equal_weights=True,
    ),
    "A4": AblationPreset(
        code="A4",
        name="四源完整融合",
        purpose="完整方案",
        use_rq=True,
        use_behavior=True,
        use_mm=True,
        equal_weights=False,
        use_full_weights=True,
    ),
    "A5": AblationPreset(
        code="A5",
        name="CAGF 自适应门控",
        purpose="验证逐物品置信度融合",
        use_rq=True,
        use_behavior=True,
        use_mm=True,
        equal_weights=False,
        use_gated=True,
    ),
}

ORDER = ["A0", "A1", "A2", "A3", "A4", "A5"]
