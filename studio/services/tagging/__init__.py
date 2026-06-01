"""tagger 家族 —— PR-3 从 services/ 平铺移到本子包。

文件对应：
  - base.py            原 tagger.py（Protocol + factory + VALID_TAGGER_NAMES）
  - caption_format.py  原 caption_format.py
  - caption_snapshot.py 原 caption_snapshot.py
  - onnx_base.py       原 onnx_tagger_base.py
  - wd14.py            原 wd14_tagger.py
  - cltagger.py        原 cltagger_tagger.py
  - llm.py             原 llm_tagger.py
  - joycaption.py      原 joycaption_tagger.py

re-export 主流公共名供 `from studio.services.tagging import X` 使用。
原 import 路径（`studio.services.tagger` 等）通过同层 shim 文件保持兼容。
"""
from .base import VALID_TAGGER_NAMES, ProgressFn, TagResult, Tagger, get_tagger
from .caption_format import (
    caption_json_to_tags,
    caption_json_to_text,
    normalize_caption_json,
    split_tags,
    standard_to_documented_full,
)
from .caption_snapshot import (
    SnapshotError,
    create_snapshot,
    delete_snapshot,
    list_snapshots,
    restore_snapshot,
    snapshot_root,
)
from .cltagger import CLTagger
from .joycaption import JoyCaptionTagger
from .llm import LLMTagger, fetch_openai_compatible_models, test_openai_compatible_connection
from .onnx_base import OnnxTaggerBase, safe_dir_name, silenced_fd_stderr
from .wd14 import WD14Tagger

__all__ = [
    "CLTagger",
    "JoyCaptionTagger",
    "LLMTagger",
    "OnnxTaggerBase",
    "ProgressFn",
    "SnapshotError",
    "TagResult",
    "Tagger",
    "VALID_TAGGER_NAMES",
    "WD14Tagger",
    "caption_json_to_tags",
    "caption_json_to_text",
    "create_snapshot",
    "delete_snapshot",
    "fetch_openai_compatible_models",
    "get_tagger",
    "list_snapshots",
    "normalize_caption_json",
    "restore_snapshot",
    "safe_dir_name",
    "silenced_fd_stderr",
    "snapshot_root",
    "split_tags",
    "standard_to_documented_full",
    "test_openai_compatible_connection",
]
