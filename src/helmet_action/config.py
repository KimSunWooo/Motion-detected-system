from __future__ import annotations

import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.yaml").exists():
            return parent
    return Path.cwd()


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2)
            return os.environ.get(key, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = _expand_env(data)

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self._data, dotted, default)

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else repo_root() / "config" / "default.yaml"
    return Config(_load_yaml(cfg_path))


def reload_config(path: str | Path | None = None) -> Config:
    load_config.cache_clear()
    return load_config(path)


def as_dict(cfg: Config | None = None) -> dict[str, Any]:
    return deepcopy((cfg or load_config()).data)
