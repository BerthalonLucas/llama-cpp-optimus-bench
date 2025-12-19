"""Profile management for saving/loading benchmark configurations."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .bench import BenchConfig


@dataclass
class BenchProfile:
    """A saved benchmark configuration profile."""
    name: str
    description: str
    config: BenchConfig
    model_pattern: str | None  # Optional: model name pattern to auto-match
    created_at: str
    updated_at: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config": asdict(self.config),
            "model_pattern": self.model_pattern,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchProfile":
        config_data = data.get("config", {})
        config = BenchConfig(**config_data)
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            config=config,
            model_pattern=data.get("model_pattern"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass  
class SweepProfile:
    """A saved sweep configuration profile."""
    name: str
    description: str
    mode: str  # "1d" or "multi"
    base_config: BenchConfig
    sweep_values: dict[str, list[Any]]  # param_name -> list of values
    weight_tg: float
    weight_pp: float
    model_pattern: str | None
    created_at: str
    updated_at: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "base_config": asdict(self.base_config),
            "sweep_values": self.sweep_values,
            "weight_tg": self.weight_tg,
            "weight_pp": self.weight_pp,
            "model_pattern": self.model_pattern,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SweepProfile":
        config_data = data.get("base_config", {})
        base_config = BenchConfig(**config_data)
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            mode=data.get("mode", "multi"),
            base_config=base_config,
            sweep_values=data.get("sweep_values", {}),
            weight_tg=data.get("weight_tg", 0.7),
            weight_pp=data.get("weight_pp", 0.3),
            model_pattern=data.get("model_pattern"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class ProfileManager:
    """Manages saving/loading profiles to/from disk."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.bench_profiles_file = storage_dir / "bench_profiles.json"
        self.sweep_profiles_file = storage_dir / "sweep_profiles.json"
        storage_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── Bench Profiles ───────────────────────────────────────────────────
    
    def list_bench_profiles(self) -> list[BenchProfile]:
        """List all saved bench profiles."""
        if not self.bench_profiles_file.exists():
            return []
        try:
            data = json.loads(self.bench_profiles_file.read_text(encoding="utf-8"))
            return [BenchProfile.from_dict(p) for p in data.get("profiles", [])]
        except (json.JSONDecodeError, KeyError):
            return []
    
    def get_bench_profile(self, name: str) -> BenchProfile | None:
        """Get a specific bench profile by name."""
        profiles = self.list_bench_profiles()
        for p in profiles:
            if p.name == name:
                return p
        return None
    
    def save_bench_profile(self, profile: BenchProfile) -> None:
        """Save or update a bench profile."""
        profiles = self.list_bench_profiles()
        
        # Update existing or add new
        found = False
        for i, p in enumerate(profiles):
            if p.name == profile.name:
                profiles[i] = profile
                found = True
                break
        
        if not found:
            profiles.append(profile)
        
        # Save to disk
        data = {"profiles": [p.to_dict() for p in profiles]}
        self.bench_profiles_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def delete_bench_profile(self, name: str) -> bool:
        """Delete a bench profile by name. Returns True if deleted."""
        profiles = self.list_bench_profiles()
        new_profiles = [p for p in profiles if p.name != name]
        
        if len(new_profiles) == len(profiles):
            return False  # Not found
        
        data = {"profiles": [p.to_dict() for p in new_profiles]}
        self.bench_profiles_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return True
    
    # ─── Sweep Profiles ───────────────────────────────────────────────────
    
    def list_sweep_profiles(self) -> list[SweepProfile]:
        """List all saved sweep profiles."""
        if not self.sweep_profiles_file.exists():
            return []
        try:
            data = json.loads(self.sweep_profiles_file.read_text(encoding="utf-8"))
            return [SweepProfile.from_dict(p) for p in data.get("profiles", [])]
        except (json.JSONDecodeError, KeyError):
            return []
    
    def get_sweep_profile(self, name: str) -> SweepProfile | None:
        """Get a specific sweep profile by name."""
        profiles = self.list_sweep_profiles()
        for p in profiles:
            if p.name == name:
                return p
        return None
    
    def save_sweep_profile(self, profile: SweepProfile) -> None:
        """Save or update a sweep profile."""
        profiles = self.list_sweep_profiles()
        
        found = False
        for i, p in enumerate(profiles):
            if p.name == profile.name:
                profiles[i] = profile
                found = True
                break
        
        if not found:
            profiles.append(profile)
        
        data = {"profiles": [p.to_dict() for p in profiles]}
        self.sweep_profiles_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def delete_sweep_profile(self, name: str) -> bool:
        """Delete a sweep profile by name. Returns True if deleted."""
        profiles = self.list_sweep_profiles()
        new_profiles = [p for p in profiles if p.name != name]
        
        if len(new_profiles) == len(profiles):
            return False
        
        data = {"profiles": [p.to_dict() for p in new_profiles]}
        self.sweep_profiles_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return True


def create_bench_profile(
    name: str,
    config: BenchConfig,
    description: str = "",
    model_pattern: str | None = None,
) -> BenchProfile:
    """Create a new bench profile with current timestamp."""
    now = datetime.now().isoformat()
    return BenchProfile(
        name=name,
        description=description,
        config=config,
        model_pattern=model_pattern,
        created_at=now,
        updated_at=now,
    )


def create_sweep_profile(
    name: str,
    base_config: BenchConfig,
    sweep_values: dict[str, list[Any]],
    weight_tg: float = 0.7,
    weight_pp: float = 0.3,
    description: str = "",
    model_pattern: str | None = None,
    mode: str = "multi",
) -> SweepProfile:
    """Create a new sweep profile with current timestamp."""
    now = datetime.now().isoformat()
    return SweepProfile(
        name=name,
        description=description,
        mode=mode,
        base_config=base_config,
        sweep_values=sweep_values,
        weight_tg=weight_tg,
        weight_pp=weight_pp,
        model_pattern=model_pattern,
        created_at=now,
        updated_at=now,
    )
