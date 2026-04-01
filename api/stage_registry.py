"""
Stage registry: introspects NeMo Curator's auto-registered stages
and provides a catalog for the API.

Two-layer architecture:
  - "builtin" stages: NeMo Curator's _STAGE_REGISTRY, loaded at import time, immutable.
  - "custom" stages: User-defined ProcessingStage subclasses stored as .py files
    in CUSTOM_STAGES_DIR, dynamically imported at runtime. Each gets a UUID on save.

Pipeline configs reference stages as {"source": "builtin"|"custom", "ref": "<name-or-uuid>"}.
"""

import inspect
import importlib
import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

CUSTOM_STAGES_DIR = Path("/workspace/curator/custom_stages")
CUSTOM_STAGES_INDEX = CUSTOM_STAGES_DIR / "index.json"


def _get_type_name(annotation: Any) -> str:
    """Convert a type annotation to a human-readable string."""
    if annotation is inspect.Parameter.empty:
        return "any"
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        arg_names = ", ".join(_get_type_name(a) for a in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{arg_names}]" if arg_names else origin_name
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _categorize_stage(cls_name: str, module_path: str) -> str:
    """Determine the category of a stage from its module path or class name."""
    module_lower = module_path.lower()
    if "classifier" in module_lower or "classifier" in cls_name.lower():
        return "classifiers"
    if "filter" in module_lower or "filter" in cls_name.lower():
        return "filters"
    if "modifier" in module_lower or "modifier" in cls_name.lower():
        return "modifiers"
    if "dedup" in module_lower or "dedup" in cls_name.lower():
        return "deduplication"
    if "io" in module_lower or "reader" in module_lower or "writer" in module_lower:
        return "io"
    if "download" in module_lower:
        return "download"
    if "embed" in module_lower:
        return "embedders"
    if "modules" in module_lower:
        return "document_ops"
    return "other"


def _load_text_stages() -> None:
    """Import all text stage modules to trigger StageMeta registration."""
    _modules = [
        "nemo_curator.stages.text.modules",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.modifiers",
        "nemo_curator.stages.text.deduplication",
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
    ]
    for mod_name in _modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            logger.warning(f"Could not import {mod_name}: {e}")


def _is_text_stage(cls: type) -> bool:
    """Check if a stage class belongs to the text modality."""
    module = getattr(cls, "__module__", "")
    # Include stages from text-specific modules and generic stages
    # that are commonly used with text (like base filters/modifiers)
    return (
        "stages.text" in module
        or "stages.deduplication" in module
        or "stages.file_partitioning" in module
    )


def _resolve_display_name(cls: type, cls_name: str) -> str:
    """Get a meaningful display name for a stage class.

    Many subclasses inherit `name = "ProcessingStage"` from the base class
    without overriding it. In that case, fall back to the class name itself.
    """
    raw_name = getattr(cls, "name", None)
    if raw_name and raw_name != "ProcessingStage" and raw_name.strip():
        return raw_name
    return cls_name


def get_text_stages_by_category() -> dict[str, list[dict[str, str]]]:
    """Return text stages grouped by category.

    Returns:
        Dict mapping category -> list of {id, name} dicts.
    """
    from nemo_curator.stages.base import _STAGE_REGISTRY

    _load_text_stages()

    categories: dict[str, list[dict[str, str]]] = {}
    for cls_name, cls in sorted(_STAGE_REGISTRY.items()):
        if not _is_text_stage(cls):
            continue
        module = getattr(cls, "__module__", "")
        category = _categorize_stage(cls_name, module)
        display_name = _resolve_display_name(cls, cls_name)
        categories.setdefault(category, []).append({
            "id": cls_name,
            "name": display_name,
            "source": "builtin",
        })
    return categories


def get_category_stages(category: str) -> list[dict[str, str]] | None:
    """Return stages for a specific category.

    Args:
        category: The category name (classifiers, filters, modifiers, etc.)

    Returns:
        List of {id, name} dicts, or None if category doesn't exist.
    """
    all_categories = get_text_stages_by_category()
    return all_categories.get(category)


def get_text_stage_detail(stage_id: str) -> dict[str, Any] | None:
    """Return full details for a specific text stage.

    Args:
        stage_id: The class name of the stage.

    Returns:
        Dict with id, name, category, description, module, parameters, resources,
        or None if not found.
    """
    from nemo_curator.stages.base import _STAGE_REGISTRY

    _load_text_stages()

    cls = _STAGE_REGISTRY.get(stage_id)
    if cls is None or not _is_text_stage(cls):
        return None

    module = getattr(cls, "__module__", "")
    category = _categorize_stage(stage_id, module)
    display_name = _resolve_display_name(cls, stage_id)

    # Extract docstring
    description = inspect.getdoc(cls) or ""

    # Introspect __init__ parameters
    parameters = []
    try:
        sig = inspect.signature(cls)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue
            param_info = {
                "name": param_name,
                "type": _get_type_name(param.annotation),
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                # Serialize the default value safely
                default = param.default
                if callable(default) and not isinstance(default, type):
                    param_info["default"] = str(default)
                else:
                    try:
                        import json
                        json.dumps(default)
                        param_info["default"] = default
                    except (TypeError, ValueError):
                        param_info["default"] = str(default)
            parameters.append(param_info)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not introspect {stage_id}: {e}")

    # Resource info
    resources_cls = getattr(cls, "resources", None)
    resource_info = {}
    if resources_cls is not None:
        resource_info = {
            "cpus": getattr(resources_cls, "cpus", 1.0),
            "requires_gpu": getattr(resources_cls, "requires_gpu", False),
        }
        if getattr(resources_cls, "requires_gpu", False):
            resource_info["gpu_memory_gb"] = getattr(resources_cls, "gpu_memory_gb", None)
            resource_info["gpus"] = getattr(resources_cls, "gpus", 1)

    return {
        "id": stage_id,
        "name": display_name,
        "source": "builtin",
        "category": category,
        "description": description,
        "module": module,
        "parameters": parameters,
        "resources": resource_info,
        "batch_size": getattr(cls, "batch_size", 1),
    }


# ─── Custom Stages ──────────────────────────────────────────────────────


def _ensure_custom_dirs() -> None:
    """Create custom stages directory and index if missing."""
    CUSTOM_STAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_STAGES_INDEX.exists():
        CUSTOM_STAGES_INDEX.write_text("{}")


def _load_custom_index() -> dict[str, dict]:
    """Load the custom stages index.

    Returns:
        Dict mapping UUID -> {name, category, filename, created_at}
    """
    _ensure_custom_dirs()
    try:
        return json.loads(CUSTOM_STAGES_INDEX.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_custom_index(index: dict[str, dict]) -> None:
    """Persist custom stages index (atomic write)."""
    tmp = CUSTOM_STAGES_INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, default=str))
    tmp.replace(CUSTOM_STAGES_INDEX)


def _load_custom_stage_class(stage_uuid: str) -> type | None:
    """Dynamically import a custom stage's .py file and return the class.

    The file must define exactly one concrete ProcessingStage subclass.
    Importing it triggers StageMeta registration automatically.
    """
    index = _load_custom_index()
    entry = index.get(stage_uuid)
    if not entry:
        return None

    filepath = CUSTOM_STAGES_DIR / entry["filename"]
    if not filepath.exists():
        return None

    # Snapshot registry before import to detect what's new
    from nemo_curator.stages.base import _STAGE_REGISTRY
    before = set(_STAGE_REGISTRY.keys())

    module_name = f"custom_stage_{stage_uuid.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.error(f"Failed to load custom stage {stage_uuid}: {e}")
        return None

    # Find newly registered class(es)
    new_classes = set(_STAGE_REGISTRY.keys()) - before
    if not new_classes:
        logger.error(f"Custom stage {stage_uuid} did not register any ProcessingStage")
        return None

    # Return the first new class (user files should define exactly one)
    return _STAGE_REGISTRY[next(iter(new_classes))]


def validate_custom_stage_name(name: str) -> str | None:
    """Check if a custom stage name conflicts with builtins or other custom stages.

    Returns:
        Error message if conflict found, None if name is available.
    """
    _load_text_stages()

    from nemo_curator.stages.base import _STAGE_REGISTRY
    # Check against builtin class names
    if name in _STAGE_REGISTRY:
        return f"Name '{name}' conflicts with builtin stage '{name}'"

    # Check against builtin display names
    for cls_name, cls in _STAGE_REGISTRY.items():
        display = _resolve_display_name(cls, cls_name)
        if display == name:
            return f"Name '{name}' conflicts with builtin stage '{cls_name}'"

    # Check against existing custom stage names
    index = _load_custom_index()
    for uid, entry in index.items():
        if entry["name"] == name:
            return f"Name '{name}' conflicts with existing custom stage (id: {uid})"

    return None


def save_custom_stage(name: str, category: str, code: str) -> dict[str, Any]:
    """Save a user-defined custom stage.

    Args:
        name: Display name for the stage (must be unique).
        category: Category to file it under.
        code: Python source code defining a ProcessingStage subclass.

    Returns:
        Dict with the saved stage info including its UUID.

    Raises:
        ValueError: If name conflicts or code is invalid.
    """
    # Validate name uniqueness
    conflict = validate_custom_stage_name(name)
    if conflict:
        raise ValueError(conflict)

    stage_uuid = str(uuid.uuid4())
    filename = f"{stage_uuid}.py"
    filepath = CUSTOM_STAGES_DIR / filename

    # Write the code
    _ensure_custom_dirs()
    filepath.write_text(code)

    # Try to load it to validate it's a valid ProcessingStage
    cls = _load_custom_stage_class(stage_uuid)
    if cls is None:
        filepath.unlink(missing_ok=True)
        raise ValueError(
            "Code must define exactly one concrete ProcessingStage subclass. "
            "Ensure it inherits from ProcessingStage and implements process()."
        )

    # Introspect inputs/outputs for the node graph
    try:
        instance = cls.__new__(cls)
        inputs = instance.inputs()
        outputs = instance.outputs()
    except Exception:
        inputs = ([], [])
        outputs = ([], [])

    # Update index
    from datetime import datetime
    index = _load_custom_index()
    index[stage_uuid] = {
        "name": name,
        "category": category,
        "filename": filename,
        "class_name": cls.__name__,
        "created_at": datetime.now().isoformat(),
        "inputs": {"attributes": inputs[0], "data_fields": inputs[1]},
        "outputs": {"attributes": outputs[0], "data_fields": outputs[1]},
    }
    _save_custom_index(index)

    return {
        "id": stage_uuid,
        "name": name,
        "source": "custom",
        "category": category,
        "class_name": cls.__name__,
        "inputs": index[stage_uuid]["inputs"],
        "outputs": index[stage_uuid]["outputs"],
    }


def get_custom_stage_detail(stage_uuid: str) -> dict[str, Any] | None:
    """Return full details for a custom stage by UUID."""
    index = _load_custom_index()
    entry = index.get(stage_uuid)
    if not entry:
        return None

    filepath = CUSTOM_STAGES_DIR / entry["filename"]
    code = filepath.read_text() if filepath.exists() else ""

    # Try to load and introspect the class
    cls = _load_custom_stage_class(stage_uuid)
    parameters = []
    if cls:
        try:
            sig = inspect.signature(cls)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs"):
                    continue
                param_info = {
                    "name": param_name,
                    "type": _get_type_name(param.annotation),
                    "required": param.default is inspect.Parameter.empty,
                }
                if param.default is not inspect.Parameter.empty:
                    default = param.default
                    if callable(default) and not isinstance(default, type):
                        param_info["default"] = str(default)
                    else:
                        try:
                            json.dumps(default)
                            param_info["default"] = default
                        except (TypeError, ValueError):
                            param_info["default"] = str(default)
                parameters.append(param_info)
        except (ValueError, TypeError):
            pass

    return {
        "id": stage_uuid,
        "name": entry["name"],
        "source": "custom",
        "category": entry["category"],
        "class_name": entry.get("class_name", ""),
        "description": inspect.getdoc(cls) if cls else "",
        "code": code,
        "parameters": parameters,
        "inputs": entry.get("inputs", {"attributes": [], "data_fields": []}),
        "outputs": entry.get("outputs", {"attributes": [], "data_fields": []}),
        "created_at": entry.get("created_at"),
    }


def delete_custom_stage(stage_uuid: str) -> bool:
    """Delete a custom stage by UUID.

    Returns:
        True if deleted, False if not found.
    """
    index = _load_custom_index()
    entry = index.pop(stage_uuid, None)
    if not entry:
        return False

    filepath = CUSTOM_STAGES_DIR / entry["filename"]
    filepath.unlink(missing_ok=True)
    _save_custom_index(index)
    return True
