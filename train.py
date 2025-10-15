"""Training utilities for the casting recognition project."""

# NOTE: 이 스크립트는 이제 Roboflow API를 통해 데이터셋을 자동으로 내려받아
#       다른 컴퓨터에서도 동일한 설정으로 학습을 재현할 수 있게 합니다.

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Union
from ultralytics import YOLO

DEFAULT_PROJECT_DIR = Path("runs/detect")
DEFAULT_RUN_NAME = "train_fixed_aug"
DEFAULT_MODEL_WEIGHTS = Path("yolov8s-seg.pt")
# NOTE: 기본 epoch/batch 값을 조정해 전체 반복 횟수를 약 절반으로 줄여
#       학습 시간이 이전 대비 빠르게 끝나도록 구성합니다.
DEFAULT_EPOCHS = 15
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 14
DEFAULT_DATA_CONFIG = None
DEFAULT_ROBOFLOW_FORMAT = "yolov8"
DEFAULT_ROBOFLOW_CONFIG = Path("data.yaml")
DEFAULT_DATASET_DIR = Path("datasets")
DATA_CONFIG_FILENAME = "data.yaml"
DEFAULT_API_KEY_ENV = "ROBOFLOW_API_KEY"


def _coerce_scalar(value: str) -> Union[str, int]:
    """Convert simple scalar string values to integers when appropriate."""

    value = value.strip()
    if value.isdigit():
        try:
            return int(value)
        except ValueError:  # pragma: no cover - defensive
            return value
    return value


def _minimal_yaml_parse(text: str) -> Mapping[str, object]:
    """Parse a tiny subset of YAML required for the Roboflow configuration."""

    result: dict[str, object] = {}
    current_section: Optional[dict[str, object]] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if not raw_line.startswith(" ") and ":" in raw_line:
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = _coerce_scalar(value)
                current_section = None
            else:
                section: dict[str, object] = {}
                result[key] = section
                current_section = section
            continue

        if current_section is not None and ":" in line:
            key, value = line.split(":", 1)
            current_section[key.strip()] = _coerce_scalar(value)

    return result


def _load_config(path: Path) -> Mapping[str, object]:
    """Load the repository configuration, falling back to a minimal parser."""

    text = path.read_text()
    try:
        import yaml  # type: ignore
    except ImportError:
        return _minimal_yaml_parse(text)

    return yaml.safe_load(text) or {}


def _resolve_path(base_dir: Path, value: Union[str, Path]) -> Path:
    """Resolve *value* relative to *base_dir* when the value is not absolute."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_roboflow_metadata(base_dir: Path) -> Mapping[str, object]:
    """Load Roboflow configuration metadata from the repository config file."""

    config_path = _resolve_path(base_dir, DEFAULT_ROBOFLOW_CONFIG)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Roboflow configuration not found: {config_path}. "
            "Ensure data.yaml is present in the project root."
        )

    config = _load_config(config_path)
    metadata = config.get("roboflow")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Roboflow configuration is missing from data.yaml")

    required_keys = {"workspace", "project", "version"}
    missing = sorted(required_keys.difference(metadata))
    if missing:
        raise RuntimeError(
            "Roboflow configuration is incomplete. Missing keys: " + ", ".join(missing)
        )

    return metadata


# NOTE: 아래 함수들은 Roboflow에서 직접 데이터셋을 내려받는 과정과 관련된
#       로직을 한글 주석으로 설명하여, 기존 로컬 경로 고정 방식과의 차이를
#       쉽게 파악할 수 있도록 구성했습니다.


def _sanitize_identifier(value: str) -> str:
    """Return a filesystem-friendly representation of *value*."""

    sanitized = [
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip()
    ]
    result = "".join(sanitized).strip("-_")
    return result or "dataset"


def _format_dataset_cache_dir(workspace: str, project: str, version: int) -> str:
    """Create a deterministic cache directory name for a Roboflow dataset."""

    return "_".join(
        [
            _sanitize_identifier(workspace),
            _sanitize_identifier(project),
            f"v{version}",
        ]
    )


def _find_cached_data_config(directory: Path) -> Optional[Path]:
    """Locate an existing data.yaml inside *directory* if it exists."""

    if not directory.exists():
        return None

    candidates = sorted(directory.rglob(DATA_CONFIG_FILENAME))
    if candidates:
        return candidates[0]
    return None


def _prepare_clean_directory(path: Path) -> None:
    """Ensure *path* exists and is empty."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prune_archives(directory: Path) -> None:
    """Remove stray archive files left by Roboflow downloads."""

    if not directory.exists():
        return

    for archive in directory.glob("*.zip"):
        try:
            archive.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def _download_dataset_via_roboflow(
    *,
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    dataset_format: str,
    download_dir: Path,
) -> Path:
    """Download a dataset from Roboflow and return the extracted data.yaml path."""

    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover - error path
        raise ImportError(
            "The 'roboflow' package is required to download datasets. "
            "Install it with 'pip install roboflow'."
        ) from exc

    # Roboflow SDK 초기화 후, 워크스페이스/프로젝트/버전을 순서대로 지정해
    #    기존의 수동 다운로드 링크 대신 API로 데이터셋 아카이브를 요청합니다.
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version)
    result = dataset.download(dataset_format, location=str(download_dir))

    dataset_location = Path(result.location)
    if not dataset_location.is_absolute():
        dataset_location = (download_dir / dataset_location).resolve()

    data_config_path = dataset_location / DATA_CONFIG_FILENAME
    if not data_config_path.exists():
        fallback = download_dir / DATA_CONFIG_FILENAME
        if fallback.exists():
            data_config_path = fallback
        else:  # pragma: no cover - defensive branch
            raise FileNotFoundError(
                "The downloaded dataset did not contain data.yaml at the expected location."
            )

    _prune_archives(download_dir)
    return data_config_path


def ensure_dataset_available(
    base_dir: Path,
    data_config_path: Optional[Path],
    *,
    roboflow_api_key: Optional[str] = None,
    dataset_format: Optional[str] = None,
    roboflow_workspace: Optional[str] = None,
    roboflow_project: Optional[str] = None,
    roboflow_version: Optional[Union[str, int]] = None,
) -> Path:
    """Ensure a dataset is present locally, downloading it from Roboflow if necessary.

    Optional workspace/project/version overrides make it easy to switch Roboflow
    dataset links without editing ``data.yaml``.
    """

    if data_config_path and data_config_path.exists():
        return data_config_path

    # data.yaml 내부에 포함된 Roboflow 메타데이터를 읽어 API 호출에 필요한
    #    workspace/project/version 값을 가져옵니다.
    metadata = dict(load_roboflow_metadata(base_dir))

    # 사용자가 workspace/project/version 값을 직접 넘겨줄 수 있게 하여
    #    data.yaml을 수정하지 않아도 필요한 Roboflow 링크로 쉽게 전환하도록 한다.
    if roboflow_workspace is not None:
        metadata["workspace"] = roboflow_workspace
    if roboflow_project is not None:
        metadata["project"] = roboflow_project
    if roboflow_version is not None:
        metadata["version"] = roboflow_version

    dataset_format = dataset_format or str(metadata.get("format", DEFAULT_ROBOFLOW_FORMAT))
    dataset_dir = metadata.get("dataset_dir", DEFAULT_DATASET_DIR)
    dataset_dir_path = _resolve_path(base_dir, Path(dataset_dir))
    # 사용자가 workspace/project/version 값을 바꿔도, 실제 내려받은 데이터셋은
    #    위에서 계산한 dataset_dir 경로(기본값은 프로젝트 내부 datasets 폴더)
    #    아래에 버전별 캐시 폴더로 저장됩니다. 즉, 기존 실행 결과와 동일한
    #    위치 구조를 유지하면서 링크만 교체할 수 있습니다.
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    api_key_env = str(metadata.get("api_key_env", DEFAULT_API_KEY_ENV))
    # 환경 변수(기본값은 ROBOFLOW_API_KEY) 또는 함수 인자로 전달된 키를 사용해
    #    Roboflow 인증을 수행합니다. 예전에는 로컬 데이터 경로만 필요했지만,
    #    이제는 API 키가 없으면 데이터셋을 받을 수 없습니다.
    api_key = roboflow_api_key or os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            "A Roboflow API key is required to download the dataset. "
            f"Provide it explicitly or set the {api_key_env} environment variable."
        )

    workspace = str(metadata["workspace"])
    project = str(metadata["project"])
    version = int(metadata["version"])

    cache_dir_name = _format_dataset_cache_dir(workspace, project, version)
    cache_dir = dataset_dir_path / cache_dir_name

    cached_config = _find_cached_data_config(cache_dir)
    if cached_config:
        return cached_config

    _prepare_clean_directory(cache_dir)

    # 위 정보를 바탕으로 Roboflow에 다운로드를 요청하고, 내려받은 data.yaml
    #    경로를 반환합니다. 이렇게 받아온 경로를 학습에 그대로 사용합니다.
    return _download_dataset_via_roboflow(
        workspace=workspace,
        project=project,
        version=version,
        api_key=api_key,
        dataset_format=dataset_format,
        download_dir=cache_dir,
    )


def train_model(
    *,
    base_dir: Optional[Union[str, Path]] = None,
    data_config: Optional[Union[str, Path]] = DEFAULT_DATA_CONFIG,
    project_dir: Union[str, Path] = DEFAULT_PROJECT_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    model_weights: Union[str, Path] = DEFAULT_MODEL_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    batch: int = DEFAULT_BATCH_SIZE,
    roboflow_api_key: Optional[str] = None,
    dataset_format: Optional[str] = None,
    roboflow_workspace: Optional[str] = None,
    roboflow_project: Optional[str] = None,
    roboflow_version: Optional[Union[str, int]] = None,
) -> None:
    """Train a YOLO model, downloading the dataset from Roboflow if required.

    Additional Roboflow override 인자를 사용하면 코드 실행 시점에 원하는
    workspace/project/version으로 쉽게 전환할 수 있다.
    """

    base_dir_path = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    base_dir_path = base_dir_path.resolve()

    project_dir_path = _resolve_path(base_dir_path, project_dir)
    project_dir_path.mkdir(parents=True, exist_ok=True)

    run_dir = project_dir_path / run_name
    if run_dir.exists():
        try:
            run_dir.resolve().relative_to(project_dir_path)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(
                f"Refusing to delete path outside {project_dir_path}: {run_dir}"
            ) from exc

        try:
            shutil.rmtree(run_dir)
        except OSError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Failed to remove previous run directory: {run_dir}") from exc

    data_config_path = None
    if data_config:
        data_config_path = _resolve_path(base_dir_path, data_config)

    # ensure_dataset_available이 실제로 Roboflow에서 데이터셋을 내려받거나
    #    이미 내려받은 폴더를 재사용해 data.yaml 경로를 제공합니다. 필요하면
    #    함수 인자로 전달된 workspace/project/version 값을 우선 적용합니다.
    data_config_path = ensure_dataset_available(
        base_dir_path,
        data_config_path,
        roboflow_api_key=roboflow_api_key,
        dataset_format=dataset_format,
        roboflow_workspace=roboflow_workspace,
        roboflow_project=roboflow_project,
        roboflow_version=roboflow_version,
    )

    if isinstance(model_weights, Path):
        weights_arg = str(_resolve_path(base_dir_path, model_weights))
    else:
        candidate_path = Path(model_weights)
        resolved_candidate = _resolve_path(base_dir_path, candidate_path)
        if resolved_candidate.exists():
            weights_arg = str(resolved_candidate)
        else:
            weights_arg = str(model_weights)

    model = YOLO(weights_arg)
    model.train(
        data=str(data_config_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir_path),
        name=run_name,
        auto_augment=None,
    )


if __name__ == "__main__":
    train_model()
