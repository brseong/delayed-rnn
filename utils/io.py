import torch
from torch import nn
from pathlib import Path
from uuid import uuid4
from json import dump, load
from enum import Enum
from importlib import import_module


def _json_default(obj):
    if isinstance(obj, Enum):
        enum_type = type(obj)
        return {
            "__kind__": "enum",
            "module": enum_type.__module__,
            "name": enum_type.__name__,
            "value": obj.name,
        }
    if isinstance(obj, torch.device):
        return {"__kind__": "torch.device", "value": str(obj)}
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _json_object_hook(obj):
    kind = obj.get("__kind__")
    if kind == "enum":
        try:
            module = import_module(obj["module"])
            enum_type = getattr(module, obj["name"])
            return enum_type[obj["value"]]
        except Exception:
            # Fallback to string when enum import/lookup fails.
            return obj["value"]
    if kind == "torch.device":
        return torch.device(obj["value"])
    return obj

def _save_model_to_path(model:nn.Module, data: dict, path:Path):
    """모델 저장"""
    torch.save(model.state_dict(), path / "model.pt")
    with open(path.parent / "contents.json", "r") as f:
        loaded_data:dict = load(f, object_hook=_json_object_hook)
    loaded_data[path.name] = data
    with open(path.parent / "contents.json", "w") as f:
        dump(loaded_data, f, indent=4, default=_json_default)

def _load_model_from_path(model:nn.Module, path:Path) -> dict:
    """모델 불러오기"""
    model.load_state_dict(torch.load(path / "model.pt"))
    with open(path.parent / "contents.json", "r") as f:
        data:dict = load(f, object_hook=_json_object_hook)
    return data[path.name]

def save_model(model:nn.Module, data: dict, *, path_root:Path|None = None):
    """모델 저장 (경로 자동 생성)"""
    if path_root is None:
        path_root = Path('models')
    model_name = model.__class__.__name__
    
    path = path_root / model_name / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    
    if not (path.parent / "contents.json").exists():
        with open(path.parent / "contents.json", "w") as f:
            dump({}, f, indent=4, default=_json_default)
            
    _save_model_to_path(model, data, path)
    print(f"Model saved to: {path}")
    
def load_model(model:nn.Module, uuid_str:str, *, path_root:Path|None = None):
    """모델 불러오기 (UUID로 경로 자동 생성)"""
    if path_root is None:
        path_root = Path('models')
    model_name = model.__class__.__name__
    
    path = path_root / model_name / uuid_str
    
    if not path.exists():
        raise FileNotFoundError(f"No model found at: {path}")
    
    data = _load_model_from_path(model, path)
    print(f"Model loaded from: {path}")
    return data

if __name__ == "__main__":
    # 테스트 코드
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    
    model = DummyModel()
    data = {"epoch": 5, "accuracy": 0.85}
    
    save_model(model, data)
    
    # 저장된 모델의 UUID를 복사해서 여기에 붙여넣기
    uuid_str = input("Enter the UUID of the model to load: ")
    loaded_data = load_model(model, uuid_str)
    print("Loaded Data:", loaded_data)