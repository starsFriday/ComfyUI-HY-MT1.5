import threading
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import folder_paths

LANGUAGE_NAMES_EN: Dict[str, str] = {
    "zh": "Chinese",
    "zh-Hant": "Traditional Chinese",
    "yue": "Cantonese",
    "en": "English",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "tr": "Turkish",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "th": "Thai",
    "it": "Italian",
    "de": "German",
    "vi": "Vietnamese",
    "ms": "Malay",
    "id": "Indonesian",
    "tl": "Filipino",
    "hi": "Hindi",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
    "km": "Khmer",
    "my": "Burmese",
    "fa": "Persian",
    "gu": "Gujarati",
    "ur": "Urdu",
    "te": "Telugu",
    "mr": "Marathi",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "uk": "Ukrainian",
    "bo": "Tibetan",
    "kk": "Kazakh",
    "mn": "Mongolian",
    "ug": "Uyghur",
}

LANGUAGE_NAMES_ZH: Dict[str, str] = {
    "zh": "中文",
    "zh-Hant": "繁体中文",
    "yue": "粤语",
    "en": "英语",
    "fr": "法语",
    "pt": "葡萄牙语",
    "es": "西班牙语",
    "ja": "日语",
    "tr": "土耳其语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "ko": "韩语",
    "th": "泰语",
    "it": "意大利语",
    "de": "德语",
    "vi": "越南语",
    "ms": "马来语",
    "id": "印尼语",
    "tl": "菲律宾语",
    "hi": "印地语",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
    "km": "高棉语",
    "my": "缅甸语",
    "fa": "波斯语",
    "gu": "古吉拉特语",
    "ur": "乌尔都语",
    "te": "泰卢固语",
    "mr": "马拉地语",
    "he": "希伯来语",
    "bn": "孟加拉语",
    "ta": "泰米尔语",
    "uk": "乌克兰语",
    "bo": "藏语",
    "kk": "哈萨克语",
    "mn": "蒙古语",
    "ug": "维吾尔语",
}

LANGUAGE_ORDER = (
    "zh",
    "zh-Hant",
    "yue",
    "en",
    "fr",
    "pt",
    "es",
    "ja",
    "tr",
    "ru",
    "ar",
    "ko",
    "th",
    "it",
    "de",
    "vi",
    "ms",
    "id",
    "tl",
    "hi",
    "pl",
    "cs",
    "nl",
    "km",
    "my",
    "fa",
    "gu",
    "ur",
    "te",
    "mr",
    "he",
    "bn",
    "ta",
    "uk",
    "bo",
    "kk",
    "mn",
    "ug",
)

LANGUAGE_OPTIONS = [f"{code} | {LANGUAGE_NAMES_ZH.get(code, code)}" for code in LANGUAGE_ORDER]

CH_PROMPT = (
    "将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
    "{source_text}"
)

EN_PROMPT = (
    "Translate the following segment into {target_language}, without additional explanation.\n\n"
    "{source_text}"
)

MODEL_ROOT = Path("models") / "HY-MT1.5"
_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
_MODEL_LOCK = threading.Lock()


def _normalize_language_code(choice: str) -> str:
    if "|" in choice:
        return choice.split("|", 1)[0].strip()
    return choice.strip()


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _available_model_names() -> Tuple[str, ...]:
    root = Path(folder_paths.base_path) / MODEL_ROOT
    if not root.exists():
        return tuple()
    names = [child.name for child in root.iterdir() if child.is_dir()]
    return tuple(sorted(names))


def _path_from_name(name: str) -> Path:
    base = Path(folder_paths.base_path) / MODEL_ROOT
    path = base / name
    if not path.exists():
        raise RuntimeError(
            f"Model directory '{name}' not found under {base}. "
            "Place weights under models/HY-MT1.5/<model>."
        )
    return path


def _model_candidates() -> Tuple[Path, ...]:
    """Return candidate model directories in preference order."""
    base = Path(folder_paths.base_path)
    root = base / MODEL_ROOT
    candidates = []
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                candidates.append(child)
    return tuple(candidates)


def _resolve_model_path() -> Path:
    for path in _model_candidates():
        if (path / "config.json").exists() or (path / "tokenizer_config.json").exists():
            return path
    raise RuntimeError(
        "HY-MT1.5 model not found. "
        "Place weights under models/HY-MT1.5/<model> (e.g. HY-MT1.5-1.8B)."
    )


def _load_model(model_path: Path) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if not model_path.exists():
        raise RuntimeError(
            f"HY-MT1.5 model not found at {model_path}. "
            "Place weights under models/HY-MT1.5/<model>."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _get_model(model_path: Path | None = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    target_path = model_path or _resolve_model_path()
    cache_key = str(target_path.resolve())
    with _MODEL_LOCK:
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = _load_model(target_path)
        return _MODEL_CACHE[cache_key]


def _get_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _language_name_for_prompt(code: str, use_chinese_prompt: bool) -> str:
    if use_chinese_prompt:
        return LANGUAGE_NAMES_ZH.get(code, code)
    return LANGUAGE_NAMES_EN.get(code, code)


def _build_prompt(text: str, target_code: str) -> Tuple[str, bool]:
    use_chinese_prompt = _contains_cjk(text) or target_code.startswith("zh") or target_code == "yue"
    prompt = (
        CH_PROMPT if use_chinese_prompt else EN_PROMPT
    ).format(target_language=_language_name_for_prompt(target_code, use_chinese_prompt), source_text=text)
    return prompt, use_chinese_prompt


class HYMT15Loader:
    @classmethod
    def INPUT_TYPES(cls):
        options = list(_available_model_names())
        if not options:
            options = ["<missing models/HY-MT1.5>"]
        return {
            "required": {
                "model": (options, {"default": options[0]}),
            }
        }

    RETURN_TYPES = ("HYMT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "HY-MT1.5"

    def load(self, model: str):
        if model.startswith("<missing"):
            raise RuntimeError("No models found. Place weights under models/HY-MT1.5/<model>.")
        path = _path_from_name(model)
        tokenizer, model = _get_model(path)
        return ((tokenizer, model),)


class HYMT15Translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "target_language": (LANGUAGE_OPTIONS, {"default": "en | 英语"}),

            },
            "optional": {
                "model": ("HYMT_MODEL", {}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation",)
    FUNCTION = "translate"
    CATEGORY = "HY-MT1.5"

    def translate(
        self,
        target_language: str,
        text: str,
        model=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.6,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        if not text:
            raise RuntimeError("Input text is empty.")

        target_code = _normalize_language_code(target_language)
        if model is not None:
            try:
                tokenizer, loaded_model = model
            except Exception as exc:
                raise RuntimeError("Invalid model input. Use the HY-MT1.5 Loader node or leave empty.") from exc
            model_ref = loaded_model
        else:
            tokenizer, model_ref = _get_model()
        prompt, _ = _build_prompt(text, target_code)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(_get_device(model_ref))

        with torch.inference_mode():
            output_ids = model_ref.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        generated_tokens = output_ids[0][input_ids.shape[-1] :]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return (translation,)


NODE_CLASS_MAPPINGS = {
    "HY-MT1.5 Loader": HYMT15Loader,
    "HY-MT1.5 Translator": HYMT15Translator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HY-MT1.5 Loader": "HY-MT1.5 Loader",
    "HY-MT1.5 Translator": "HY-MT1.5 Translator",
}
