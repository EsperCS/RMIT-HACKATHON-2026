"""
src/text_features.py - De-obfuscation & Explicit Reflexivity Features (arXiv:2510.01644)
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import re
import math
import base64
import codecs
import urllib.parse
from typing import Tuple, List, Dict
import numpy as np


# -----------------------------------------------------------------------------
# 1. Obfuscation Patterns & De-obfuscation Engine
# -----------------------------------------------------------------------------

ZERO_WIDTH_CHARS_REGEX = re.compile(r"[\u200B-\u200D\uFEFF\u202A-\u202E]")
BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
HEX_PATTERN = re.compile(r"(?:(?:0x|\\x)?[0-9a-fA-F]{2}[\s,]*){6,}")
DOTTED_WORD_PATTERN = re.compile(r"\b([a-zA-Z])(?:\.|\s)([a-zA-Z])(?:\.|\s)([a-zA-Z])(?:\.|\s)([a-zA-Z]+)\b")

LEETSPEAK_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '@': 'a', '$': 's', '!': 'i'
}

SUSPICIOUS_DECODED_KEYWORDS = [
    "hack", "bypass", "exploit", "malware", "ransomware", "bomb", "kill",
    "attack", "password", "root", "unauthorized", "vulnerability", "độc hại",
    "tấn công", "chiếm quyền", "vũ khí"
]


def clean_zero_width_chars(text: str) -> str:
    """Removes invisible zero-width and bidirectional control characters."""
    return ZERO_WIDTH_CHARS_REGEX.sub("", text)


def reconstruct_spaced_words(text: str) -> str:
    """Reconstructs deliberately separated words like 'b.o.m.b' or 'k i l l'."""
    def replacer(match):
        return "".join([g for g in match.groups() if g is not None])
    # Run twice to catch overlapping dotted characters
    cleaned = DOTTED_WORD_PATTERN.sub(replacer, text)
    return cleaned


def try_decode_base64(candidate: str) -> str:
    """Attempts to decode a base64 string if it contains valid UTF-8 text."""
    candidate = candidate.strip()
    if len(candidate) < 12:
        return ""
    try:
        decoded_bytes = base64.b64decode(candidate, validate=True)
        decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
        if len(decoded_str) >= 4 and any(c.isalnum() for c in decoded_str):
            return decoded_str
    except Exception:
        pass
    return ""


def try_decode_hex(candidate: str) -> str:
    """Attempts to decode hex representations like \\x68\\x61\\x63\\x6b or 0x68..."""
    cleaned = re.sub(r"[0x\\s,]+", "", candidate)
    if len(cleaned) < 8 or len(cleaned) % 2 != 0:
        return ""
    try:
        decoded_bytes = bytes.fromhex(cleaned)
        decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
        if len(decoded_str) >= 3 and any(c.isalnum() for c in decoded_str):
            return decoded_str
    except Exception:
        pass
    return ""


def try_decode_rot13(text: str) -> str:
    """Applies ROT13 and checks if it reveals suspicious adversarial keywords."""
    try:
        rot = codecs.decode(text, "rot_13")
        rot_lower = rot.lower()
        if any(kw in rot_lower for kw in SUSPICIOUS_DECODED_KEYWORDS):
            return rot
    except Exception:
        pass
    return ""


def deobfuscate_prompt(text: str) -> Tuple[str, bool]:
    """
    Normalizes text, strips zero-width chars, reconstructs dotted words,
    detects Base64/Hex/URL/ROT13 encoding, and appends '[DECODED]: ...' if found.
    Returns: (processed_text, was_obfuscated_flag)
    """
    if not text:
        return "", False

    was_obfuscated = False
    cleaned_text = clean_zero_width_chars(text)
    if cleaned_text != text:
        was_obfuscated = True

    reconstructed = reconstruct_spaced_words(cleaned_text)
    if reconstructed != cleaned_text:
        was_obfuscated = True
        cleaned_text = reconstructed

    # URL Unquoting
    unquoted = urllib.parse.unquote(cleaned_text)
    if unquoted != cleaned_text:
        was_obfuscated = True
        cleaned_text = unquoted

    decoded_fragments = []

    # Detect Base64
    for match in BASE64_PATTERN.finditer(cleaned_text):
        token = match.group(0)
        dec = try_decode_base64(token)
        if dec and dec not in decoded_fragments:
            decoded_fragments.append(dec)
            was_obfuscated = True

    # Detect Hex
    for match in HEX_PATTERN.finditer(cleaned_text):
        token = match.group(0)
        dec = try_decode_hex(token)
        if dec and dec not in decoded_fragments:
            decoded_fragments.append(dec)
            was_obfuscated = True

    # Detect ROT13
    rot_dec = try_decode_rot13(cleaned_text)
    if rot_dec and rot_dec not in decoded_fragments:
        decoded_fragments.append(rot_dec)
        was_obfuscated = True

    if decoded_fragments:
        cleaned_text = f"{cleaned_text} [DECODED]: " + " | ".join(decoded_fragments)

    return cleaned_text, was_obfuscated


# -----------------------------------------------------------------------------
# 2. Explicit Reflexivity Features (Hawkins et al., arXiv:2510.01644)
# -----------------------------------------------------------------------------

FORMAT_FORCING_KEYWORDS = [
    "start your response with", "do not say", "ignore previous instructions",
    "bắt đầu bằng", "làm theo hướng dẫn trong ảnh", "you must answer",
    "always reply with", "never apologize", "hypothetical scenario",
    "in this fictional story", "dan mode", "developer mode", "jailbreak",
    "bỏ qua các hạn chế", "không được từ chối", "hãy đóng vai", "hãy làm theo"
]

SYSTEM_REFLEXIVITY_PRONOUNS = [
    "you are", "your instructions", "your system prompt", "bạn là",
    "hệ thống", "system:", "assistant:", "as an ai", "your rules",
    "safety policy", "chính sách an toàn", "quy định", "bạn không có cảm xúc"
]


def compute_shannon_entropy(text: str) -> float:
    """Computes Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return float(entropy)


def extract_reflexivity_features(text: str) -> np.ndarray:
    """
    Extracts a fixed-dimensional numerical feature vector measuring prompt reflexivity,
    output forcing constraints, syntax anomalies, and entropy.
    Dimensionality: 9 features.
    """
    if not text:
        return np.zeros(9, dtype=np.float32)

    cleaned_text, was_obfuscated = deobfuscate_prompt(text)
    text_lower = cleaned_text.lower()
    char_len = len(cleaned_text)
    word_tokens = text_lower.split()
    word_count = max(len(word_tokens), 1)

    # 1. Output format forcing keyword density
    forcing_hits = sum(1 for kw in FORMAT_FORCING_KEYWORDS if kw in text_lower)
    forcing_density = float(forcing_hits / word_count)

    # 2. System reflexivity pronoun density
    reflexivity_hits = sum(1 for kw in SYSTEM_REFLEXIVITY_PRONOUNS if kw in text_lower)
    reflexivity_density = float(reflexivity_hits / word_count)

    # 3. Special character ratio
    special_char_count = sum(1 for c in cleaned_text if not c.isalnum() and not c.isspace())
    special_char_ratio = float(special_char_count / max(char_len, 1))

    # 4. Punctuation density (dots, colons, brackets)
    punct_count = sum(1 for c in cleaned_text if c in ":;?![]{}()<>*`~'\"")
    punct_density = float(punct_count / max(char_len, 1))

    # 5. Shannon entropy
    entropy = compute_shannon_entropy(cleaned_text)

    # 6. Length measures (normalized)
    log_char_len = float(math.log1p(char_len))
    log_word_len = float(math.log1p(word_count))

    # 7. Uppercase character ratio
    uppercase_count = sum(1 for c in cleaned_text if c.isupper())
    uppercase_ratio = float(uppercase_count / max(char_len, 1))

    # 8. Obfuscation detection flag
    obfuscation_flag = 1.0 if was_obfuscated else 0.0

    features = np.array([
        forcing_density,
        reflexivity_density,
        special_char_ratio,
        punct_density,
        entropy,
        log_char_len,
        log_word_len,
        uppercase_ratio,
        obfuscation_flag
    ], dtype=np.float32)

    return features


def extract_reflexivity_features_batch(texts: List[str]) -> np.ndarray:
    """
    Vectorized extraction for a batch of strings.
    Returns: np.ndarray of shape (N, 9)
    """
    return np.stack([extract_reflexivity_features(t) for t in texts], axis=0)


def get_reflexivity_feature_dim() -> int:
    """Returns feature dimensionality (9)."""
    return 9

