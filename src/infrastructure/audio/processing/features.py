"""
Voice feature extraction using OpenSMILE.
"""
import os
import re
import subprocess
import logging
from typing import Dict, List, Tuple

from ....config import OPENSMILE_CONFIG_CANDIDATES, CANONICAL_FEATURE_MAPPING

logger = logging.getLogger("audio_features")


def parse_opensmile_arff(arff_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract @attribute names and @data rows from OpenSMILE ARFF-style text.
    Returns (attributes_list, data_lines_list)
    """
    lines = [ln.strip() for ln in arff_text.splitlines() if ln.strip()]
    attrs: List[str] = []
    data_lines: List[str] = []
    in_data = False
    
    for ln in lines:
        low = ln.lower()
        if low.startswith("@attribute"):
            m = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", ln, flags=re.IGNORECASE)
            if m:
                name = m.group(1) or m.group(2) or m.group(3)
                attrs.append(name)
        elif low.startswith("@data"):
            in_data = True
        elif in_data and not (ln.startswith("%") or ln.startswith("#")):
            data_lines.append(ln)
    
    return attrs, data_lines


def read_opensmile_csv_file(csv_path: str) -> Tuple[List[str], List[str]]:
    """Read OpenSMILE CSV/ARFF file and return attributes and data lines."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return parse_opensmile_arff(content)


def extract_voice_features(wav_path: str) -> Dict[str, float]:
    """
    Run SMILExtract on WAV file and return canonical voice features.
    Returns dict with canonical feature names mapped to float values.
    """
    # Find OpenSMILE config
    cfg = next((p for p in OPENSMILE_CONFIG_CANDIDATES if p and os.path.isfile(p)), 
               OPENSMILE_CONFIG_CANDIDATES[1])
    
    wav_abs = os.path.abspath(wav_path)
    csv_candidate = wav_abs.replace(".wav", "_features.csv")
    
    # Check if CSV is stale (older than WAV file) and delete it
    if os.path.isfile(csv_candidate) and os.path.isfile(wav_abs):
        csv_mtime = os.path.getmtime(csv_candidate)
        wav_mtime = os.path.getmtime(wav_abs)
        if csv_mtime < wav_mtime:
            logger.debug(f"Removing stale features CSV: {csv_candidate}")
            os.remove(csv_candidate)

    if not os.path.isfile(wav_abs):
        raise RuntimeError(f"Input WAV not found: {wav_abs}")

    # Run SMILExtract
    cmd = ["SMILExtract", "-C", cfg, "-I", wav_abs, "-O", csv_candidate]
    logger.info("Running SMILExtract: %s", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, text=True, timeout=60)
    except FileNotFoundError:
        raise RuntimeError("SMILExtract not found. Install OpenSMILE or ensure SMILExtract is in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("SMILExtract timed out (60s).")

    # Parse output
    attrs, data_lines = [], []
    if os.path.isfile(csv_candidate):
        try:
            attrs, data_lines = read_opensmile_csv_file(csv_candidate)
            logger.debug("Read OpenSMILE CSV: %s (attrs=%d, rows=%d)", 
                        csv_candidate, len(attrs), len(data_lines))
        except Exception as e:
            logger.warning("Failed to parse OpenSMILE CSV file %s: %s", csv_candidate, e)

    # Fallback to stdout parsing
    if not attrs or not data_lines:
        attrs, data_lines = parse_opensmile_arff(proc.stdout or "")
        logger.debug("Parsed OpenSMILE ARFF from stdout (attrs=%d, rows=%d)", 
                    len(attrs), len(data_lines))

    if not attrs or not data_lines:
        debug_msg = (
            f"SMILExtract did not produce readable CSV/ARFF.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout (truncated):\n{(proc.stdout or '')[:2000]}\n"
            f"csv_exists: {os.path.isfile(csv_candidate)}"
        )
        logger.error(debug_msg)
        raise RuntimeError(debug_msg)

    # Extract first numeric data row
    first_row = next((r for r in data_lines if re.search(r"[0-9]", r)), None)
    if first_row is None:
        raise RuntimeError("No numeric data rows found in OpenSMILE output")

    # Parse values
    if "," in first_row:
        row_vals = [v.strip() for v in first_row.split(",")]
    elif ";" in first_row:
        row_vals = [v.strip() for v in first_row.split(";")]
    else:
        row_vals = [v.strip() for v in first_row.split()]

    # Build name->value mapping
    col_to_val: Dict[str, float] = {}
    for i, name in enumerate(attrs):
        v = row_vals[i].strip().strip('"').strip("'") if i < len(row_vals) else ""
        
        if v.lower() in ("unknown", "na", "nan", "?"):
            col_to_val[name] = 0.0
            continue
            
        try:
            col_to_val[name] = float(v)
        except Exception:
            # Extract numeric substring
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", v)
            col_to_val[name] = float(m.group(0)) if m else 0.0

    # DEBUG: Log parsing details
    logger.info(f"=== DEBUG for {os.path.basename(wav_path)} ===")
    logger.info(f"CSV file: {os.path.basename(csv_candidate)}")
    logger.info(f"Found {len(attrs)} attributes: {attrs[:5]}...")
    logger.info(f"Data lines: {len(data_lines)}")
    logger.info(f"First data row: {first_row[:100]}...")
    logger.info(f"Column mapping attempts:")
    for canon, exact in list(CANONICAL_FEATURE_MAPPING.items())[:5]:  # Show first 5
        found = exact in col_to_val
        value = col_to_val.get(exact, "NOT_FOUND")
        logger.info(f"  {canon} -> {exact}: {found} = {value}")
    logger.info(f"col_to_val has {len(col_to_val)} entries")

    # Extract features using the canonical mapping (now includes all best features)
    features = {}
    
    # Map all canonical features (this now includes the best 29 features for speaker ID)
    for canon, exact in CANONICAL_FEATURE_MAPPING.items():
        if exact in col_to_val:
            features[canon] = float(col_to_val[exact])
    
    # If no mapped features found, fall back to returning available numeric features
    # This handles cases where OpenSMILE output format differs
    if not features:
        logger.warning("No canonical features found, using available numeric features")
        for name, value in col_to_val.items():
            # Skip non-numeric or timestamp columns
            if any(skip in name.lower() for skip in ['time', 'frame', 'name', 'class']):
                continue
            # Use simplified feature names
            simple_name = name.lower().replace('sma3', '').replace('_amean', '').replace('nz', '')
            features[simple_name] = float(value)

    # Write canonical CSV for debugging
    try:
        canonical_csv = wav_abs.replace(".wav", "_features_canonical.csv")
        with open(canonical_csv, "w", encoding="utf-8") as wf:
            wf.write(",".join(features.keys()) + "\n")
            wf.write(",".join(str(features[k]) for k in features.keys()) + "\n")
        logger.debug("Wrote canonical features CSV: %s", canonical_csv)
    except Exception:
        pass

    # Add debugging to detect identical feature extraction
    wav_file_info = f"{os.path.basename(wav_path)} ({os.path.getsize(wav_path)} bytes)"
    logger.info(f"=== FINAL RESULT for {wav_file_info} ===")
    logger.info(f"Extracted {len(features)} features: {list(features.keys())}")
    logger.info(f"Feature values: {features}")
    
    # Check if this looks like repeated/cached data
    feature_hash = hash(tuple(sorted(features.items())))
    logger.info(f"Feature hash: {feature_hash}")
    logger.info(f"=== END DEBUG ===")
    
    return features
