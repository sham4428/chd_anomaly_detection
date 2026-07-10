import os
from typing import Dict, List, Tuple

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("CHD_API_BASE_URL", "http://127.0.0.1:7860")


def build_upload_files(
    ecg_hea,
    ecg_dat,
    pcg_file,
    cxr_file,
) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    files = []
    if ecg_hea is not None and ecg_dat is not None:
        files.append(("ecg_hea", (ecg_hea.name, ecg_hea.getvalue(), "application/octet-stream")))
        files.append(("ecg_dat", (ecg_dat.name, ecg_dat.getvalue(), "application/octet-stream")))
    if pcg_file is not None:
        files.append(("pcg_file", (pcg_file.name, pcg_file.getvalue(), pcg_file.type or "audio/wav")))
    if cxr_file is not None:
        files.append(("cxr_file", (cxr_file.name, cxr_file.getvalue(), cxr_file.type or "image/png")))
    return files


def post_multimodal(api_base_url: str, files, timeout: int = 120) -> Dict:
    endpoint = api_base_url.rstrip("/") + "/api/predict/multimodal"
    response = requests.post(endpoint, files=files, timeout=timeout)
    try:
        payload = response.json()
    except ValueError:
        payload = {"code": "BAD_RESPONSE", "message": response.text}
    if response.status_code >= 400:
        message = payload.get("message") or payload.get("error") or response.text
        raise RuntimeError(f"{response.status_code}: {message}")
    return payload


def render_fusion(fusion: Dict):
    score = fusion.get("fusion_score")
    confidence = fusion.get("confidence")
    risk = fusion.get("risk_level", "unknown")
    is_abnormal = fusion.get("is_abnormal")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fusion Score", "N/A" if score is None else f"{score:.3f}")
    col2.metric("Risk Level", str(risk).upper())
    col3.metric("Confidence", "N/A" if confidence is None else f"{confidence:.2f}")
    col4.metric("Abnormal", "N/A" if is_abnormal is None else ("Yes" if is_abnormal else "No"))

    st.caption(fusion.get("recommendation", ""))
    with st.expander("Fusion details", expanded=True):
        st.json(fusion)


def render_modality_results(results: List[Dict]):
    if not results:
        return

    st.subheader("Modality Results")
    rows = []
    for item in results:
        rows.append(
            {
                "modality": item.get("modality"),
                "mse": item.get("mse"),
                "threshold": item.get("threshold"),
                "score": item.get("score"),
                "risk_level": item.get("risk_level"),
                "is_abnormal": item.get("is_abnormal"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Raw modality payload"):
        st.json(results)


st.set_page_config(
    page_title="CHD Multimodal Screening Demo",
    page_icon="CHD",
    layout="wide",
)

st.title("CHD Multimodal Screening Demo")
st.caption("Upload ECG, PCG, and/or CXR files. This frontend calls the Flask API and displays the returned screening result.")

with st.sidebar:
    st.header("API")
    api_base_url = st.text_input("Base URL", DEFAULT_API_BASE_URL)
    st.caption("Start the API first with `python app.py`.")
    st.divider()
    st.header("Inputs")
    st.caption("ECG requires a matching WFDB `.hea` and `.dat` pair. PCG and CXR are optional.")

st.subheader("Upload Files")
ecg_col, pcg_col, cxr_col = st.columns(3)

with ecg_col:
    st.markdown("**ECG**")
    ecg_hea = st.file_uploader("ECG .hea", type=["hea"])
    ecg_dat = st.file_uploader("ECG .dat", type=["dat"])

with pcg_col:
    st.markdown("**PCG**")
    pcg_file = st.file_uploader("Heart sound audio", type=["wav", "mp3", "flac", "m4a", "ogg"])

with cxr_col:
    st.markdown("**CXR**")
    cxr_file = st.file_uploader("Chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])

files = build_upload_files(ecg_hea, ecg_dat, pcg_file, cxr_file)
ecg_half_uploaded = (ecg_hea is None) ^ (ecg_dat is None)

if ecg_half_uploaded:
    st.warning("ECG must include both `.hea` and `.dat`; the incomplete ECG upload will not be submitted.")

submit = st.button("Run Screening", type="primary", disabled=not files or ecg_half_uploaded)

if submit:
    with st.spinner("Calling API..."):
        try:
            payload = post_multimodal(api_base_url, files)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at `{api_base_url}`. Start it with `python app.py` first.")
        except requests.exceptions.Timeout:
            st.error("API request timed out.")
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected frontend error: {exc}")
        else:
            st.success("Prediction complete.")
            render_fusion(payload.get("fusion", {}))
            render_modality_results(payload.get("results", []))

st.divider()
st.caption(
    "Research/demo use only. This system must not replace physician diagnosis or medical decision-making."
)
