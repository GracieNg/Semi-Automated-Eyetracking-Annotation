import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
from PIL import Image

st.set_page_config(layout="wide")
st.title("Semi-Automated Eye-Tracking Annotation Dashboard")

st.markdown("""
Upload an eye-tracking dataset to automatically generate
behavioural annotations and review recommendations.
""")

uploaded_file = st.file_uploader(
    "Upload gaze data (CSV or TSV)",
    type=["csv", "tsv"]
)

if uploaded_file:
    st.success("File uploaded successfully")

    data_dir = Path("data/ui_upload")
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Preview of uploaded data:")
    df_preview = pd.read_csv(file_path, sep=None, engine="python")
    st.dataframe(df_preview.head())

    # Optional: copy uploaded file into sandbox/raw so the pipeline picks it up
    sandbox_raw = Path("data/sandbox/raw")
    sandbox_raw.mkdir(parents=True, exist_ok=True)
    upload_suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_stem = Path(uploaded_file.name).stem

    if upload_suffix == ".csv":
        df_upload = pd.read_csv(file_path, sep=None, engine="python")
        sandbox_path = sandbox_raw / f"{uploaded_stem}.tsv"
        df_upload.to_csv(sandbox_path, sep="\t", index=False)
        st.info(f"Converted CSV to TSV for preprocessing: {sandbox_path}")
    else:
        sandbox_path = sandbox_raw / uploaded_file.name
        sandbox_path.write_bytes(file_path.read_bytes())
        st.info(f"Copied upload to {sandbox_path}")
    st.session_state["last_uploaded_stem"] = uploaded_stem
    st.session_state.setdefault("uploaded_stems", [])
    if uploaded_stem not in st.session_state["uploaded_stems"]:
        st.session_state["uploaded_stems"].append(uploaded_stem)

show_full_logs = st.checkbox("Show full logs", value=False)

if st.button("Run Semi-Automated Annotation"):
    with st.spinner("Running annotation pipeline..."):
        error_logs = []
        full_logs = []
        data_root = "data/sandbox"
        steps = [
            ["python", "src/preprocessing.py", "--save", "--data-root", data_root],
            ["python", "src/features.py", "--data-root", data_root],
            ["python", "src/model.py", "--data-root", data_root],
            ["python", "src/evaluation.py", "--data-root", data_root],
        ]
        for cmd in steps:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if show_full_logs:
                full_logs.append(f"$ {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
            if result.stderr:
                ignore_substrings = [
                    "UserWarning",
                    "warnings.warn",
                    "Matplotlib created a temporary cache directory",
                    "Matplotlib is building the font cache",
                    ".matplotlib is not a writable directory",
                    "it is highly recommended to set the MPLCONFIGDIR",
                ]
                stderr_lines = [
                    line for line in result.stderr.splitlines()
                    if not any(s in line for s in ignore_substrings)
                ]
                if stderr_lines:
                    error_logs.append("\n".join(stderr_lines).strip())

        if show_full_logs and full_logs:
            with st.expander("Run Pipeline Full Logs", expanded=False):
                st.code("\n\n".join(full_logs).strip(), language="text")

        if error_logs:
            with st.expander("Run Pipeline Errors", expanded=True):
                st.code("\n\n".join(error_logs), language="text")

    st.success("Annotation completed")
    st.session_state["pipeline_ran"] = True

st.header("Active Learning Results")

pred_path = Path("data/sandbox/predictions_with_uncertainty_pool.csv")
if pred_path.exists():
    annotated = pd.read_csv(pred_path)
    st.dataframe(annotated.head())

    st.metric(
        "Samples requiring human review",
        f"{annotated['needs_human_review'].mean()*100:.2f}%"
    )

    st.download_button(
        label="Download Annotated Data",
        data=annotated.to_csv(index=False),
        file_name="annotated_gaze_data.csv",
        mime="text/csv"
    )
else:
    st.info("Run the pipeline to generate predictions_with_uncertainty_pool.csv")

st.header("Processed Data (Labelled)")
processed_dir = Path("data/sandbox/processed")
pipeline_ran = st.session_state.get("pipeline_ran", False)
last_stem = st.session_state.get("last_uploaded_stem")
uploaded_stems = st.session_state.get("uploaded_stems", [])

def _stem_variants(stem: str) -> list[str]:
    variants = {
        stem,
        stem.replace("0", "O"),
        stem.replace("O", "0"),
        stem.replace("P0", "PO"),
        stem.replace("PO", "P0"),
    }
    return [v for v in variants if v]

if pipeline_ran:
    user_processed = []
    for stem in uploaded_stems:
        for variant in _stem_variants(stem):
            candidate = processed_dir / f"{variant}_processed.csv"
            if candidate.exists() and candidate.name not in user_processed:
                user_processed.append(candidate.name)

    if user_processed:
        default_index = 0
        if last_stem:
            for variant in _stem_variants(last_stem):
                name = f"{variant}_processed.csv"
                if name in user_processed:
                    default_index = user_processed.index(name)
                    break

        selected = st.selectbox(
            "Select a processed file",
            options=user_processed,
            index=default_index
        )
        sel_path = processed_dir / selected
        st.caption("Showing processed data from your uploads only.")

        df_proc = pd.read_csv(sel_path)
        st.dataframe(df_proc.head())
        st.download_button(
            label=f"Download {sel_path.name}",
            data=df_proc.to_csv(index=False),
            file_name=sel_path.name,
            mime="text/csv"
        )
    else:
        st.info("No processed files found for your uploads yet.")
elif not pipeline_ran:
    st.info("Run the pipeline to generate processed data.")
else:
    st.info("No processed files found. Run preprocessing first.")

st.header("Model Performance & Analytics")

col1, col2 = st.columns(2)

with col1:
    confusion_path = Path("results/figures/confusion_matrix.png")
    if confusion_path.exists():
        st.image(str(confusion_path), caption="Confusion Matrix")
    else:
        st.info("Confusion matrix not generated yet.")

with col2:
    confidence_path = Path("results/figures/confidence_distribution.png")
    if confidence_path.exists():
        st.image(str(confidence_path), caption="Confidence Distribution")
    else:
        st.info("Confidence distribution not generated yet.")

annotation_path = Path("results/figures/annotation_time_comparison.png")
if annotation_path.exists():
    st.image(str(annotation_path), caption="Annotation Time Comparison")
else:
    st.info("Annotation time comparison not generated yet.")

st.header("Evaluation Tables")

tables = [
    ("Active Learning Summary", Path("results/tables/active_learning_summary.csv")),
    ("Metrics (Macro-F1)", Path("results/tables/metrics.csv")),
    ("Class Report", Path("results/tables/class_report.csv")),
    ("Annotation Time Comparison", Path("results/tables/annotation_time_comparison.csv")),
]

for title, tpath in tables:
    st.subheader(title)
    if tpath.exists():
        st.dataframe(pd.read_csv(tpath))
    else:
        st.info(f"{tpath} not generated yet.")
