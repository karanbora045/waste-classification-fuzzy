import html
import json
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image

try:
    from config import (
        STREAMLIT_CONFIG,
        CLASS_NAMES,
        CLASS_DESCRIPTIONS,
        SUPPORTED_FORMATS,
        CONFIDENCE_THRESHOLD,
        RESULTS_DIR,
    )
    from src.predict import WasteClassifier
    from utils.image_utils import validate_image_file

    PROJECT_IMPORT_ERROR = None
except Exception as exc:
    PROJECT_IMPORT_ERROR = str(exc)
    STREAMLIT_CONFIG = {
        "page_title": "Smart Waste Classification",
        "page_icon": "♻️",
        "layout": "wide",
        "initial_sidebar_state": "collapsed",
    }
    CLASS_NAMES = []
    CLASS_DESCRIPTIONS = {}
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    CONFIDENCE_THRESHOLD = 0.5
    RESULTS_DIR = "results"
    WasteClassifier = None
    validate_image_file = None


st.set_page_config(
    page_title=STREAMLIT_CONFIG.get("page_title", "Smart Waste Classification"),
    page_icon=STREAMLIT_CONFIG.get("page_icon", "♻️"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #090d11;
            --panel: rgba(20, 27, 33, 0.86);
            --panel-strong: #151d23;
            --line: rgba(180, 204, 214, 0.14);
            --text: #eef4f7;
            --muted: #9aaab4;
            --subtle: #61717b;
            --accent: #65d2ad;
            --accent-soft: rgba(101, 210, 173, 0.13);
            --warning: #e8bd72;
        }

        * { box-sizing: border-box; }

        .stApp {
            background:
                radial-gradient(circle at 8% -10%, rgba(48, 113, 103, 0.18), transparent 32rem),
                radial-gradient(circle at 95% 0%, rgba(42, 81, 111, 0.16), transparent 30rem),
                var(--bg);
            color: var(--text);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] { background: transparent; }

        [data-testid="stMainBlockContainer"],
        .main .block-container {
            max-width: 1320px;
            padding: 0.7rem 1.5rem 2rem;
        }

        [data-testid="stHeader"] { display: none; }
        #MainMenu, footer { visibility: hidden; }

        .workspace-bar {
            min-height: 52px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            font: 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }

        .workspace-left, .workspace-right, .inline {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .workspace-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 25px;
            height: 25px;
            border: 1px solid var(--line);
            border-radius: 6px;
            color: var(--text);
            background: rgba(255,255,255,0.06);
            font-weight: 700;
        }

        .workspace-name { color: var(--text); font-weight: 650; }
        .workspace-muted { color: var(--subtle); }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: rgba(255,255,255,0.025);
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 0.9rem;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 550;
        }

        .stTabs [data-baseweb="tab"]:hover { color: var(--text); }
        .stTabs [aria-selected="true"] { color: var(--text) !important; }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--text); }
        .stTabs [data-baseweb="tab-border"] { background: var(--line); }

        .page-intro {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.15rem 0 0.25rem;
        }

        .eyebrow {
            color: var(--subtle);
            font: 0.68rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .page-title {
            margin: 0.28rem 0 0;
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 680;
        }

        .page-copy {
            margin: 0.35rem 0 0;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .stat-card, .panel, .node, .mini-card {
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025));
            border-radius: 7px;
        }

        .stat-card { min-height: 106px; padding: 0.85rem; }
        .stat-label, .meta, .mini-label {
            color: var(--subtle);
            font: 0.64rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .stat-value {
            margin-top: 0.4rem;
            color: var(--text);
            font: 1.35rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }

        .stat-note { margin-top: 0.22rem; color: var(--subtle); font-size: 0.68rem; }
        .panel { padding: 1rem; }
        .panel-title {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding-bottom: 0.7rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid var(--line);
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 650;
        }

        .panel-copy { color: #c4cfd5; font-size: 0.76rem; line-height: 1.65; }
        .spec-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.45rem; margin-top: 0.8rem; }
        .mini-card { padding: 0.65rem; background: rgba(0,0,0,0.14); }
        .mini-value { display: block; margin-top: 0.25rem; color: var(--text); font: 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .node { display: flex; gap: 0.7rem; align-items: flex-start; padding: 0.62rem; margin: 0.42rem 0; background: rgba(0,0,0,0.14); }
        .node-index { width: 22px; color: var(--subtle); font: 0.65rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .node-main { flex: 1; }
        .node-name { display: flex; justify-content: space-between; gap: 1rem; color: #dce6ea; font: 600 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .node-shape { color: var(--accent); white-space: nowrap; }
        .node-desc { display: block; margin-top: 0.25rem; color: var(--subtle); font-size: 0.68rem; line-height: 1.45; }

        .class-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.62rem 0; border-bottom: 1px solid rgba(180,204,214,0.1); }
        .class-row:last-child { border-bottom: 0; }
        .class-name { color: #dce6ea; font-size: 0.76rem; }
        .class-note { display: block; margin-top: 0.18rem; color: var(--subtle); font-size: 0.66rem; }
        .class-count { color: #aebbc2; font: 0.64rem ui-monospace, SFMono-Regular, Menlo, monospace; }

        .prediction-banner { display: flex; justify-content: space-between; gap: 1rem; padding: 0.9rem; border: 1px solid var(--line); border-radius: 6px; background: var(--panel-strong); }
        .prediction-label, .prediction-value { font: 700 1.2rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .prediction-label { color: var(--text); }
        .prediction-value { color: var(--accent); }
        .prob-row { margin: 0.7rem 0; }
        .prob-head { display: flex; justify-content: space-between; color: #b9c6cc; font: 0.72rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .prob-track { height: 6px; margin-top: 0.3rem; overflow: hidden; border-radius: 999px; background: #263139; }
        .prob-fill { height: 100%; border-radius: inherit; background: #687782; }
        .prob-fill-top { background: var(--accent); }

        .matrix { overflow-x: auto; }
        .matrix-grid { min-width: 650px; display: grid; grid-template-columns: 1.35fr repeat(6, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); }
        .matrix-cell { min-height: 35px; display: flex; align-items: center; justify-content: center; padding: 0.35rem; background: #10171c; color: #9eacb4; font: 0.66rem ui-monospace, SFMono-Regular, Menlo, monospace; text-align: center; }
        .matrix-head { background: #192229; color: var(--subtle); }
        .matrix-label { justify-content: flex-start; color: #c8d3d9; font-weight: 600; }
        .matrix-diagonal { background: rgba(101,210,173,0.15); color: #a8efd5; font-weight: 700; }
        .matrix-warn { background: #222c33; color: #c7d1d6; }

        .stButton > button {
            border: 1px solid var(--line);
            border-radius: 5px;
            color: var(--muted);
            background: rgba(255,255,255,0.035);
            font-size: 0.72rem;
        }

        .stButton > button:hover { border-color: var(--accent); color: var(--text); background: var(--accent-soft); }
        [data-testid="stFileUploader"] section { border: 1px dashed var(--line); border-radius: 7px; background: rgba(255,255,255,0.035); }
        [data-testid="stImage"] img { border-radius: 5px; }
        .stAlert { border-radius: 6px; }
        pre { border: 1px solid var(--line) !important; border-radius: 6px !important; background: #05080a !important; }

        @media (max-width: 800px) {
            [data-testid="stMainBlockContainer"], .main .block-container { padding: 0.5rem 0.8rem 1.5rem; }
            .workspace-right { display: none; }
            .spec-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics():
    metrics_path = Path(RESULTS_DIR) / "evaluation_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def load_report():
    report_path = Path(RESULTS_DIR) / "classification_report.txt"
    if not report_path.exists():
        return None
    try:
        return report_path.read_text(encoding="utf-8")
    except OSError:
        return None


def result_path(name):
    return Path(RESULTS_DIR) / name


@st.cache_resource(show_spinner=False)
def load_runtime():
    if WasteClassifier is None:
        return None, False, PROJECT_IMPORT_ERROR

    classifier = WasteClassifier()
    try:
        loaded = bool(classifier.load())
        if loaded:
            return classifier, True, None
        return classifier, False, "The classifier could not load the model files."
    except Exception as exc:
        return classifier, False, str(exc)


def get_model_info(classifier, model_loaded):
    if not model_loaded or classifier is None:
        return {}
    try:
        info = classifier.get_model_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def safe_text(value):
    return html.escape(str(value))


def render_workspace_header(model_info):
    model_name = model_info.get("model_name", model_info.get("name", "Waste Classifier"))
    params = model_info.get("total_params")
    params_text = f"{int(params):,}" if isinstance(params, (int, float)) else "—"

    st.markdown(
        f"""
        <div class="workspace-bar">
            <div class="workspace-left">
                <span class="workspace-mark">W</span>
                <span class="workspace-muted">models</span>
                <span class="workspace-muted">/</span>
                <span class="workspace-name">{safe_text(model_name)}</span>
            </div>
            <div class="workspace-right">
                <span>classes: <b style="color:#c8d3d9">{len(CLASS_NAMES) or "—"}</b></span>
                <span>•</span>
                <span>parameters: <b style="color:#65d2ad">{params_text}</b></span>
                <span class="workspace-muted">Streamlit runtime</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow, title, copy):
    st.markdown(
        f"""
        <div class="page-intro">
            <div>
                <div class="eyebrow">{safe_text(eyebrow)}</div>
                <div class="page-title">{safe_text(title)}</div>
                <div class="page-copy">{safe_text(copy)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title, meta=""):
    st.markdown(
        f"""
        <div class="panel-title">
            <span>{safe_text(title)}</span>
            <span class="meta">{safe_text(meta)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label, value, note):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="inline">
                <span class="stat-label">{safe_text(label)}</span>
            </div>
            <div class="stat-value">{safe_text(value)}</div>
            <div class="stat-note">{safe_text(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(model_loaded, model_info, metrics):
    page_intro(
        "Live model registry / model card",
        "Waste Classifier · Runtime Overview",
        "This page reads the current model, configuration, and evaluation artifacts from your project.",
    )

    accuracy = metrics.get("accuracy") if metrics else None
    macro_f1 = metrics.get("macro_avg", {}).get("f1_score") if metrics else None
    params = model_info.get("total_params") if model_info else None

    metric_cols = st.columns(4)
    stats = [
        ("Test Accuracy", f"{accuracy:.2%}" if accuracy is not None else "—", "evaluation metrics"),
        ("Macro F1-Score", f"{macro_f1:.2%}" if macro_f1 is not None else "—", "evaluation metrics"),
        ("Classes", str(model_info.get("num_classes", len(CLASS_NAMES))) if model_loaded else "—", "loaded model"),
        ("Total Parameters", f"{int(params):,}" if params is not None else "—", "loaded model"),
    ]
    for column, stat in zip(metric_cols, stats):
        with column:
            stat_card(*stat)

    st.write("")
    left, right = st.columns([2, 1], gap="large")

    with left:
        panel_title("Model Description & Runtime Setup", "live project configuration")
        description = (
            "The current model is loaded from the project runtime and predictions are generated "
            "from the uploaded image.."
        )
        st.markdown(f'<div class="panel-copy">{description}</div>', unsafe_allow_html=True)

        dynamic_specs = [
            ("Classes", model_info.get("num_classes", len(CLASS_NAMES)) if model_loaded else "—"),
            ("Parameters", f"{int(params):,}" if params is not None else "—"),
            ("Formats", ", ".join("." + item for item in SUPPORTED_FORMATS)),
        ]
        specs_html = '<div class="spec-grid">'
        for label, value in dynamic_specs:
            specs_html += (
                f'<div class="mini-card"><span class="mini-label">{safe_text(label)}</span>'
                f'<span class="mini-value">{safe_text(value)}</span></div>'
            )
        specs_html += "</div>"
        st.markdown(specs_html, unsafe_allow_html=True)

    with right:
        panel_title("Target Classes", "from config.py")
        if CLASS_DESCRIPTIONS:
            for name, description in CLASS_DESCRIPTIONS.items():
                st.markdown(
                    f"""
                    <div class="class-row">
                        <div>
                            <span class="class-name">{safe_text(str(name).capitalize())}</span>
                            <span class="class-note">{safe_text(description)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif CLASS_NAMES:
            for name in CLASS_NAMES:
                st.markdown(
                    f'<div class="class-row"><span class="class-name">{safe_text(str(name).capitalize())}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No classes were loaded from config.py.")

        st.write("")
        if metrics:
            st.markdown(
                '<div class="panel"><div class="panel-title"><span>Evaluation Source</span></div>'
                '<div class="panel-copy">Metrics shown in this dashboard are read from '
                '<b style="color:#eef4f7">results/evaluation_metrics.json</b>.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No live evaluation_metrics.json was found.")


def get_image_from_input():
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_FORMATS,
        help="The image is sent to the loaded WasteClassifier.",
        key="live_upload",
    )

    if uploaded_file is None:
        return None, None

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")
        return None, None

    if validate_image_file is not None:
        try:
            uploaded_file.seek(0)
            valid, error_message = validate_image_file(uploaded_file)
            if not valid:
                st.error(f"Invalid image: {error_message}")
                return None, None
        except Exception:
            pass

    return image, "upload"


def render_probability_bars(probabilities):
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for index, (name, probability) in enumerate(ordered):
        fill_class = "prob-fill prob-fill-top" if index == 0 else "prob-fill"
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-head"><span>{safe_text(str(name).capitalize())}</span><span>{float(probability):.2%}</span></div>
                <div class="prob-track"><div class="{fill_class}" style="width:{float(probability) * 100:.2f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_live_payload(result, elapsed_ms, source):
    probabilities = result.get("all_probabilities", {})
    predicted_class = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0.0)
    return {
        "source": source,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence,
        },
        "probabilities": probabilities,
        "timing": {
            "inference_ms": round(elapsed_ms, 2),
        },
    }


def render_playground(classifier, model_loaded):
    page_intro(
        "Live inference",
        "Run a Real Prediction",
        "Upload an image. Results below are generated by the current WasteClassifier.",
    )

    if not model_loaded:
        st.error("The live model is not loaded. Check the model path and runtime error in the sidebar.")
        return

    image, source = get_image_from_input()
    if image is None:
        st.info("Choose an image to start a live prediction.")
        return

    started = time.perf_counter()
    with st.spinner("Running live inference..."):
        result = classifier.predict_pil_image(image)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not isinstance(result, dict) or "error" in result:
        st.error(f"Prediction failed: {result.get('error', 'Unknown prediction error') if isinstance(result, dict) else result}")
        return

    probabilities = result.get("all_probabilities", {})
    predicted_class = result.get("predicted_class", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    confidence_text = result.get("confidence_percentage", f"{confidence:.2%}")

    left, right = st.columns([5, 7], gap="large")
    with left:
        panel_title("Input Image")
        st.image(image, use_container_width=True)

    with right:
        if "output_view" not in st.session_state:
            st.session_state.output_view = "Formatted"
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            if st.button(
                "Formatted Output",
                key="formatted_output_view",
                use_container_width=True,
                type="primary" if st.session_state.output_view == "Formatted" else "secondary",
            ):
                st.session_state.output_view = "Formatted"
                st.rerun()
        with view_col2:
            if st.button(
                "JSON Raw",
                key="json_output_view",
                use_container_width=True,
                type="primary" if st.session_state.output_view == "JSON" else "secondary",
            ):
                st.session_state.output_view = "JSON"
                st.rerun()
        view = st.session_state.output_view
        if view == "Formatted":
            st.markdown(
                f"""
                <div class="prediction-banner">
                    <div><div class="meta">Predicted Label</div><div class="prediction-label">{safe_text(str(predicted_class).capitalize())}</div></div>
                    <div style="text-align:right"><div class="meta">Confidence</div><div class="prediction-value">{safe_text(confidence_text)}</div></div>
                </div>
                <div class="meta" style="margin-top:1.2rem">Live Class Probability Distribution</div>
                """,
                unsafe_allow_html=True,
            )
            if probabilities:
                render_probability_bars(probabilities)
            else:
                st.warning("The prediction did not return probability values.")
            if result.get("low_confidence"):
                st.warning(f"Prediction is below the configured confidence threshold of {CONFIDENCE_THRESHOLD:.0%}.")
        else:
            st.code(json.dumps(build_live_payload(result, elapsed_ms, source), indent=2, default=str), language="json")


def render_benchmarks(metrics):
    page_intro(
        "Live evaluation artifacts",
        "Model Evaluation & Validation Report",
        "Every value and image below is read from the current results directory.",
    )

    if metrics is None:
        st.warning(f"No evaluation_metrics.json found in {RESULTS_DIR}. Run the evaluation pipeline first.")
        return

    per_class = metrics.get("per_class", {})
    rows = []
    for class_name in CLASS_NAMES:
        values = per_class.get(class_name, {})
        rows.append(
            {
                "Class": str(class_name).capitalize(),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1-Score": values.get("f1_score"),
                "Support": values.get("support"),
            }
        )

    if rows:
        panel_title("Per-Class Classification Report", "from evaluation_metrics.json")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.write("")
    image_columns = st.columns(3)
    artifact_names = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("accuracy_graph.png", "Accuracy Curves"),
        ("loss_graph.png", "Loss Curves"),
    ]
    for column, (filename, caption) in zip(image_columns, artifact_names):
        with column:
            path = result_path(filename)
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{filename} not found.")

    st.write("")
    report = load_report()
    if report:
        with st.expander("Detailed Classification Report"):
            st.code(report, language="text")
    else:
        st.info(f"No classification_report.txt found in {RESULTS_DIR}.")


def main():
    with st.spinner("Loading the live model runtime..."):
        classifier, model_loaded, _ = load_runtime()

    model_info = get_model_info(classifier, model_loaded)
    render_workspace_header(model_info)
    render_playground(classifier, model_loaded)


if __name__ == "__main__":
    main()
