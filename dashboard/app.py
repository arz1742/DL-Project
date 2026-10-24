"""
dashboard/app.py — Image Captioning: from Show-and-Tell to attention

Run from the project folder:
    streamlit run dashboard/app.py

Needs the files from the "dashboard bundle" (notebook cell 21): trained models
in runs/, results in results/ and dashboard/data/, and data/processed/tokenizer.json.
"""

import os
import io
import sys
import json
import tempfile

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "data")
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
PROCESSED = os.path.join(ROOT, "data", "processed")
CNN_WEIGHTS = None if os.environ.get("DASH_CNN_WEIGHTS") == "none" else "imagenet"  # 'none' = offline tests

st.set_page_config(page_title="Image Captioning Lab", page_icon="🖼️", layout="wide")

# Reference palette (dataviz skill), categorical slots in fixed order; family -> slot
FAMILY_COLORS = {
    "Part 1: classic decoders": "#2a78d6",
    "Encoder comparison": "#eb6834",
    "Part 2: attention": "#1baf7a",
    "Pretrained reference": "#eda100",
    "Replication": "#e87ba4",
    "Human reference": "#6250d6",
}
METRIC_NAMES = {
    "cider": "CIDEr", "bleu1": "BLEU-1", "bleu4": "BLEU-4", "meteor": "METEOR", "rouge_l": "ROUGE-L",
    "clip_s": "CLIPScore (vs photo)", "refclip_s": "RefCLIPScore", "key_object_recall": "Key-object recall",
    "unique_caption_pct": "Unique captions (%)", "masked_acc": "Token accuracy (padding ignored)",
    "unmasked_acc": "Token accuracy (padding counted)", "top5_acc": "Top-5 next-word accuracy",
}
METRIC_HELP = {
    "cider": "Agreement with what most of the 5 human captions say; the main captioning metric.",
    "bleu1": "Share of single words that also appear in the human captions.",
    "bleu4": "Share of 4-word phrases that also appear in the human captions.",
    "meteor": "Word matching that also credits stems and synonyms.",
    "rouge_l": "Longest shared word sequence with the human captions.",
    "clip_s": "How well the caption matches the PHOTO itself, judged by CLIP ViT-L/14. No reference sentences.",
    "refclip_s": "CLIPScore combined with similarity to the human captions.",
    "key_object_recall": "Does the caption name the photo's main subject (dog, man, girl, ...)?",
    "unique_caption_pct": "Share of the 1000 test photos that get a caption no other photo got.",
}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_json(name: str):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def summary_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for cfg, m in summary["models"].items():
        row = {"config": cfg, "model": m["label"], "family": m["family"], "seeds": m.get("seeds")}
        for k, v in m.get("metrics", {}).items():
            row[k] = v["mean"]
            row[k + "_sd"] = v["sd"]
        rows.append(row)
    return pd.DataFrame(rows)


def thumb(img_id: str, large: bool = False) -> str | None:
    for d in (["l", "s"] if large else ["s"]):
        p = os.path.join(DATA, "img", d, img_id + ".jpg")
        if os.path.exists(p):
            return p
    return None


def need_data():
    st.warning("Results not found. Run notebook cell 21 (dashboard bundle), download the zip from Drive "
               "and unzip it into the project folder (see dashboard/README.md).")


# ----------------------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading ResNet50 (first time only) ...")
def get_resnet():
    from engine import ResNetFeatures
    return ResNetFeatures(CNN_WEIGHTS)


@st.cache_resource(show_spinner="Loading CLIP ...")
def get_clip():
    from src.feature_extraction import ClipFeaturizer
    return ClipFeaturizer()


@st.cache_resource(show_spinner="Loading BLIP-large (first time downloads ~1.9 GB) ...")
def get_blip():
    from engine import Blip
    return Blip()


@st.cache_resource(show_spinner="Loading trained models ...")
def get_models(cfgs: tuple):
    from engine import LiveModel
    from src.tokenizer import CaptionTokenizer
    tok = CaptionTokenizer.load(os.path.join(PROCESSED, "tokenizer.json"))
    return {c: LiveModel(c, RUNS, PROCESSED, tok) for c in cfgs}


def speak_button(text: str, key: str):
    """Read-aloud button using the browser's own speech synthesis (no internet needed)."""
    import streamlit.components.v1 as components
    safe = json.dumps(text)
    components.html(f"""
      <button id="b" style="font:14px system-ui;padding:6px 12px;border-radius:8px;border:1px solid #c3c2b7;
        background:transparent;color:inherit;cursor:pointer">🔊 Read aloud</button>
      <script>document.getElementById('b').onclick=()=>{{speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance({safe}));}};</script>""", height=44)


# ============================================================================= pages
def page_live():
    from engine import available_models, torch_available, heat_overlay
    from src.analysis import word_strip

    st.title("📷 Live captioning")
    st.caption("Upload a photo or use your webcam. Every trained model writes a caption; the attention models "
               "also show where they looked for each word.")
    if not os.path.exists(os.path.join(PROCESSED, "tokenizer.json")):
        need_data()
        return
    avail = available_models(RUNS)
    if not avail:
        st.error("No trained models found in runs/. Unzip the dashboard bundle into the project folder.")
        return

    with st.sidebar:
        st.subheader("Live settings")
        chosen = st.multiselect("Models", list(avail), default=list(avail),
                                format_func=lambda c: avail[c]["label"])
        beam_width = st.select_slider("Beam width", [1, 3, 5, 10], value=5,
                                      help="1 = greedy. Larger keeps more candidate sentences.")
        use_blip = st.toggle("Also ask BLIP-large (2022)", value=False, disabled=not torch_available(),
                             help="Pretrained reference model; downloads ~1.9 GB the first time.")
        if not torch_available():
            st.caption("Install torch + transformers to enable CLIP and BLIP models.")

    src = st.segmented_control("Photo source", ["Upload", "Webcam"], default="Upload")
    file = (st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if src == "Upload" else st.camera_input("Take a photo", label_visibility="collapsed"))
    if file is None:
        st.info("Tip: photos of people or dogs outdoors (parks, beaches, streets, sports) match what the "
                "models were trained on. Indoor scenes, food and other animals show their limits.")
        return
    if not chosen:
        st.warning("Pick at least one model in the sidebar.")
        return

    image = Image.open(file).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, quality=95)
        path = tmp.name

    with st.spinner("Looking at the photo ..."):
        feats = get_resnet()(path)
        if any(avail[c]["needs"] == "clip" for c in chosen):
            feats["clip"] = get_clip()([path])[0]
        models = get_models(tuple(sorted(chosen)))
        outs = {c: models[c].run(feats, beam_width) for c in chosen}
        blip_cap = get_blip()(path) if use_blip else None

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.image(image, use_container_width=True)
    with right:
        method = "greedy" if beam_width == 1 else f"beam {beam_width}"
        for c in chosen:
            with st.container(border=True):
                st.caption(avail[c]["label"])
                st.markdown(f'**{outs[c]["beam"]}**')
                if beam_width > 1 and outs[c]["greedy"] != outs[c]["beam"]:
                    st.caption(f'greedy: {outs[c]["greedy"]}  ·  shown: {method}')
        if blip_cap:
            with st.container(border=True):
                st.caption("BLIP-large (pretrained, 2022)")
                st.markdown(f"**{blip_cap}**")
        best = "hybrid_clip" if "hybrid_clip" in outs else chosen[0]
        speak_button(outs[best]["beam"], "speak")

    heat_models = [c for c in chosen if outs[c].get("maps")]
    if heat_models:
        st.subheader("Where did the model look?")
        hm = st.radio("Model", heat_models, horizontal=True, format_func=lambda c: avail[c]["label"])
        maps = outs[hm]["maps"]
        method = avail[hm]["heatmap"]
        st.caption(f"{'Built-in attention weights' if method == 'attention' else 'Grad-CAM (the model has no attention; this shows which regions its word choice depended on)'}. "
                   "Bright = looked here.")
        words = [m["word"] for m in maps]
        i = st.select_slider("Word", options=list(range(len(words))), format_func=lambda j: words[j], value=0)
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.image(heat_overlay(image, maps[i]["map"]), caption=f'"{words[i]}"', use_container_width=True)
        with c2:
            probs = outs[hm]["probs"]
            if i < len(probs):
                st.markdown(f"**Top 5 guesses for word {i + 1}**")
                dfp = pd.DataFrame(probs[i]["top"], columns=["word", "probability"])
                st.altair_chart(alt.Chart(dfp).mark_bar(cornerRadiusEnd=4, color="#2a78d6", height=18).encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("word:N", sort="-x", title=None),
                    tooltip=["word", alt.Tooltip("probability:Q", format=".1%")]).properties(height=170),
                    use_container_width=True)
        with st.expander("All words in one strip (for slides)"):
            out_png = os.path.join(tempfile.gettempdir(), f"strip_{hm}.png")
            word_strip(image, maps, f'{avail[hm]["label"]}: "{outs[hm]["beam"]}"', out_png)
            st.image(out_png, use_container_width=True)
            with open(out_png, "rb") as f:
                st.download_button("Download strip (PNG)", f, file_name=f"heatmap_{hm}.png")

    with st.expander("Beam search tree (k = 3)"):
        bt = st.radio("Model ", chosen, horizontal=True, format_func=lambda c: avail[c]["label"], key="bt")
        from src.analysis import draw_beamtree as draw
        png = os.path.join(tempfile.gettempdir(), f"tree_{bt}.png")
        draw(outs[bt]["trace"], png)
        st.image(png, use_container_width=True)


def page_overview():
    s = load_json("summary.json")
    st.title("🖼️ Image Captioning Lab")
    st.markdown("#### A photo goes in, a sentence comes out. Which design choices actually matter?")
    if not s:
        need_data()
        return
    m = s["models"]

    def val(cfg, k):
        return m.get(cfg, {}).get("metrics", {}).get(k, {}).get("mean")

    c = st.columns(4)
    base, att, clip = val("hybrid_frozen", "cider"), val("attention", "cider"), val("hybrid_clip", "cider")
    c6 = val("hybrid_finetuned", "cider")
    if base and clip:
        c[0].metric("Better encoder (CLIP vs ResNet50)", f"{clip:.3f} CIDEr", f"{(clip / base - 1):+.0%}")
    if c6 and att:
        c[1].metric("Adding attention", f"{att:.3f} CIDEr", f"{(att / c6 - 1):+.0%} vs best classic")
    if base and c6:
        c[2].metric("Fine-tuning ResNet50", f"{c6:.3f} CIDEr", f"{(c6 / base - 1):+.0%}")
    pp = m.get("paper_protocol", {}).get("metrics", {})
    if pp:
        c[3].metric("Paper-style accuracy", f"{pp['unmasked_acc']['mean']:.2f}",
                    f"honest: {pp['masked_acc']['mean']:.2f}", delta_color="off")

    st.markdown("""
**The story in five findings**
1. **The encoder matters most.** Swapping ResNet50 for CLIP raises CIDEr by about 45%.
2. **Attention is next.** Letting the decoder look at 49 image regions raises CIDEr by about 24% over the best classic model.
3. **Fine-tuning and decoder type help a little** (about 5% each). Paper 1's hybrid LSTM→GRU is no better than a plain GRU.
4. **Paper 1's 0.89 accuracy is mostly padding.** The same kind of model scores 0.79 when blank padding counts and 0.41 when it doesn't.
5. **BLEU usually ranks models correctly** (rank correlation 0.92 with CLIPScore), but punishes models that write in a different style, like BLIP-large.
""")
    st.markdown("**Models compared**")
    df = summary_frame(s)
    show = df[["model", "family", "seeds"] + [k for k in ["bleu4", "cider", "meteor", "clip_s"] if k in df]]
    st.dataframe(show.rename(columns={k: METRIC_NAMES.get(k, k) for k in show.columns}),
                 hide_index=True, use_container_width=True)
    st.caption("Flickr8k: 6,091 training / 1,000 validation / 1,000 test images, 5 human captions each. "
               "Trained models: mean over 3 seeds, beam search k = 5. BLIP: one run, beam 3.")


def bar_chart(df: pd.DataFrame, metric: str, height_per_row: int = 30):
    d = df.dropna(subset=[metric]).copy()
    d["lo"] = d[metric] - d.get(metric + "_sd", 0).fillna(0)
    d["hi"] = d[metric] + d.get(metric + "_sd", 0).fillna(0)
    order = d.sort_values(metric, ascending=False).model.tolist()
    color = alt.Color("family:N", scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
                      legend=alt.Legend(orient="bottom", title=None, columns=3))
    tip = ["model", "family", alt.Tooltip(f"{metric}:Q", format=".3f", title=METRIC_NAMES.get(metric, metric)),
           alt.Tooltip(f"{metric}_sd:Q", format=".3f", title="± sd over seeds"), "seeds"]
    base = alt.Chart(d).encode(y=alt.Y("model:N", sort=order, title=None))
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{metric}:Q", title=METRIC_NAMES.get(metric, metric)), color=color, tooltip=tip)
    err = base.mark_rule(strokeWidth=1.5, color="#52514e").encode(x="lo:Q", x2="hi:Q")
    text = base.mark_text(align="left", dx=6, fontSize=12).encode(
        x=alt.X("hi:Q"), text=alt.Text(f"{metric}:Q", format=".3f"))
    return (bars + err + text).properties(height=alt.Step(height_per_row))


def page_results():
    s = load_json("summary.json")
    st.title("📊 Results")
    if not s:
        need_data()
        return
    df = summary_frame(s)
    metrics = [k for k in METRIC_NAMES if k in df.columns and k not in ("masked_acc", "unmasked_acc", "top5_acc")]
    metric = st.pills("Metric", metrics, default="cider", format_func=lambda k: METRIC_NAMES[k]) or "cider"
    st.caption(METRIC_HELP.get(metric, ""))
    fams = st.multiselect("Show", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    st.altair_chart(bar_chart(df[df.family.isin(fams)], metric), use_container_width=True)
    st.caption("Bars: mean over seeds. Thin lines: ± one standard deviation across seeds.")
    with st.expander("Full table"):
        cols = ["model", "family", "seeds"] + [k for k in METRIC_NAMES if k in df.columns]
        st.dataframe(df[cols].rename(columns=METRIC_NAMES).style.format(precision=3), hide_index=True,
                     use_container_width=True)

    st.subheader("Greedy vs beam search (BLEU-1)")
    dec = s.get("decodings", {})
    rows = []
    for cfg, by in dec.items():
        for d, v in by.items():
            if "bleu1" in v and cfg in s["models"]:
                rows.append({"model": s["models"][cfg]["label"], "decoding": d, "BLEU-1": v["bleu1"]["mean"],
                             "CIDEr": v.get("cider", {}).get("mean")})
    if rows:
        dd = pd.DataFrame(rows)
        dd = dd[dd.decoding.isin(["greedy", "beam3", "beam5"])]
        st.altair_chart(alt.Chart(dd).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("BLEU-1:Q"), y=alt.Y("decoding:N", title=None, sort=["greedy", "beam3", "beam5"]),
            color=alt.Color("decoding:N", scale=alt.Scale(domain=["greedy", "beam3", "beam5"],
                                                          range=["#86b6ef", "#2a78d6", "#184f95"]), legend=None),
            row=alt.Row("model:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["model", "decoding", alt.Tooltip("BLEU-1:Q", format=".3f"), alt.Tooltip("CIDEr:Q", format=".3f")],
        ).properties(height=60), use_container_width=True)
        p1 = s["paper1"]
        st.caption(f"Paper 1 reported greedy → beam 3 BLEU-1 of {p1['bleu1_greedy']['hybrid']} → "
                   f"{p1['bleu1_beam3']['hybrid']} for its hybrid model, on a 40-image sample.")


def page_significance():
    s = load_json("summary.json")
    st.title("✅ Which differences are real?")
    st.markdown("Each test photo is re-sampled 10,000 times (paired bootstrap). If model A beats model B in almost "
                "every re-sample, the gap is **real**; otherwise it could be luck in which photos were picked. "
                "p-values are Holm-corrected for running 11 comparisons.")
    if not s or not s.get("significance"):
        need_data()
        return
    metric = st.segmented_control("Metric", list(s["significance"]), default="cider",
                                  format_func=lambda k: "CIDEr" if k == "cider" else "CLIPScore") or "cider"
    d = pd.DataFrame(s["significance"][metric])
    d["label"] = d.question
    d["verdict"] = np.where(d.p_holm < 0.05, "Real difference", "Within noise")
    d["rel_pct"] = d.rel * 100
    order = d.label.tolist()
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781", strokeDash=[4, 3]).encode(x="x:Q")
    ci = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lo:Q", title=f"Difference in {'CIDEr' if metric == 'cider' else 'CLIPScore'} (A − B), 95% interval"),
        x2="hi:Q", color=alt.Color("verdict:N", scale=alt.Scale(domain=["Real difference", "Within noise"],
                                                               range=["#2a78d6", "#898781"]),
                                   legend=alt.Legend(orient="bottom", title=None)))
    pts = base.mark_point(filled=True, size=90).encode(
        x="diff:Q", color="verdict:N",
        tooltip=["question", alt.Tooltip("a_mean:Q", format=".3f", title="A"),
                 alt.Tooltip("b_mean:Q", format=".3f", title="B"), alt.Tooltip("diff:Q", format="+.3f"),
                 alt.Tooltip("rel_pct:Q", format="+.1f", title="change %"),
                 alt.Tooltip("p_holm:Q", format=".3f", title="p (Holm)"), "verdict"])
    st.altair_chart((zero + ci + pts).properties(height=36 * len(d)), use_container_width=True)
    t = d[["question", "a_mean", "b_mean", "diff", "lo", "hi", "rel_pct", "p_holm", "verdict"]].rename(columns={
        "question": "Comparison", "a_mean": "A", "b_mean": "B", "diff": "A − B", "lo": "CI low", "hi": "CI high",
        "rel_pct": "Change %", "p_holm": "p (Holm)", "verdict": "Verdict"})
    st.dataframe(t.style.format({"A": "{:.3f}", "B": "{:.3f}", "A − B": "{:+.3f}", "CI low": "{:+.3f}",
                                 "CI high": "{:+.3f}", "Change %": "{:+.1f}", "p (Holm)": "{:.3f}"}),
                 hide_index=True, use_container_width=True)


def page_illusion():
    s = load_json("summary.json")
    st.title("🎭 The accuracy illusion")
    st.markdown("Paper 1 (Kavitha & Karpagam, 2025) reports **0.8932 accuracy**. Captions are padded with blank "
                "`<pad>` tokens to a fixed length of 34, and predicting a blank after the sentence has ended is easy. "
                "Counting those blanks inflates accuracy.")
    st.markdown("""<div style="font-family:ui-monospace,monospace;font-size:14px;line-height:2;margin:8px 0 16px">
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">a</span>
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">dog</span>
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">runs</span>
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">on</span>
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">grass</span>
    <span style="background:#cde2fb;color:#0b0b0b;padding:3px 6px;border-radius:4px">&lt;end&gt;</span>
    """ + " ".join(['<span style="background:#e1e0d9;color:#52514e;padding:3px 6px;border-radius:4px">&lt;pad&gt;</span>'] * 10)
                + """ … ×27</div>""", unsafe_allow_html=True)
    st.caption("A typical caption: 6 real tokens, 27 padding tokens. A model that only learns 'after <end> comes "
               "<pad>' already gets most positions right.")
    if not s:
        need_data()
        return
    m = s["models"]
    rows = [{"model": "Paper 1 (reported)", "measure": "Padding counted", "accuracy": s["paper1"]["accuracy"]["hybrid"]}]
    for cfg in ["paper_protocol", "hybrid_finetuned"]:
        if cfg in m:
            mm = m[cfg]["metrics"]
            rows += [{"model": m[cfg]["label"], "measure": "Padding counted", "accuracy": mm["unmasked_acc"]["mean"]},
                     {"model": m[cfg]["label"], "measure": "Padding ignored (honest)", "accuracy": mm["masked_acc"]["mean"]}]
    d = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"), title="Token accuracy"),
        y=alt.Y("measure:N", title=None),
        color=alt.Color("measure:N", scale=alt.Scale(domain=["Padding counted", "Padding ignored (honest)"],
                                                     range=["#eb6834", "#2a78d6"]), legend=alt.Legend(orient="bottom", title=None)),
        row=alt.Row("model:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"),
                    sort=["Paper 1 (reported)", m.get("paper_protocol", {}).get("label", ""), m.get("hybrid_finetuned", {}).get("label", "")]),
        tooltip=["model", "measure", alt.Tooltip("accuracy:Q", format=".3f")]).properties(height=alt.Step(22)),
        use_container_width=True)
    st.markdown("""
- **Paper-protocol replication:** same hybrid model, trained the paper's way (padding in the loss). Padded accuracy **~0.79**, honest accuracy **~0.41**, and its captions are no better (BLEU-1 0.588 vs 0.590).
- Our models are trained to ignore padding, so their padded accuracy is low (~0.15) while their captions are just as good.
- Padding explains **most** of Paper 1's 0.89, not all of it; the paper doesn't give the details needed to close the rest.
""")


def page_attention():
    a = load_json("attention.json")
    st.title("🔦 Attention explorer")
    st.caption("Precomputed for 40 test photos. A1 and A2 show their built-in attention; C3 (no attention) shows "
               "Grad-CAM. Bright = where the model looked when writing that word.")
    if not a or not a.get("models"):
        need_data()
        return
    imgs = a["images"]
    if "att_img" not in st.session_state:
        st.session_state.att_img = imgs[0]
    with st.expander("Choose a photo", expanded=False):
        cols = st.columns(8)
        for i, img in enumerate(imgs):
            with cols[i % 8]:
                p = thumb(img)
                if p:
                    st.image(p, use_container_width=True)
                if st.button("Pick", key=f"pick{img}", use_container_width=True):
                    st.session_state.att_img = img
    img = st.session_state.att_img
    model = st.radio("Model", list(a["models"]), horizontal=True,
                     format_func=lambda c: f'{a["models"][c]["label"]} ({a["models"][c]["method"]})')
    item = a["models"][model]["items"].get(img)
    if not item:
        st.info("No maps for this photo.")
        return
    st.markdown(f'### "{item["caption"]}"')
    words = item["words"]
    i = st.select_slider("Word", list(range(len(words))), format_func=lambda j: words[j], value=min(1, len(words) - 1))
    p = thumb(img, large=True)
    c1, c2 = st.columns(2, gap="large")
    from engine import heat_overlay
    with c1:
        st.image(Image.open(p), caption="Photo", use_container_width=True)
    with c2:
        st.image(heat_overlay(Image.open(p), np.array(item["maps"][i])), caption=f'"{words[i]}"',
                 use_container_width=True)
    others = [c for c in a["models"] if c != model and img in a["models"][c]["items"]]
    if others:
        st.markdown("**Other models on the same photo**")
        for c in others:
            st.write(f'- {a["models"][c]["label"]}: "{a["models"][c]["items"][img]["caption"]}"')


def page_gallery():
    g = load_json("gallery.json")
    s = load_json("summary.json") or {"models": {}}
    st.title("🖼️ Caption gallery")
    st.caption("All 1000 test photos with each model's caption (seed 1, beam 5; BLIP-large beam 3).")
    if not g:
        need_data()
        return
    items = g["items"]
    label = lambda c: s["models"].get(c, {}).get("label", c)  # noqa: E731
    c1, c2, c3 = st.columns([1, 1, 2])
    subjects = sorted({it["subject"] for it in items if it["subject"]})
    subj = c1.selectbox("Main subject", ["all"] + subjects)
    sort = c2.selectbox("Order", ["by name", "models disagree most", "hardest (lowest BLEU-4)", "random"])
    q = c3.text_input("Search captions", placeholder="e.g. bench, surfboard, two")
    sel = [it for it in items if (subj == "all" or it["subject"] == subj)]
    if q:
        ql = q.lower()
        sel = [it for it in sel if any(ql in v["c"].lower() for v in it["m"].values()) or any(ql in r.lower() for r in it["refs"])]
    if sort == "models disagree most":
        sel.sort(key=lambda it: -len({v["c"] for v in it["m"].values()}))
    elif sort == "hardest (lowest BLEU-4)":
        sel.sort(key=lambda it: np.mean([v["b4"] or 0 for v in it["m"].values()]))
    elif sort == "random":
        rng = np.random.default_rng(st.session_state.get("seed", 0))
        sel = list(rng.permutation(sel))
    per_page = 12
    pages = max(1, (len(sel) + per_page - 1) // per_page)
    page = st.number_input(f"Page (of {pages}) — {len(sel)} photos", 1, pages, 1)
    chunk = sel[(page - 1) * per_page: page * per_page]
    for row in range(0, len(chunk), 3):
        cols = st.columns(3, gap="medium")
        for col, it in zip(cols, chunk[row:row + 3]):
            with col:
                p = thumb(it["id"])
                if p:
                    st.image(p, use_container_width=True)
                best = it["m"].get("hybrid_clip") or next(iter(it["m"].values()))
                st.markdown(f'**{best["c"]}**')
                with st.expander("All models + human captions"):
                    for c, v in it["m"].items():
                        cs = f' · CLIPScore {v["cs"]:.2f}' if v.get("cs") is not None else ""
                        errs = f' · ⚠ {", ".join(v["e"])}' if v.get("e") else ""
                        st.markdown(f'- *{label(c)}*: {v["c"]}  \n  <small>BLEU-4 {v["b4"]:.2f}{cs}{errs}</small>',
                                    unsafe_allow_html=True)
                    st.markdown("**Human captions**")
                    for r in it["refs"]:
                        st.markdown(f"- {r}")


def page_errors():
    s = load_json("summary.json")
    st.title("⚠️ Errors & hallucination")
    st.markdown("**Hallucination** = the caption names an object none of the 5 human captions mention "
                "(CHAIR-s: share of captions with at least one). Error types are detected automatically and "
                "are approximate.")
    if not s:
        need_data()
        return
    rows = []
    for cfg, m in s["models"].items():
        for k, v in (m.get("errors") or {}).items():
            rows.append({"model": m["label"], "family": m["family"], "type": k, "rate": v["mean"],
                         "sd": v["sd"], "top": m.get("top_hallucinated", "")})
    if not rows:
        need_data()
        return
    d = pd.DataFrame(rows)
    ch = d[d.type == "chair_s"].copy()
    order = ch.sort_values("rate").model.tolist()
    color = alt.Color("family:N", scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
                      legend=alt.Legend(orient="bottom", title=None, columns=3))
    st.altair_chart(alt.Chart(ch).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("rate:Q", axis=alt.Axis(format=".0%"), title="Captions with a made-up object (lower is better)"),
        y=alt.Y("model:N", sort=order, title=None), color=color,
        tooltip=["model", alt.Tooltip("rate:Q", format=".1%"), alt.Tooltip("top:N", title="most made-up objects")]
    ).properties(height=alt.Step(30)), use_container_width=True)

    st.subheader("Error types")
    names = {"wrong_subject": "Wrong subject", "wrong_colour": "Wrong colour", "wrong_count": "Wrong count",
             "generic": "Generic caption", "repetition": "Repetition"}
    et = d[d.type.isin(names)].copy()
    et["type"] = et.type.map(names)
    st.altair_chart(alt.Chart(et).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("rate:Q", axis=alt.Axis(format=".0%"), title=None), y=alt.Y("model:N", sort=order, title=None),
        color=color, column=alt.Column("type:N", title=None),
        tooltip=["model", "type", alt.Tooltip("rate:Q", format=".1%")]).properties(width=150, height=alt.Step(16)),
        use_container_width=False)
    gal = os.path.join(RESULTS, "analysis", "failure_gallery.png")
    if os.path.exists(gal):
        with st.expander("Failure gallery (12 lowest-BLEU captions of the best classic model)"):
            st.image(gal, use_container_width=True)
            st.caption("Note the first photo: 'two dogs play in the grass' is correct, but scored low because "
                       "the human captions were worded differently.")


def page_tsne():
    t = load_json("tsne.json")
    st.title("🗺️ What ResNet50 sees: t-SNE map")
    st.caption("Each dot is one test photo; ResNet50's 2048 numbers per photo squeezed into 2-D. Photos the "
               "network sees as similar sit close together. Click a dot to see the photo.")
    if not t:
        need_data()
        return
    d = pd.DataFrame(t["points"], columns=["id", "x", "y", "subject"])
    person = {"man", "woman", "boy", "girl", "child", "people"}
    d["group"] = np.where(d.subject == "dog", "dog", np.where(d.subject.isin(person), "person",
                          np.where(d.subject.isin({"player", "biker", "skier", "surfer", "climber", "skateboarder"}),
                                   "sport", "other / unclear")))
    focus = st.selectbox("Highlight", ["groups (dog / person / sport)"] + sorted(d.subject.replace("", np.nan).dropna().unique()))
    if focus.startswith("groups"):
        color = alt.Color("group:N", scale=alt.Scale(domain=["dog", "person", "sport", "other / unclear"],
                                                     range=["#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"]),
                          legend=alt.Legend(orient="bottom", title=None))
    else:
        d["group"] = np.where(d.subject == focus, focus, "other")
        color = alt.Color("group:N", scale=alt.Scale(domain=[focus, "other"], range=["#2a78d6", "#e1e0d9"]),
                          legend=alt.Legend(orient="bottom", title=None))
    pick = alt.selection_point(fields=["id"], name="pick")
    chart = alt.Chart(d).mark_circle(size=40, opacity=0.85, stroke="white", strokeWidth=0.5).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), color=color,
        tooltip=["subject", "id"]).add_params(pick).properties(height=560)
    ev = st.altair_chart(chart, use_container_width=True, on_select="rerun", key="tsne")
    sel = (ev or {}).get("selection", {}).get("pick") if isinstance(ev, dict) else getattr(getattr(ev, "selection", None), "pick", None)
    if sel:
        img_id = sel[0].get("id")
        g = load_json("gallery.json")
        it = next((x for x in (g or {}).get("items", []) if x["id"] == img_id), None)
        c1, c2 = st.columns([1, 2])
        if thumb(img_id):
            c1.image(thumb(img_id), use_container_width=True)
        if it:
            c2.markdown("**Human captions**\n" + "\n".join(f"- {r}" for r in it["refs"]))
    st.markdown("**Reading the map:** dogs form their own island; sports photos group together; men, women, boys "
                "and girls are mixed, which is why every classic model often gets *which person* wrong "
                "(wrong subject in about a third of captions).")


def page_beam():
    st.title("🌳 Beam search")
    st.markdown("**Greedy** takes the single most likely next word every time. **Beam search (k)** keeps the k most "
                "likely partial sentences at each step and returns the best complete one. Red = the winning path.")
    d = os.path.join(RESULTS, "analysis", "beamtrees")
    pngs = sorted(f for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []
    if not pngs:
        need_data()
        return
    f = st.selectbox("Photo", pngs, format_func=lambda x: x.rsplit(".", 2)[0])
    c1, c2 = st.columns([1, 4])
    p = thumb(f.rsplit(".", 2)[0], large=True)
    if p:
        c1.image(p, use_container_width=True)
    c2.image(os.path.join(d, f), use_container_width=True)
    st.caption("Try your own photo on the Live captioning page: it draws the beam tree for any upload.")


def page_curves():
    c = load_json("curves.json")
    s = load_json("summary.json") or {"models": {}}
    st.title("📈 Training curves")
    if not c:
        need_data()
        return
    metric = st.segmented_control("Curve", ["val_loss", "val_masked_acc", "val_top5_acc"], default="val_loss",
                                  format_func=lambda k: {"val_loss": "Validation loss", "val_masked_acc": "Validation accuracy (honest)",
                                                         "val_top5_acc": "Validation top-5 accuracy"}[k]) or "val_loss"
    cfgs = st.multiselect("Models", list(c), default=[k for k in ["hybrid_frozen", "hybrid_finetuned", "attention", "transformer", "hybrid_clip"] if k in c],
                          format_func=lambda k: s["models"].get(k, {}).get("label", k))
    rows = []
    for cfg in cfgs:
        for seed, h in c[cfg].items():
            for e, v in enumerate(h.get(metric, []), 1):
                rows.append({"model": s["models"].get(cfg, {}).get("label", cfg), "seed": seed, "epoch": e, "value": v})
    if not rows:
        return
    d = pd.DataFrame(rows)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    models = list(dict.fromkeys(d.model))
    st.altair_chart(alt.Chart(d).mark_line(strokeWidth=2, opacity=0.85).encode(
        x=alt.X("epoch:Q", title="Epoch"), y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=alt.Scale(domain=models, range=palette[:len(models)]),
                        legend=alt.Legend(orient="bottom", title=None)),
        detail="seed:N", tooltip=["model", "seed", "epoch", alt.Tooltip("value:Q", format=".4f")]).properties(height=420),
        use_container_width=True)
    st.caption("One line per seed. Training stops 5 epochs after validation loss stops improving (early stopping); "
               "fine-tuned models start from their frozen model and train for up to 10 more epochs.")


# ----------------------------------------------------------------------------- nav
pages = [
    st.Page(page_live, title="Live captioning", icon="📷", default=True),
    st.Page(page_overview, title="Overview", icon="🏁", url_path="overview"),
    st.Page(page_results, title="Results", icon="📊", url_path="results"),
    st.Page(page_significance, title="Which differences are real?", icon="✅", url_path="significance"),
    st.Page(page_illusion, title="Accuracy illusion", icon="🎭", url_path="illusion"),
    st.Page(page_attention, title="Attention explorer", icon="🔦", url_path="attention"),
    st.Page(page_gallery, title="Caption gallery", icon="🖼️", url_path="gallery"),
    st.Page(page_errors, title="Errors & hallucination", icon="⚠️", url_path="errors"),
    st.Page(page_tsne, title="t-SNE map", icon="🗺️", url_path="tsne"),
    st.Page(page_beam, title="Beam search", icon="🌳", url_path="beam"),
    st.Page(page_curves, title="Training curves", icon="📈", url_path="curves"),
]
st.navigation(pages).run()
