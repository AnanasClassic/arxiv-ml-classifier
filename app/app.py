import streamlit as st
import datetime
from model_utils import CATEGORY_LABELS, classify, find_similar, load_model, load_recent_index

st.set_page_config(page_title="arXiv ML Classifier", layout="wide")
st.title("arXiv ML Paper Classifier")
st.caption("Classify ML papers by arXiv category and find similar papers from the last 7 days.")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource(ttl=3600)
def get_index():
    try:
        return load_recent_index(), None
    except Exception as e:
        return None, str(e)


def render_similar_card(row):
    with st.container(border=True):
        st.markdown(f"**[{row['title']}]({row['url']})**")
        if row.get("abstract_preview"):
            st.caption(row["abstract_preview"])
        col1, col2 = st.columns([3, 1])
        with col1:
            tags = row.get("categories", "")
            st.caption(tags if isinstance(tags, str) else " | ".join(tags))
        with col2:
            row["published"] = datetime.datetime.strptime(row["published"], "%Y-%m-%d").strftime("%d %b %Y")
            st.caption(f"{row['published']}, sim {row['score']:.0%}")


title = st.text_input("Title", placeholder="Attention Is All You Need")
abstract = st.text_area("Abstract (optional)", placeholder="We propose a new simple network architecture...", height=160)

if not abstract.strip():
    st.info("Classification without abstract may be less accurate.")

if st.button("Classify", type="primary"):
    if not title.strip():
        st.warning("Please enter a paper title.")
        st.stop()

    try:
        tokenizer, model = get_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    with st.spinner("Classifying..."):
        results = classify(title, abstract, tokenizer, model)

    top_cat, top_prob = results[0]

    if top_cat == "other":
        st.subheader("Classification")
        st.warning(f"This paper doesn't appear to be ML-related ({top_prob:.0%} confidence).")
    else:
        classification_col, similar_col = st.columns([6, 5], gap="large")

        with classification_col:
            st.subheader("Classification")
            for cat, prob in results:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{CATEGORY_LABELS.get(cat, cat)}**")
                    st.progress(prob)
                with col2:
                    st.metric(label="", value=f"{prob:.0%}")

        with similar_col:
            st.subheader("Similar papers (last 7 days)")
            (embeddings, meta), err = get_index()
            if embeddings is None:
                st.info(f"Index is updating, try again later. ({err})")
            else:
                with st.spinner("Searching..."):
                    similar = find_similar(title, abstract, tokenizer, model, embeddings, meta)
                for _, row in similar.iterrows():
                    render_similar_card(row)
