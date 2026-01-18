import streamlit as st
import pickle
import json
import re


# Setup page
st.set_page_config(
    page_title="Sistem Pencarian Obat",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_search_engine():
    """Load IR system (cached)"""

    class SearchEngine:
        def __init__(self, data_dir='data', corpus_dir='corpus', metadata_file='metadata/obat_metadata.json'):
            # Load preprocessed data
            with open(f'{data_dir}/processed_docs.pkl', 'rb') as f:
                self.processed_docs = pickle.load(f)

            with open(f'{data_dir}/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)

            with open(f'{data_dir}/inverted_index.pkl', 'rb') as f:
                self.inverted_index = pickle.load(f)

            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                self.metadata = {item['id']: item for item in metadata_list}

            # Initialize preprocessor
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

                self.stemmer = StemmerFactory().create_stemmer()
                self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
                self.has_sastrawi = True
            except ImportError:
                self.stemmer = None
                self.stopword_remover = None
                self.has_sastrawi = False

        def preprocess_query(self, query: str):
            query = query.lower()
            query = re.sub(r'[^a-z\s]', ' ', query)
            query = re.sub(r'\s+', ' ', query).strip()

            tokens = query.split()

            if self.has_sastrawi and self.stopword_remover:
                text = ' '.join(tokens)
                text = self.stopword_remover.remove(text)
                tokens = text.split()

            if self.has_sastrawi and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]

            tokens = [t for t in tokens if len(t) > 2]
            return tokens

        def cosine_similarity(self, vec1, vec2, mag1, mag2):
            if mag1 == 0 or mag2 == 0:
                return 0.0

            all_terms = set(vec1.keys()) | set(vec2.keys())
            dot_product = sum(vec1.get(term_id, 0) * vec2.get(term_id, 0) for term_id in all_terms)
            return dot_product / (mag1 * mag2)

        def search(self, query, top_k=10, filter_resep=None, min_price=None, max_price=None):
            query_tokens = self.preprocess_query(query)

            # IMPORTANT: selalu return (results, tokens)
            if not query_tokens:
                return [], []

            query_vector, query_magnitude = self.vectorizer.transform_query(query_tokens)

            scores = []
            for doc_id in self.processed_docs.keys():
                doc_vector = self.vectorizer.doc_vectors.get(doc_id, {})
                doc_magnitude = self.vectorizer.doc_lengths.get(doc_id, 0)

                similarity = self.cosine_similarity(query_vector, doc_vector, query_magnitude, doc_magnitude)

                if similarity > 0:
                    scores.append((doc_id, similarity))

            scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc_id, score in scores:
                metadata = self.metadata.get(doc_id, {})

                # Filters resep
                if filter_resep is not None and metadata.get('perlu_resep') != filter_resep:
                    continue

                harga_min = metadata.get('harga_min', 0)
                harga_max = metadata.get('harga_max', 0)

                # Filters harga (pakai is not None supaya 0 tetap kebaca)
                if min_price is not None and harga_max < min_price:
                    continue
                if max_price is not None and harga_min > max_price:
                    continue

                results.append((doc_id, score, metadata))

                if len(results) >= top_k:
                    break

            return results, query_tokens

    return SearchEngine()


def render_results(results):
    """Render hasil pencarian"""
    for rank, (doc_id, score, metadata) in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### #{rank}. {metadata.get('nama_obat', 'N/A')}")
                # Tampilkan generik kalau ada, kalau tidak ada tampilkan field lain yang relevan
                st.markdown(f"**Nama Generik:** {metadata.get('nama_generik', 'N/A')}")

            with col2:
                st.markdown(f'<span class="score-badge">Score: {score:.4f}</span>', unsafe_allow_html=True)

            st.markdown(f"**Golongan:** {metadata.get('golongan', 'N/A')}")
            st.markdown(f"**Indikasi:** {metadata.get('indikasi', 'N/A')}")

            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Harga Min", f"Rp {metadata.get('harga_min', 0):,}")
            with col4:
                st.metric("Harga Max", f"Rp {metadata.get('harga_max', 0):,}")
            with col5:
                resep_badge = "✅ Ya" if metadata.get('perlu_resep') == 'Ya' else "❌ Tidak"
                st.metric("Perlu Resep", resep_badge)

            with st.expander("📋 Detail Lengkap"):
                st.markdown(f"**Komposisi:** {metadata.get('komposisi', 'N/A')}")
                st.markdown(f"**Dosis:** {metadata.get('dosis', 'N/A')}")
                st.markdown(f"**Efek Samping:** {metadata.get('efek_samping', 'N/A')}")
                st.markdown(f"**Kontraindikasi:** {metadata.get('kontraindikasi', 'N/A')}")
                st.markdown(f"**Produsen:** {metadata.get('produsen', 'N/A')}")
                st.markdown(f"**Tags:** {metadata.get('tags', 'N/A')}")

            st.markdown("---")


def main():
    # Title
    st.markdown('<h1 class="main-title">Sistem Pencarian Obat</h1>', unsafe_allow_html=True)
    st.markdown("### Sistem Temu Kembali Informasi Obat Indonesia")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>DISCLAIMER:</strong> Sistem ini hanya untuk informasi edukatif. 
        Selalu konsultasikan dengan dokter atau apoteker sebelum menggunakan obat.
        Tidak menggantikan konsultasi medis profesional.
    </div>
    """, unsafe_allow_html=True)

    # Load search engine
    try:
        search_engine = load_search_engine()
        st.success("Search engine loaded successfully!")
    except Exception as e:
        st.error(f"Error loading search engine: {e}")
        st.stop()

    # Sidebar - Filters
    st.sidebar.header("Filter Pencarian")

    top_k = st.sidebar.slider("Jumlah hasil", min_value=5, max_value=20, value=10)

    filter_resep_options = ["Semua", "Tanpa Resep", "Dengan Resep"]
    filter_resep_choice = st.sidebar.selectbox("Kebutuhan Resep", filter_resep_options)

    filter_resep = None
    if filter_resep_choice == "Tanpa Resep":
        filter_resep = "Tidak"
    elif filter_resep_choice == "Dengan Resep":
        filter_resep = "Ya"

    # Price filter
    st.sidebar.subheader("Range Harga")
    use_price_filter = st.sidebar.checkbox("Aktifkan filter harga")

    min_price = None
    max_price = None
    if use_price_filter:
        price_range = st.sidebar.slider(
            "Harga (Rp)",
            min_value=0,
            max_value=200000,
            value=(0, 100000),
            step=5000,
            format="Rp %d"
        )
        min_price, max_price = price_range

    # Main search (SATU KOLOM + ENTER SUBMIT)
    st.header("Pencarian Obat")
    st.subheader("Cari Obat (Keluhan/Gejala atau Nama Obat)")

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Masukkan keluhan/gejala ATAU nama obat:",
            placeholder="Contoh: demam dan sakit kepala / Panadol / Promag",
            key="unified_query"
        )
        submitted = st.form_submit_button("Cari Obat")

    if submitted:
        if query.strip():
            with st.spinner("Mencari obat yang sesuai..."):
                results, tokens = search_engine.search(
                    query,
                    top_k=top_k,
                    filter_resep=filter_resep,
                    min_price=min_price,
                    max_price=max_price
                )

            if tokens:
                st.info(f"Query tokens: {', '.join(tokens)}")
            else:
                st.info("Query tokens: (kosong setelah preprocessing)")

            if results:
                st.success(f"Ditemukan {len(results)} obat")
                render_results(results)
            else:
                st.warning("Tidak ada obat yang ditemukan. Coba gunakan kata kunci lain.")
        else:
            st.warning("Mohon masukkan query pencarian (keluhan/gejala atau nama obat).")

    # Statistics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Statistik Sistem")
    st.sidebar.metric("Total Dokumen", len(search_engine.processed_docs))
    st.sidebar.metric("Vocabulary Size", len(search_engine.vectorizer.vocabulary))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Sistem Temu Kembali Informasi Obat | Tugas Akhir Mata Kuliah</p>
        <p>Dibuat dengan Python, TF-IDF, dan Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
