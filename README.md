# Perancangan dan Implementasi Sistem Pencarian Obat Berdasarkan Keluhan dan Nama Obat Berbasis TF-IDF dan Information Retrieval

Sistem pencarian obat berbasis Information Retrieval (IR) menggunakan TF-IDF dan Cosine Similarity. Sistem ini memungkinkan pencarian obat berdasarkan nama obat atau keluhan/gejala.
---

---

## Fitur  

- Pencarian obat berdasarkan keluhan atau gejala  
- Pencarian obat berdasarkan nama obat  
- Pencarian terpadu dalam satu kolom input (keluhan dan nama obat)  
- Filter kebutuhan resep (dengan resep / tanpa resep)  
- Filter berdasarkan rentang harga  
- Perankingan hasil pencarian menggunakan TF-IDF dan Cosine Similarity  
- Dukungan Bahasa Indonesia dengan stemming dan stopword removal  
- Antarmuka web menggunakan Streamlit  

---

## Teknologi yang Digunakan  

### Backend  
- Python 3.9+  
- TF-IDF Vectorization  
- Cosine Similarity  
- Sastrawi (stemming dan stopwords Bahasa Indonesia)  
- Inverted Index  

### Frontend  
- Streamlit  

### Library  
```txt
streamlit==1.29.0  
sastrawi==1.0.1  
pandas==2.1.3  
numpy==1.24.3``` 

```

##  Panduan Struktur Sistem 


#### **File 1: `text_preprocessor.py`** 
**Konsep:** Text preprocessing untuk Bahasa Indonesia

**Yang Dipelajari:**
- Case folding (lowercase)
- Tokenization (split by space)
- Stopwords removal
- Stemming (Sastrawi)

**Cara Struktur Sistem:**
```bash
# Jalankan demo
python text_preprocessor.py
```

**Output Demo:**
- Melihat transformasi text step-by-step
- Memahami perbedaan sebelum/sesudah preprocessing

**Latihan:**
```python
from text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
result = preprocessor.preprocess("Meredakan demam dan sakit kepala")
print(result)  # ['reda', 'demam', 'sakit', 'kepala']
```

---

#### **File 2: `tfidf_vectorizer.py`**
**Konsep:** TF-IDF vectorization

**Yang Dipelajari:**
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- TF-IDF calculation
- Vector representation

**Cara Menjalankan:**
```bash
# Jalankan demo
python tfidf_vectorizer.py
```

**Output Demo:**
- Melihat vocabulary building
- Memahami IDF values (rare vs common terms)
- Melihat TF-IDF vectors

**Formula:**
```
TF(t, d) = count(t in d) / length(d)
IDF(t) = log10(N / df(t))
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Latihan:**
```python
from tfidf_vectorizer import TFIDFVectorizer

documents = {
    'DOC1': ['demam', 'sakit', 'kepala'],
    'DOC2': ['batuk', 'dahak']
}

vectorizer = TFIDFVectorizer()
vectors = vectorizer.fit_transform(documents)
```

---

#### **File 3: `inverted_index.py`**
**Konsep:** Inverted index untuk efficient retrieval

**Yang Dipelajari:**
- Inverted index structure
- Posting lists
- Term-based retrieval

**Cara Struktur Sistem:**
```bash
# Jalankan demo
python inverted_index.py
```

**Output Demo:**
- Melihat struktur inverted index
- Memahami posting lists
- Search by term

**Structure:**
```
{
    'demam': [('DOC1', 0.5), ('DOC4', 0.3)],
    'batuk': [('DOC2', 0.6), ('DOC5', 0.4)]
}
```

**Latihan:**
```python
from inverted_index import InvertedIndex

inv_index = InvertedIndex()
index = inv_index.build(doc_vectors, vocabulary)

# Search term
results = inv_index.search_term('demam', top_k=5)
```

---

#### **File 4: `similarity.py`**
**Konsep:** Similarity calculation

**Yang Dipelajari:**
- Cosine similarity
- Euclidean distance
- Jaccard similarity

**Cara Struktur Sistem:**
```bash
# Jalankan demo
python similarity.py
```

**Output Demo:**
- Comparison 3 metode similarity
- Memahami kelebihan/kekurangan masing-masing

**Formula Cosine Similarity:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Latihan:**
```python
from similarity import SimilarityCalculator

calc = SimilarityCalculator()
sim = calc.cosine_similarity(query_vec, doc_vec, query_mag, doc_mag)
```

---

#### **File 5: `search_engine.py`**
**Konsep:** Main search engine

**Yang Dipelajari:**
- Menggabungkan semua komponen
- Query processing pipeline
- Ranking & filtering

**Cara Struktur Sistem:**
```bash
# Jalankan interactive mode
python search_engine.py
```

**Pipeline:**
1. Preprocess query
2. Transform to TF-IDF vector
3. Calculate similarity dengan semua dokumen
4. Rank by similarity
5. Apply filters
6. Return top-K results

**Latihan:**
```python
from search_engine import SearchEngine

engine = SearchEngine()
results, tokens = engine.search("demam sakit kepala", top_k=5)
engine.print_results(results, tokens)
```

---

#### **File 6: `main.py`**
**Konsep:** Full pipeline automation

**Cara Pakai:**

```bash
# Step 1: Preprocessing (run once)
python main.py --preprocess

# Step 2: Interactive search
python main.py --search

# Step 3: Demo queries
python main.py --demo
```

**Flow:**
1. Load corpus
2. Preprocess semua dokumen
3. Build TF-IDF
4. Build inverted index
5. Save hasil
6. Search engine ready

---

#### **File 7: `app.py`**
**Konsep:** User-friendly web interface

**Cara Pakai:**
```bash
streamlit run app.py
```

**Features:**
- Search by keluhan
- Search by nama obat
- Filter resep
- Filter harga
- Visual results

---

