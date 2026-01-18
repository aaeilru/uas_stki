"""
Fungsi: Main search engine yang menggabungkan semua komponen
- Load preprocessed data
- Process query
- Calculate similarity
- Rank results
- Apply filters
"""

import pickle
import json
from pathlib import Path

# Import komponen-komponen yang sudah dibuat
# Pastikan file-file ini ada di folder yang sama:
# - text_preprocessor.py
# - tfidf_vectorizer.py
# - similarity.py

try:
    from text_preprocessor import TextPreprocessor
    from tfidf_vectorizer import TFIDFVectorizer
    from similarity import SimilarityCalculator
    COMPONENTS_AVAILABLE = True
except ImportError:
    print("Pastikan file text_preprocessor.py, tfidf_vectorizer.py, dan similarity.py ada")
    COMPONENTS_AVAILABLE = False


class SearchEngine:
    """
    Main Search Engine untuk IR Obat
    
    Components:
    - TextPreprocessor: Preprocessing query
    - TFIDFVectorizer: Transform query to vector
    - SimilarityCalculator: Calculate similarity
    - Metadata: Informasi detail obat
    
    Features:
    - Search by query (keluhan/nama)
    - Filter by resep (Ya/Tidak)
    - Filter by price range
    - Ranking by cosine similarity
    """
    
    def __init__(self, data_dir='data', metadata_file='metadata/obat_metadata.json'):
        """
        Initialize Search Engine
        
        Args:
            data_dir: folder berisi preprocessed data
            metadata_file: file JSON metadata obat
        """
        print("Loading Search Engine...")
        
        # ===== Load preprocessed data =====
        self._load_data(data_dir)
        
        # ===== Load metadata =====
        self._load_metadata(metadata_file)
        
        # ===== Initialize components =====
        if COMPONENTS_AVAILABLE:
            self.preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
            self.similarity_calc = SimilarityCalculator()
        else:
            self.preprocessor = None
            self.similarity_calc = None
        
        print(f"Search Engine ready!")
        print(f"   Documents: {len(self.vectorizer.doc_vectors)}")
        print(f"   Vocabulary: {len(self.vectorizer.vocabulary)}")
    
    def _load_data(self, data_dir):
        """Load preprocessed data dari pickle files"""
        data_path = Path(data_dir)
        
        # Load vectorizer (contains all TF-IDF data)
        with open(data_path / 'vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"   Loaded vectorizer")
    
    def _load_metadata(self, metadata_file):
        """Load metadata obat dari JSON"""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        # Convert to dict for faster access
        self.metadata = {item['id']: item for item in metadata_list}
        
        print(f"  Loaded metadata ({len(self.metadata)} obat)")
    
    def search(self, query, top_k=10, filter_resep=None, min_price=None, max_price=None):
        """
        Search obat berdasarkan query
        
        Args:
            query: string - keluhan atau nama obat
            top_k: int - jumlah hasil yang dikembalikan
            filter_resep: str - None/'Ya'/'Tidak'
            min_price: int - harga minimal
            max_price: int - harga maksimal
        
        Returns:
            tuple (results, query_tokens)
            - results: list of (doc_id, score, metadata)
            - query_tokens: list of processed tokens
        
        Example:
            >>> engine = SearchEngine()
            >>> results, tokens = engine.search("demam sakit kepala", top_k=5)
            >>> for doc_id, score, meta in results:
            >>>     print(f"{meta['nama_obat']}: {score:.4f}")
        """
        if not self.preprocessor:
            return [], []
        
        # ===== STEP 1: Preprocess query =====
        query_tokens = self.preprocessor.preprocess(query)
        
        if not query_tokens:
            print(" Query tidak menghasilkan token (mungkin hanya stopwords)")
            return [], []
        
        # ===== STEP 2: Transform query to TF-IDF vector =====
        query_vector, query_magnitude = self.vectorizer.transform_query(query_tokens)
        
        if not query_vector:
            print("⚠️  Query tokens tidak ada di vocabulary")
            return [], query_tokens
        
        # ===== STEP 3: Calculate similarity dengan semua dokumen =====
        scores = []
        
        for doc_id in self.vectorizer.doc_vectors.keys():
            # Get document vector dan magnitude
            doc_vector = self.vectorizer.doc_vectors[doc_id]
            doc_magnitude = self.vectorizer.doc_lengths[doc_id]
            
            # Calculate cosine similarity
            similarity = self.similarity_calc.cosine_similarity(
                query_vector, doc_vector,
                query_magnitude, doc_magnitude
            )
            
            # Only keep documents with similarity > 0
            if similarity > 0:
                scores.append((doc_id, similarity))
        
        # ===== STEP 4: Sort by similarity descending =====
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # ===== STEP 5: Apply filters =====
        results = []
        
        for doc_id, score in scores:
            metadata = self.metadata.get(doc_id, {})
            
            # Filter: Resep
            if filter_resep and metadata.get('perlu_resep') != filter_resep:
                continue
            
            # Filter: Harga
            harga_min_obat = metadata.get('harga_min', 0)
            harga_max_obat = metadata.get('harga_max', 0)
            
            if min_price and harga_max_obat < min_price:
                continue
            
            if max_price and harga_min_obat > max_price:
                continue
            
            # Add to results
            results.append((doc_id, score, metadata))
            
            # Stop when we have top_k results
            if len(results) >= top_k:
                break
        
        return results, query_tokens
    
    def get_document_details(self, doc_id):
        """
        Get detail lengkap dokumen
        
        Args:
            doc_id: string
        
        Returns:
            dict metadata dokumen
        """
        return self.metadata.get(doc_id, {})
    
    def get_top_terms_in_doc(self, doc_id, top_k=10):
        """
        Get top-k terms dengan TF-IDF tertinggi dalam dokumen
        
        Args:
            doc_id: string
            top_k: int
        
        Returns:
            list of (term, tfidf_value)
        """
        if doc_id not in self.vectorizer.doc_vectors:
            return []
        
        # Reverse vocabulary
        id_to_term = {idx: term for term, idx in self.vectorizer.vocabulary.items()}
        
        # Get vector
        vector = self.vectorizer.doc_vectors[doc_id]
        
        # Sort by TF-IDF descending
        sorted_terms = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to terms
        result = [(id_to_term[term_id], tfidf) 
                  for term_id, tfidf in sorted_terms[:top_k]]
        
        return result
    
    def print_results(self, results, query_tokens=None):
        """
        Print search results dengan format rapi
        
        Args:
            results: list of (doc_id, score, metadata)
            query_tokens: list of tokens (optional)
        """
        if query_tokens:
            print(f"\n Query tokens: {', '.join(query_tokens)}")
        
        if not results:
            print("\n Tidak ada hasil ditemukan")
            return
        
        print(f"\n✅ Ditemukan {len(results)} hasil:\n")
        print("="*100)
        
        for rank, (doc_id, score, metadata) in enumerate(results, 1):
            print(f"\n#{rank} | Score: {score:.4f} | ID: {doc_id}")
            print(f"Nama: {metadata.get('nama_obat', 'N/A')}")
            print(f"Generik: {metadata.get('nama_generik', 'N/A')}")
            print(f"Golongan: {metadata.get('golongan', 'N/A')}")
            print(f"Indikasi: {metadata.get('indikasi', 'N/A')[:150]}...")
            print(f"Harga: Rp {metadata.get('harga_min', 0):,} - Rp {metadata.get('harga_max', 0):,}")
            print(f"Resep: {metadata.get('perlu_resep', 'N/A')}")
            print(f"Tags: {metadata.get('tags', 'N/A')}")
            print("-"*100)


# ===== DEMO / INTERACTIVE MODE =====
def interactive_mode():
    """Interactive search mode"""
    print("="*80)
    print("SISTEM PENCARIAN OBAT - INTERACTIVE MODE")
    print("="*80)
    
    # Initialize search engine
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Error loading search engine: {e}")
        print("\n Pastikan Anda sudah menjalankan preprocessing:")
        print("   python preprocessing.py")
        return
    
    print("\n" + "="*80)
    print("Perintah:")
    print("  - Ketik query untuk mencari (contoh: 'demam sakit kepala')")
    print("  - 'filter:resep' untuk filter obat dengan resep")
    print("  - 'filter:no-resep' untuk filter obat tanpa resep")
    print("  - 'filter:harga:50000' untuk filter harga max")
    print("  - 'quit' atau 'exit' untuk keluar")
    print("="*80)
    
    while True:
        query = input("\n Query: ").strip()
        
        # Check exit
        if query.lower() in ['quit', 'exit', 'q']:
            print("Terima kasih!")
            break
        
        if not query:
            continue
        
        # Parse filters
        filter_resep = None
        max_price = None
        
        if 'filter:resep' in query:
            filter_resep = 'Ya'
            query = query.replace('filter:resep', '').strip()
        
        if 'filter:no-resep' in query:
            filter_resep = 'Tidak'
            query = query.replace('filter:no-resep', '').strip()
        
        if 'filter:harga:' in query:
            try:
                parts = [p for p in query.split() if 'filter:harga:' in p]
                if parts:
                    max_price = int(parts[0].split(':')[-1])
                    query = query.replace(parts[0], '').strip()
            except:
                print("Format filter harga: filter:harga:50000")
        
        # Search
        results, tokens = engine.search(
            query,
            top_k=5,
            filter_resep=filter_resep,
            max_price=max_price
        )
        
        # Print results
        engine.print_results(results, tokens)


if __name__ == "__main__":
    interactive_mode()