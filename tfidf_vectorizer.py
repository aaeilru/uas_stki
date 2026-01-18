"""
Fungsi: Build TF-IDF vectors dari dokumen
- Calculate Term Frequency (TF)
- Calculate Inverse Document Frequency (IDF)
- Build TF-IDF vectors
- Transform query
"""

import math
from collections import defaultdict, Counter


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer untuk Information Retrieval
    
    Formula:
        TF(t, d) = count(t in d) / length(d)
        IDF(t) = log10(N / df(t))
        TF-IDF(t, d) = TF(t, d) * IDF(t)
    
    Attributes:
        vocabulary: dict {term: term_id}
        idf: dict {term: idf_value}
        doc_vectors: dict {doc_id: {term_id: tfidf}}
        doc_lengths: dict {doc_id: magnitude}
    """
    
    def __init__(self):
        self.vocabulary = {}  # term -> term_id
        self.idf = {}  # term -> idf value
        self.doc_vectors = {}  # doc_id -> {term_id: tfidf}
        self.doc_lengths = {}  # doc_id -> magnitude (for cosine similarity)
    
    def fit_transform(self, documents):
        """
        Build TF-IDF vectors dari koleksi dokumen
        
        Args:
            documents: dict {doc_id: [tokens]}
                      Contoh: {'OBT001': ['demam', 'sakit', 'kepala'], ...}
        
        Returns:
            dict {doc_id: {term_id: tfidf_value}}
        
        Process:
        1. Build vocabulary
        2. Calculate document frequency (df)
        3. Calculate IDF
        4. Calculate TF-IDF untuk setiap dokumen
        5. Calculate document magnitude (untuk cosine similarity)
        """
        print("\n Building TF-IDF vectors...")
        
        # ===== STEP 1: Build vocabulary & document frequency =====
        term_doc_count = defaultdict(int)  # term -> jumlah dokumen yang mengandung term
        all_terms = set()
        
        for doc_id, tokens in documents.items():
            # Unique terms dalam dokumen ini
            unique_terms = set(tokens)
            
            # Count document frequency
            for term in unique_terms:
                term_doc_count[term] += 1
            
            # Kumpulkan semua terms
            all_terms.update(tokens)
        
        # ===== STEP 2: Create vocabulary mapping =====
        # Urutkan alphabetically untuk konsistensi
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # ===== STEP 3: Calculate IDF =====
        N = len(documents)  # Total jumlah dokumen
        
        for term, df in term_doc_count.items():
            # IDF = log10(N / df)
            # Semakin jarang term muncul, semakin tinggi IDF
            self.idf[term] = math.log10(N / df)
        
        print(f"Total documents: {N}")
        
        # ===== STEP 4: Calculate TF-IDF untuk setiap dokumen =====
        for doc_id, tokens in documents.items():
            # Calculate Term Frequency
            tf = Counter(tokens)  # {term: count}
            doc_length = len(tokens)
            
            # TF-IDF vector untuk dokumen ini
            tfidf_vector = {}
            magnitude = 0  # Untuk cosine similarity
            
            for term, count in tf.items():
                if term in self.vocabulary and term in self.idf:
                    # TF: normalized frequency
                    tf_value = count / doc_length
                    
                    # TF-IDF
                    tfidf = tf_value * self.idf[term]
                    
                    # Simpan dengan term_id sebagai key
                    term_id = self.vocabulary[term]
                    tfidf_vector[term_id] = tfidf
                    
                    # Accumulate untuk magnitude (L2 norm)
                    magnitude += tfidf ** 2
            
            # Simpan vector dan magnitude
            self.doc_vectors[doc_id] = tfidf_vector
            self.doc_lengths[doc_id] = math.sqrt(magnitude)
        
        print(f"TF-IDF vectors created for {len(self.doc_vectors)} documents")
        
        return self.doc_vectors
    
    def transform_query(self, query_tokens):
        """
        Transform query menjadi TF-IDF vector
        
        Args:
            query_tokens: list of tokens
                         Contoh: ['demam', 'sakit', 'kepala']
        
        Returns:
            tuple (tfidf_vector, magnitude)
            - tfidf_vector: dict {term_id: tfidf_value}
            - magnitude: float (L2 norm untuk cosine similarity)
        """
        # Calculate TF
        tf = Counter(query_tokens)
        query_length = len(query_tokens)
        
        tfidf_vector = {}
        magnitude = 0
        
        for term, count in tf.items():
            # Hanya process term yang ada di vocabulary
            if term in self.vocabulary and term in self.idf:
                # TF
                tf_value = count / query_length
                
                # TF-IDF
                tfidf = tf_value * self.idf[term]
                
                # Simpan
                term_id = self.vocabulary[term]
                tfidf_vector[term_id] = tfidf
                
                # Accumulate magnitude
                magnitude += tfidf ** 2
        
        return tfidf_vector, math.sqrt(magnitude)
    
    def get_term_info(self, term):
        """
        Get informasi tentang term
        
        Args:
            term: string
        
        Returns:
            dict dengan info term
        """
        if term not in self.vocabulary:
            return None
        
        term_id = self.vocabulary[term]
        idf_value = self.idf.get(term, 0)
        
        # Count berapa dokumen yang mengandung term ini
        doc_count = 0
        for doc_vector in self.doc_vectors.values():
            if term_id in doc_vector:
                doc_count += 1
        
        return {
            'term': term,
            'term_id': term_id,
            'idf': idf_value,
            'document_frequency': doc_count
        }
    
    def get_top_terms(self, doc_id, top_k=10):
        """
        Get top-k terms dengan TF-IDF tertinggi dalam dokumen
        
        Args:
            doc_id: document id
            top_k: jumlah terms
        
        Returns:
            list of (term, tfidf_value)
        """
        if doc_id not in self.doc_vectors:
            return []
        
        # Reverse vocabulary
        id_to_term = {idx: term for term, idx in self.vocabulary.items()}
        
        # Get vector
        vector = self.doc_vectors[doc_id]
        
        # Sort by tfidf descending
        sorted_terms = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        
        # Convert term_id to term
        result = [(id_to_term[term_id], tfidf) 
                  for term_id, tfidf in sorted_terms[:top_k]]
        
        return result


# ===== DEMO PENGGUNAAN =====
if __name__ == "__main__":
    print("="*80)
    print("DEMO TF-IDF VECTORIZER")
    print("="*80)
    
    # Sample documents (sudah dalam bentuk tokens)
    documents = {
        'DOC001': ['demam', 'sakit', 'kepala', 'reda', 'nyeri'],
        'DOC002': ['batuk', 'dahak', 'reda', 'tenggorok'],
        'DOC003': ['maag', 'asam', 'lambung', 'nyeri', 'perut'],
        'DOC004': ['demam', 'panas', 'reda', 'badan'],
        'DOC005': ['batuk', 'kering', 'reda', 'tenggorok', 'gatal']
    }
    
    # Initialize vectorizer
    vectorizer = TFIDFVectorizer()
    
    # Build TF-IDF
    doc_vectors = vectorizer.fit_transform(documents)
    
    # ===== Test 1: Analyze vocabulary =====
    print(f"\n{'='*80}")
    print("Test 1: Vocabulary Analysis")
    print(f"{'='*80}")
    print(f"Total unique terms: {len(vectorizer.vocabulary)}")
    print(f"Terms: {list(vectorizer.vocabulary.keys())}")
    
    # ===== Test 2: IDF values =====
    print(f"\n{'='*80}")
    print("Test 2: IDF Values")
    print(f"{'='*80}")
    for term, idf in sorted(vectorizer.idf.items(), key=lambda x: x[1], reverse=True):
        print(f"   {term}: {idf:.4f}")
    
    # ===== Test 3: Document vectors =====
    print(f"\n{'='*80}")
    print("Test 3: Document Vectors")
    print(f"{'='*80}")
    
    for doc_id in ['DOC001', 'DOC003']:
        print(f"\n{doc_id}:")
        top_terms = vectorizer.get_top_terms(doc_id, top_k=5)
        for term, tfidf in top_terms:
            print(f"   {term}: {tfidf:.4f}")
    
    # ===== Test 4: Query transformation =====
    print(f"\n{'='*80}")
    print("Test 4: Query Transformation")
    print(f"{'='*80}")
    
    query_tokens = ['demam', 'sakit', 'kepala']
    query_vec, query_mag = vectorizer.transform_query(query_tokens)
    
    print(f"Query: {query_tokens}")
    print(f"Query vector: {query_vec}")
    print(f"Query magnitude: {query_mag:.4f}")
    
    # ===== Test 5: Term info =====
    print(f"\n{'='*80}")
    print("Test 5: Term Information")
    print(f"{'='*80}")
    
    for term in ['demam', 'reda', 'batuk']:
        info = vectorizer.get_term_info(term)
        if info:
            print(f"\nTerm: {info['term']}")
            print(f"   IDF: {info['idf']:.4f}")
            print(f"   Document Frequency: {info['document_frequency']}")
    
    print("\n" + "="*80)
    print("Demo selesai!")