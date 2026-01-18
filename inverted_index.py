"""
Fungsi: Build inverted index untuk efficient retrieval
- Structure: term -> [(doc_id, tfidf), ...]
- Sorted by TF-IDF descending
"""

from collections import defaultdict


class InvertedIndex:
    """
    Inverted Index untuk efficient document retrieval
    
    Structure:
        {
            'term1': [(doc_id1, tfidf1), (doc_id2, tfidf2), ...],
            'term2': [(doc_id3, tfidf3), ...],
            ...
        }
    
    Posting list untuk setiap term diurutkan by TF-IDF descending
    """
    
    def __init__(self):
        self.index = defaultdict(list)  # term -> [(doc_id, tfidf), ...]
    
    def build(self, doc_vectors, vocabulary):
        """
        Build inverted index dari TF-IDF vectors
        
        Args:
            doc_vectors: dict {doc_id: {term_id: tfidf}}
            vocabulary: dict {term: term_id}
        
        Returns:
            dict {term: [(doc_id, tfidf), ...]}
        
        Process:
        1. Reverse vocabulary (term_id -> term)
        2. Iterate semua dokumen
        3. Untuk setiap term dalam dokumen, tambahkan (doc_id, tfidf) ke posting list
        4. Sort posting lists by TF-IDF descending
        """
        print("\n Building Inverted Index...")
        
        # ===== STEP 1: Reverse vocabulary =====
        # Kita perlu mapping term_id -> term
        id_to_term = {idx: term for term, idx in vocabulary.items()}
        
        # ===== STEP 2: Build index =====
        for doc_id, vector in doc_vectors.items():
            # vector = {term_id: tfidf}
            for term_id, tfidf in vector.items():
                # Get term name
                term = id_to_term[term_id]
                
                # Add posting (doc_id, tfidf) ke posting list
                self.index[term].append((doc_id, tfidf))
        
        # ===== STEP 3: Sort posting lists =====
        # Sort by TF-IDF descending (dokumen paling relevan di depan)
        for term in self.index:
            self.index[term].sort(key=lambda x: x[1], reverse=True)
        
        print(f"Inverted Index built with {len(self.index)} terms")
        
        return self.index
    
    def get_posting_list(self, term):
        """
        Get posting list untuk term
        
        Args:
            term: string
        
        Returns:
            list of (doc_id, tfidf) atau empty list jika term tidak ada
        """
        return self.index.get(term, [])
    
    def get_docs_for_term(self, term):
        """
        Get document IDs yang mengandung term
        
        Args:
            term: string
        
        Returns:
            list of doc_ids
        """
        posting_list = self.get_posting_list(term)
        return [doc_id for doc_id, _ in posting_list]
    
    def search_term(self, term, top_k=10):
        """
        Search dokumen yang mengandung term
        
        Args:
            term: string
            top_k: jumlah dokumen yang dikembalikan
        
        Returns:
            list of (doc_id, tfidf) sorted by TF-IDF descending
        """
        posting_list = self.get_posting_list(term)
        return posting_list[:top_k]
    
    def search_multi_terms(self, terms, top_k=10):
        """
        Search dokumen yang mengandung salah satu term (OR query)
        
        Args:
            terms: list of terms
            top_k: jumlah dokumen yang dikembalikan
        
        Returns:
            list of (doc_id, combined_score)
        
        Note: 
            Combined score = sum of TF-IDF dari semua matching terms
        """
        # Collect all postings
        doc_scores = defaultdict(float)
        
        for term in terms:
            posting_list = self.get_posting_list(term)
            for doc_id, tfidf in posting_list:
                doc_scores[doc_id] += tfidf
        
        # Sort by combined score descending
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_term_stats(self, term):
        """
        Get statistics untuk term
        
        Args:
            term: string
        
        Returns:
            dict dengan stats term
        """
        posting_list = self.get_posting_list(term)
        
        if not posting_list:
            return None
        
        tfidf_values = [tfidf for _, tfidf in posting_list]
        
        return {
            'term': term,
            'document_frequency': len(posting_list),
            'max_tfidf': max(tfidf_values),
            'min_tfidf': min(tfidf_values),
            'avg_tfidf': sum(tfidf_values) / len(tfidf_values)
        }
    
    def get_index_stats(self):
        """
        Get overall statistics dari inverted index
        
        Returns:
            dict dengan statistik index
        """
        total_terms = len(self.index)
        total_postings = sum(len(postings) for postings in self.index.values())
        
        # Average posting list length
        avg_postings = total_postings / total_terms if total_terms > 0 else 0
        
        # Find term dengan posting list terpanjang
        longest_term = max(self.index.items(), key=lambda x: len(x[1]))
        
        # Find term dengan posting list terpendek
        shortest_term = min(self.index.items(), key=lambda x: len(x[1]))
        
        return {
            'total_terms': total_terms,
            'total_postings': total_postings,
            'avg_postings_per_term': avg_postings,
            'longest_posting_list': {
                'term': longest_term[0],
                'length': len(longest_term[1])
            },
            'shortest_posting_list': {
                'term': shortest_term[0],
                'length': len(shortest_term[1])
            }
        }


# ===== DEMO PENGGUNAAN =====
if __name__ == "__main__":
    print("="*80)
    print("DEMO INVERTED INDEX")
    print("="*80)
    
    # Sample vocabulary dan doc vectors
    # (Biasanya dari TFIDFVectorizer)
    vocabulary = {
        'demam': 0,
        'sakit': 1,
        'kepala': 2,
        'batuk': 3,
        'dahak': 4,
        'maag': 5,
        'asam': 6,
        'lambung': 7
    }
    
    doc_vectors = {
        'DOC001': {0: 0.5, 1: 0.3, 2: 0.4},  # demam, sakit, kepala
        'DOC002': {3: 0.6, 4: 0.5},           # batuk, dahak
        'DOC003': {5: 0.7, 6: 0.4, 7: 0.6},  # maag, asam, lambung
        'DOC004': {0: 0.8, 1: 0.2},           # demam, sakit
        'DOC005': {3: 0.4, 1: 0.3}            # batuk, sakit
    }
    
    # ===== Test 1: Build index =====
    print("\n" + "="*80)
    print("Test 1: Build Inverted Index")
    print("="*80)
    
    inv_index = InvertedIndex()
    index = inv_index.build(doc_vectors, vocabulary)
    
    # ===== Test 2: View index structure =====
    print("\n" + "="*80)
    print("Test 2: Index Structure")
    print("="*80)
    
    print("\nInverted Index:")
    for term, postings in sorted(index.items()):
        print(f"\n'{term}':")
        for doc_id, tfidf in postings:
            print(f"   {doc_id}: {tfidf:.4f}")
    
    # ===== Test 3: Search single term =====
    print("\n" + "="*80)
    print("Test 3: Search Single Term")
    print("="*80)
    
    term = 'demam'
    results = inv_index.search_term(term, top_k=5)
    print(f"\nSearch for '{term}':")
    for doc_id, tfidf in results:
        print(f"   {doc_id}: {tfidf:.4f}")
    
    # ===== Test 4: Search multiple terms =====
    print("\n" + "="*80)
    print("Test 4: Search Multiple Terms (OR query)")
    print("="*80)
    
    terms = ['demam', 'batuk']
    results = inv_index.search_multi_terms(terms, top_k=5)
    print(f"\nSearch for {terms}:")
    for doc_id, score in results:
        print(f"   {doc_id}: {score:.4f}")
    
    # ===== Test 5: Term statistics =====
    print("\n" + "="*80)
    print("Test 5: Term Statistics")
    print("="*80)
    
    for term in ['demam', 'sakit', 'batuk']:
        stats = inv_index.get_term_stats(term)
        if stats:
            print(f"\nTerm: '{stats['term']}'")
            print(f"   Document Frequency: {stats['document_frequency']}")
            print(f"   Max TF-IDF: {stats['max_tfidf']:.4f}")
            print(f"   Min TF-IDF: {stats['min_tfidf']:.4f}")
            print(f"   Avg TF-IDF: {stats['avg_tfidf']:.4f}")
    
    # ===== Test 6: Index statistics =====
    print("\n" + "="*80)
    print("Test 6: Overall Index Statistics")
    print("="*80)
    
    stats = inv_index.get_index_stats()
    print(f"\nTotal terms in index: {stats['total_terms']}")
    print(f"Total postings: {stats['total_postings']}")
    print(f"Average postings per term: {stats['avg_postings_per_term']:.2f}")
    print(f"\nLongest posting list:")
    print(f"   Term: '{stats['longest_posting_list']['term']}'")
    print(f"   Length: {stats['longest_posting_list']['length']}")
    print(f"\nShortest posting list:")
    print(f"   Term: '{stats['shortest_posting_list']['term']}'")
    print(f"   Length: {stats['shortest_posting_list']['length']}")
    
    print("\n" + "="*80)
    print("Demo selesai!")