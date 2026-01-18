"""
Fungsi: Calculate similarity antara query dan dokumen
- Cosine Similarity
- Euclidean Distance (optional)
"""

import math


class SimilarityCalculator:
    """
    Calculator untuk mengukur similarity antara query dan dokumen
    
    Methods:
        cosine_similarity(): Cosine similarity (recommended for IR)
        euclidean_distance(): Euclidean distance
        jaccard_similarity(): Jaccard similarity (for sets)
    """
    
    @staticmethod
    def cosine_similarity(vec1, vec2, mag1, mag2):
        """
        Calculate Cosine Similarity antara dua vector
        
        Formula:
            cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
        
        Where:
            · = dot product
            || || = magnitude (L2 norm)
        
        Args:
            vec1: dict {term_id: weight} - vector pertama (query)
            vec2: dict {term_id: weight} - vector kedua (document)
            mag1: float - magnitude vector 1
            mag2: float - magnitude vector 2
        
        Returns:
            float - similarity score (0.0 to 1.0)
                   0.0 = completely different
                   1.0 = identical
        
        Example:
            >>> vec1 = {0: 0.5, 1: 0.3, 2: 0.4}
            >>> vec2 = {0: 0.6, 1: 0.2, 3: 0.5}
            >>> mag1 = math.sqrt(0.5**2 + 0.3**2 + 0.4**2)
            >>> mag2 = math.sqrt(0.6**2 + 0.2**2 + 0.5**2)
            >>> sim = cosine_similarity(vec1, vec2, mag1, mag2)
        """
        # Handle edge case: zero magnitude
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # ===== Calculate dot product =====
        # Dot product = sum of (v1[i] * v2[i]) for all common dimensions
        dot_product = 0.0
        
        # Get union of all term_ids
        all_term_ids = set(vec1.keys()) | set(vec2.keys())
        
        for term_id in all_term_ids:
            # Get weight (0 jika term tidak ada)
            weight1 = vec1.get(term_id, 0)
            weight2 = vec2.get(term_id, 0)
            
            dot_product += weight1 * weight2
        
        # ===== Calculate cosine =====
        cosine = dot_product / (mag1 * mag2)
        
        return cosine
    
    @staticmethod
    def euclidean_distance(vec1, vec2):
        """
        Calculate Euclidean Distance antara dua vector
        
        Formula:
            distance = sqrt(sum((v1[i] - v2[i])^2))
        
        Args:
            vec1: dict {term_id: weight}
            vec2: dict {term_id: weight}
        
        Returns:
            float - distance (semakin kecil semakin similar)
        
        Note:
            Euclidean distance NOT recommended untuk IR karena:
            - Sensitive to vector magnitude
            - Cosine similarity lebih baik untuk high-dimensional sparse vectors
        """
        # Get union of all term_ids
        all_term_ids = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate sum of squared differences
        sum_squared_diff = 0.0
        
        for term_id in all_term_ids:
            weight1 = vec1.get(term_id, 0)
            weight2 = vec2.get(term_id, 0)
            
            diff = weight1 - weight2
            sum_squared_diff += diff ** 2
        
        # Euclidean distance
        distance = math.sqrt(sum_squared_diff)
        
        return distance
    
    @staticmethod
    def jaccard_similarity(set1, set2):
        """
        Calculate Jaccard Similarity antara dua sets
        
        Formula:
            J(A, B) = |A ∩ B| / |A ∪ B|
        
        Args:
            set1: set of terms
            set2: set of terms
        
        Returns:
            float - similarity (0.0 to 1.0)
        
        Example:
            >>> set1 = {'demam', 'sakit', 'kepala'}
            >>> set2 = {'demam', 'panas', 'badan'}
            >>> sim = jaccard_similarity(set1, set2)
            >>> # sim = 1/5 = 0.2 (1 common, 5 total unique)
        
        Note:
            Jaccard similarity hanya melihat presence/absence,
            tidak memperhitungkan weight/frequency
        """
        # Intersection
        intersection = set1 & set2
        
        # Union
        union = set1 | set2
        
        # Handle empty union
        if len(union) == 0:
            return 0.0
        
        # Jaccard similarity
        similarity = len(intersection) / len(union)
        
        return similarity
    
    @staticmethod
    def calculate_magnitude(vector):
        """
        Calculate magnitude (L2 norm) dari vector
        
        Formula:
            magnitude = sqrt(sum(weight^2))
        
        Args:
            vector: dict {term_id: weight}
        
        Returns:
            float - magnitude
        """
        sum_squared = sum(weight ** 2 for weight in vector.values())
        return math.sqrt(sum_squared)


# ===== DEMO PENGGUNAAN =====
if __name__ == "__main__":
    print("="*80)
    print("DEMO SIMILARITY CALCULATOR")
    print("="*80)
    
    # Initialize calculator
    sim_calc = SimilarityCalculator()
    
    # ===== Test 1: Cosine Similarity =====
    print("\n" + "="*80)
    print("Test 1: Cosine Similarity")
    print("="*80)
    
    # Sample vectors (TF-IDF)
    query_vec = {0: 0.5, 1: 0.3, 2: 0.4}  # demam, sakit, kepala
    doc1_vec = {0: 0.6, 1: 0.2, 2: 0.5}   # demam, sakit, kepala (similar)
    doc2_vec = {3: 0.7, 4: 0.6}            # batuk, dahak (different)
    doc3_vec = {0: 0.4, 1: 0.3}            # demam, sakit (partial match)
    
    # Calculate magnitudes
    query_mag = sim_calc.calculate_magnitude(query_vec)
    doc1_mag = sim_calc.calculate_magnitude(doc1_vec)
    doc2_mag = sim_calc.calculate_magnitude(doc2_vec)
    doc3_mag = sim_calc.calculate_magnitude(doc3_vec)
    
    print("\nQuery vector:", query_vec)
    print(f"Query magnitude: {query_mag:.4f}")
    
    print("\n--- Document 1 (similar) ---")
    print("Vector:", doc1_vec)
    sim1 = sim_calc.cosine_similarity(query_vec, doc1_vec, query_mag, doc1_mag)
    print(f"Cosine Similarity: {sim1:.4f}")
    
    print("\n--- Document 2 (different) ---")
    print("Vector:", doc2_vec)
    sim2 = sim_calc.cosine_similarity(query_vec, doc2_vec, query_mag, doc2_mag)
    print(f"Cosine Similarity: {sim2:.4f}")
    
    print("\n--- Document 3 (partial match) ---")
    print("Vector:", doc3_vec)
    sim3 = sim_calc.cosine_similarity(query_vec, doc3_vec, query_mag, doc3_mag)
    print(f"Cosine Similarity: {sim3:.4f}")
    
    # ===== Test 2: Euclidean Distance =====
    print("\n" + "="*80)
    print("Test 2: Euclidean Distance")
    print("="*80)
    
    dist1 = sim_calc.euclidean_distance(query_vec, doc1_vec)
    dist2 = sim_calc.euclidean_distance(query_vec, doc2_vec)
    dist3 = sim_calc.euclidean_distance(query_vec, doc3_vec)
    
    print(f"\nDistance to Doc1: {dist1:.4f} (smaller = more similar)")
    print(f"Distance to Doc2: {dist2:.4f}")
    print(f"Distance to Doc3: {dist3:.4f}")
    
    # ===== Test 3: Jaccard Similarity =====
    print("\n" + "="*80)
    print("Test 3: Jaccard Similarity")
    print("="*80)
    
    query_terms = {'demam', 'sakit', 'kepala'}
    doc1_terms = {'demam', 'sakit', 'kepala', 'panas'}
    doc2_terms = {'batuk', 'dahak', 'berdahak'}
    doc3_terms = {'demam', 'panas'}
    
    print(f"\nQuery terms: {query_terms}")
    
    j_sim1 = sim_calc.jaccard_similarity(query_terms, doc1_terms)
    print(f"\nDoc1 terms: {doc1_terms}")
    print(f"Jaccard Similarity: {j_sim1:.4f}")
    
    j_sim2 = sim_calc.jaccard_similarity(query_terms, doc2_terms)
    print(f"\nDoc2 terms: {doc2_terms}")
    print(f"Jaccard Similarity: {j_sim2:.4f}")
    
    j_sim3 = sim_calc.jaccard_similarity(query_terms, doc3_terms)
    print(f"\nDoc3 terms: {doc3_terms}")
    print(f"Jaccard Similarity: {j_sim3:.4f}")
    
    # ===== Comparison =====
    print("\n" + "="*80)
    print("Comparison: Cosine vs Euclidean vs Jaccard")
    print("="*80)
    
    print("\n                  | Cosine  | Euclidean | Jaccard")
    print("-" * 60)
    print(f"Doc1 (similar)    | {sim1:.4f}  | {dist1:.4f}    | {j_sim1:.4f}")
    print(f"Doc2 (different)  | {sim2:.4f}  | {dist2:.4f}    | {j_sim2:.4f}")
    print(f"Doc3 (partial)    | {sim3:.4f}  | {dist3:.4f}    | {j_sim3:.4f}")
    
    print("\n Insight:")
    print("   - Cosine: Best for IR (0-1 scale, angle-based)")
    print("   - Euclidean: Distance-based (sensitive to magnitude)")
    print("   - Jaccard: Simple set overlap (ignores frequency)")
    
    print("\n" + "="*80)
    print("Demo selesai!")