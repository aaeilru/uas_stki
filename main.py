import argparse
import os
import pickle
import json
from pathlib import Path

# Import komponen
from text_preprocessor import TextPreprocessor
from tfidf_vectorizer import TFIDFVectorizer
from inverted_index import InvertedIndex
from similarity import SimilarityCalculator
from search_engine import SearchEngine


def load_corpus_documents(corpus_dir='corpus'):
    """Load semua dokumen dari corpus"""
    documents = {}
    corpus_path = Path(corpus_dir)
    files = sorted(corpus_path.glob('OBT*.txt'))
    
    print(f"\n Loading documents from {corpus_dir}/")
    
    for filepath in files:
        doc_id = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        documents[doc_id] = content
    
    print(f" Loaded {len(documents)} documents")
    
    return documents


def run_preprocessing():
    """
    Run Phase 3: Preprocessing & Feature Extraction
    
    Steps:
    1. Load corpus
    2. Preprocess documents
    3. Build TF-IDF
    4. Build inverted index
    5. Save results
    """
    print("="*80)
    print("PHASE 3: PREPROCESSING & FEATURE EXTRACTION")
    print("="*80)
    
    # ===== 1. Load corpus =====
    documents = load_corpus_documents('corpus')
    
    if not documents:
        print(" No documents found in corpus/")
        return
    
    # ===== 2. Initialize preprocessor =====
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
    
    # ===== 3. Preprocess all documents =====
    print("\n Preprocessing documents...")
    processed_docs = {}
    
    for doc_id, content in documents.items():
        tokens = preprocessor.preprocess(content)
        processed_docs[doc_id] = tokens
    
    print(f" Preprocessed {len(processed_docs)} documents")
    
    # Preview
    sample_id = list(processed_docs.keys())[0]
    print(f"\n Sample from {sample_id}:")
    print(f"   Total tokens: {len(processed_docs[sample_id])}")
    print(f"   First 20 tokens: {processed_docs[sample_id][:20]}")
    
    # ===== 4. Build TF-IDF =====
    vectorizer = TFIDFVectorizer()
    doc_vectors = vectorizer.fit_transform(processed_docs)
    
    # ===== 5. Build Inverted Index =====
    inv_index = InvertedIndex()
    inverted_index = inv_index.build(doc_vectors, vectorizer.vocabulary)
    
    # ===== 6. Save preprocessed data =====
    print("\n Saving preprocessed data...")
    
    os.makedirs('data', exist_ok=True)
    
    # Save processed tokens
    with open('data/processed_docs.pkl', 'wb') as f:
        pickle.dump(processed_docs, f)
    print("    data/processed_docs.pkl")
    
    # Save vectorizer
    with open('data/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("    data/vectorizer.pkl")
    
    # Save inverted index
    with open('data/inverted_index.pkl', 'wb') as f:
        pickle.dump(dict(inverted_index), f)
    print("    data/inverted_index.pkl")
    
    # Save vocabulary
    with open('data/vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(vectorizer.vocabulary, f, ensure_ascii=False, indent=2)
    print("    data/vocabulary.json")
    
    # ===== 7. Statistics =====
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total documents: {len(processed_docs)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"Avg tokens/doc: {sum(len(t) for t in processed_docs.values()) / len(processed_docs):.1f}")
    print(f"Inverted index terms: {len(inverted_index)}")
    
    # Top terms by document frequency
    print("\n Top 10 terms by document frequency:")
    term_df = [(term, len(postings)) for term, postings in inverted_index.items()]
    term_df.sort(key=lambda x: x[1], reverse=True)
    
    for term, df in term_df[:10]:
        print(f"   {term}: {df} documents")
    
    print("\n Preprocessing completed!")
    print("\nNext: Run search with 'python main.py --search'")


def run_search():
    """
    Run Phase 4: Interactive Search
    """
    print("="*80)
    print("PHASE 4: INTERACTIVE SEARCH")
    print("="*80)
    
    # Check if preprocessed data exists
    if not os.path.exists('data/vectorizer.pkl'):
        print("\n Preprocessed data not found!")
        print(" Run preprocessing first: python main.py --preprocess")
        return
    
    # Initialize search engine
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f" Error loading search engine: {e}")
        return
    
    # Interactive mode
    print("\n" + "="*80)
    print("Perintah:")
    print("  - Ketik query untuk mencari")
    print("  - 'quit' untuk keluar")
    print("="*80)
    
    while True:
        query = input("\n Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(" Terima kasih!")
            break
        
        if not query:
            continue
        
        # Search
        results, tokens = engine.search(query, top_k=5)
        
        # Print results
        engine.print_results(results, tokens)


def run_demo():
    """
    Run demo dengan predefined queries
    """
    print("="*80)
    print("DEMO PENCARIAN OBAT")
    print("="*80)
    
    # Check if preprocessed data exists
    if not os.path.exists('data/vectorizer.pkl'):
        print("\n Preprocessed data not found!")
        print("Run preprocessing first: python main.py --preprocess")
        return
    
    # Initialize search engine
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Error loading search engine: {e}")
        return
    
    # Demo queries
    demo_queries = [
        ("demam sakit kepala", "Obat untuk demam dan sakit kepala"),
        ("batuk berdahak", "Obat batuk berdahak"),
        ("maag asam lambung", "Obat maag dan asam lambung"),
        ("nyeri haid", "Obat nyeri menstruasi"),
        ("diabetes gula darah", "Obat diabetes"),
    ]
    
    for query, description in demo_queries:
        print(f"\n{'='*80}")
        print(f" {description}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        results, tokens = engine.search(query, top_k=3)
        engine.print_results(results, tokens)
        
        input("\nTekan Enter untuk lanjut...")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Sistem IR Obat')
    
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing (Phase 3)')
    parser.add_argument('--search', action='store_true',
                       help='Run interactive search (Phase 4)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo queries')
    
    args = parser.parse_args()
    
    # Default: show help
    if not (args.preprocess or args.search or args.demo):
        parser.print_help()
        print("\n Workflow:")
        print("   1. python main.py --preprocess  (run once)")
        print("   2. python main.py --search      (interactive)")
        print("   3. python main.py --demo        (demo queries)")
        return
    
    # Run commands
    if args.preprocess:
        run_preprocessing()
    
    if args.search:
        run_search()
    
    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()