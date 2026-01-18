"""
Fungsi: Class untuk preprocessing text Bahasa Indonesia
- Case folding (lowercase)
- Tokenization
- Stopwords removal
- Stemming
"""

import re

# Sastrawi untuk Bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("Sastrawi tidak terinstall. Install dengan: pip install Sastrawi")


class TextPreprocessor:
    """
    Text Preprocessor untuk Bahasa Indonesia
    
    Methods:
        clean_text(text) -> str: Membersihkan text
        tokenize(text) -> list: Memecah text jadi tokens
        remove_stopwords(tokens) -> list: Hapus stopwords
        stem(tokens) -> list: Stemming tokens
        preprocess(text) -> list: Full pipeline
    """
    
    def __init__(self, use_stemming=True, use_stopwords=True):
        """
        Args:
            use_stemming: Boolean, aktifkan stemming atau tidak
            use_stopwords: Boolean, aktifkan stopwords removal atau tidak
        """
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        
        if SASTRAWI_AVAILABLE:
            # Setup Sastrawi Stemmer
            if use_stemming:
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
                print("astrawi Stemmer loaded")
            else:
                self.stemmer = None
            
            # Setup Sastrawi Stopwords
            if use_stopwords:
                factory = StopWordRemoverFactory()
                self.stopword_remover = factory.create_stop_word_remover()
                print("Sastrawi Stopwords loaded")
            else:
                self.stopword_remover = None
            
            # Custom stopwords khusus domain medis
            self.custom_stopwords = {
                'mg', 'ml', 'gram', 'tablet', 'kapsul', 'sirup',
                'yang', 'pada', 'untuk', 'dengan', 'dari', 'atau', 
                'dan', 'seperti', 'termasuk', 'dapat', 'akan', 'per'
            }
        else:
            self.stemmer = None
            self.stopword_remover = None
            self.custom_stopwords = set()
            print("⚠️  Sastrawi tidak tersedia, preprocessing terbatas")
    
    def clean_text(self, text):
        """
        Membersihkan text
        
        Steps:
        1. Lowercase
        2. Remove standalone numbers
        3. Remove special characters (keep only letters and spaces)
        4. Remove extra spaces
        
        Args:
            text: string input
        
        Returns:
            string yang sudah dibersihkan
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Hapus angka standalone (tapi keep angka yang menempel kata)
        # Contoh: "500mg" tetap, tapi "500" hilang
        text = re.sub(r'\b\d+\b', '', text)
        
        # 3. Hapus semua karakter kecuali huruf dan spasi
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 4. Hapus extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenization sederhana (split by space)
        
        Args:
            text: string input
        
        Returns:
            list of tokens
        """
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Hapus stopwords dari tokens
        
        Args:
            tokens: list of tokens
        
        Returns:
            list of tokens tanpa stopwords
        """
        if not self.use_stopwords:
            return tokens
        
        if not SASTRAWI_AVAILABLE:
            # Basic stopwords jika Sastrawi tidak ada
            basic_stopwords = {
                'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 
                'dengan', 'untuk', 'pada', 'atau', 'oleh'
            }
            return [t for t in tokens 
                    if t not in basic_stopwords 
                    and t not in self.custom_stopwords 
                    and len(t) > 2]
        
        # Gabung tokens jadi text
        text = ' '.join(tokens)
        
        # Hapus stopwords dengan Sastrawi
        text = self.stopword_remover.remove(text)
        
        # Split lagi dan filter custom stopwords + token pendek
        tokens = text.split()
        tokens = [t for t in tokens 
                  if t not in self.custom_stopwords 
                  and len(t) > 2]
        
        return tokens
    
    def stem(self, tokens):
        """
        Stemming untuk Bahasa Indonesia
        
        Contoh:
            meredakan -> reda
            menurunkan -> turun
            mengobati -> obat
        
        Args:
            tokens: list of tokens
        
        Returns:
            list of stemmed tokens
        """
        if not self.use_stemming or not SASTRAWI_AVAILABLE or not self.stemmer:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        
        Pipeline:
        1. Clean text
        2. Tokenize
        3. Remove stopwords (optional)
        4. Stemming (optional)
        5. Filter token pendek (< 3 chars)
        
        Args:
            text: string input
        
        Returns:
            list of processed tokens
        """
        # Step 1: Clean
        text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(text)
        
        # Step 3: Remove stopwords
        if self.use_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 4: Stemming
        if self.use_stemming:
            tokens = self.stem(tokens)
        
        # Step 5: Filter token yang terlalu pendek
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens


# ===== DEMO PENGGUNAAN =====
if __name__ == "__main__":
    print("="*80)
    print("DEMO TEXT PREPROCESSOR")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
    
    # Test cases
    test_texts = [
        "Meredakan demam dan nyeri ringan hingga sedang seperti sakit kepala",
        "Mengobati infeksi bakteri seperti infeksi saluran pernapasan",
        "Menurunkan kadar gula darah pada diabetes mellitus tipe 2",
        "Paracetamol 500mg tablet untuk demam dan sakit kepala"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}:")
        print(f"Original: {text}")
        
        # Step by step
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned:  {cleaned}")
        
        tokens = preprocessor.tokenize(cleaned)
        print(f"Tokens:   {tokens}")
        
        no_stopwords = preprocessor.remove_stopwords(tokens)
        print(f"No Stop:  {no_stopwords}")
        
        stemmed = preprocessor.stem(no_stopwords)
        print(f"Stemmed:  {stemmed}")
        
        # Full pipeline
        result = preprocessor.preprocess(text)
        print(f"RESULT:   {result}")
    
    print("\n" + "="*80)
    print("Demo selesai!")