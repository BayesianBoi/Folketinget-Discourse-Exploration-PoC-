"""
stopwords.py - implementing the stopwords
"""
import spacy

def load_spacy_stopwords():
    """Load spaCy stopwords"""
    try:
        nlp = spacy.load("da_core_news_sm") # using the small one for this assignment, imma step up to the big boi machine for the actual exam
        print(f"Loaded the spacy model: da_core_news_sm")
    except OSError:
        raise RuntimeError(
            "Danish model not found. Run python -m spacy download da_core_news_sm"
        )
    
    spacy_stopwords = list(nlp.Defaults.stop_words)
    print(f"Loaded {len(spacy_stopwords)} the stopwords from spacy")
    return spacy_stopwords

# In addition to the stopwords from spacy, the folketinget is filled with various filler words that we do NOT want to include for the topic modelling. these ones are just from a quick glanse at the data, will add more for the actual exam
def add_specific_stopwords():
    """Add folketinget specific stopwords."""
    specific_stopwords = [
        # Folketinget procedures
        "formand", "formanden", "ordfører", "ordføreren",
        "minister", "ministeren", "tak",
        "regering", "regeringen", "folketinget",
        "parti", "partiet", "medlem", "medlemmer",
        
        # We don't want to include the actual parties for our topics, so excluding those aswell
        # Party names
        "socialdemokratiet", "socialdemokrater", "socialdemokraterne",
        "venstre", "konservative", "radikale", "enhedslisten",
        "folkeparti", "alternativet", "borgerlige", "alliance", "moderaterne",

    ]
    print(f"Added {len(specific_stopwords)} stopwords")
    return specific_stopwords

def get_stopwords():
    """Combine the two lists of stopwords"""
    spacy_stopwords = load_spacy_stopwords()
    specific_stopwords = add_specific_stopwords()
    
    # Combine and then deduplicate
    all_stopwords = list(set(spacy_stopwords + specific_stopwords))
    
    print(f"Total stopwords: {len(all_stopwords)}")
    return all_stopwords