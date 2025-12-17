import spacy
from typing import List
import subprocess
import sys
from ..utils.logging import get_logger

logger = get_logger(__name__)

def ensure_spacy_model(model_name: str = "da_core_news_sm") -> None:
    """Ensure the spaCy model is installed."""
    if not spacy.util.is_package(model_name):
        logger.info(f"Downloading spaCy model '{model_name}'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        logger.info(f"Finished downloading '{model_name}'.")

def load_spacy_stopwords() -> List[str]:
    """Load spaCy stopwords, ensuring model exists."""
    model_name = "da_core_news_sm"
    ensure_spacy_model(model_name)
    
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Fallback if load fails right after download (rare but possible)
        ensure_spacy_model(model_name)
        nlp = spacy.load(model_name)
        
    return list(nlp.Defaults.stop_words)

def get_custom_stopwords() -> List[str]:
    """Return project-specific stopwords."""
    return [
        # Folketinget procedures
        "formand", "formanden", "ordfører", "ordføreren",
        "minister", "ministeren", "tak", "statsminister", "statsministeren",
        "regering", "regeringen", "folketinget",
        "parti", "partiet", "medlem", "medlemmer",
        "lovforslag", "lovforslaget", "forslag", "forslaget",
        "lov", "loven", "lovene", "behandling", "førstebehandling", "andenbehandling", "tredjebehandling",
        "nr", "nummer", "dagsorden", "dagsordenen", "punkt",
        "spørgsmål", "svar",
        "hr", "fru", "værsgo",
        "bemærkning", "bemærkninger", "kort", "korte",
        "stemme", "stemmer", "afstemning", "afstemningen", "vedtaget", "forkastet",
        "indstilling", "indstillingen", "udvalg", "udvalget",
        "aftale", "aftalen",
        "mødet", "hævet", "åbnet", "sluttet", "forhandling", "forhandlingen",
        "næste", "kl", "klokken", "dag", "år",
        "lovforslag nr", "behandling af", # Phrases might be split by tokenizer, but adding tokens here helps
        
        # New procedural terms 
        "ordførerrækken", "rækken", "videre", "velkommen", "forslagsstillerne", "går",
        "stemte", "stemmes", "fremsættelse", "ændring", "ændringsforslag", "indsigelse",
        "betragter", "betænkning", "foreslår", "vedtagelse",
        "forespørgerne", "besvarelse", "besvarelsen", "minut", "minutter", "minutters", "taletid", "taletiden", "runde",
        "udtale", "ønsker", "tilfældet", "stillet", "stedfortræder",
        "spørgsmålet", "afsluttet", "medspørger", "udgået", "omtrykt",
        "genoptaget", "henvise", "fremgå", "ugeplanen", "morgen",
        "mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag",
        "foreløbig",
        
        # Party names (normalized and variants)
        "socialdemokratiet", "socialdemokrater", "socialdemokraterne",
        "venstre", "konservative", "radikale", "enhedslisten",
        "folkeparti", "alternativet", "borgerlige", "alliance", "moderaterne",
        "dansk", "danmarksdemokraterne", "liberal", "nye", "sf", "df", "el", "rv", "kf", "la", "nb", "dd", "m",
        
        # CommonFiller
        "her", "der", "så", "nu", "ja", "nej", "jo", "vel", "kunne", "skulle", "ville", 
        "må", "kan", "får", "få", "fik", "gør", "gjorde", "sige", "sagde", "ser", "så",
        "ved", "synes", "tror", "mener", "betyder", "forstår", "hører", "lytter",
        "tale", "taler", "ordet", "selvfølgelig", "naturligvis", "klart", "tydeligt",
        "rigtigt", "faktisk", "vist", "nok", "ske", "godt", "bedre", "bedst",
        "stor", "større", "størst", "lille", "mindre", "mindst", "meget", "lidt",
        "mange", "flere", "flest", "få", "færre", "færrest", "del", "hele", "alt",
        "ingenting", "noget", "nogen", "ingen", "enhver", "alle", "hver", "hinanden",
        "selv", "egen", "egne", "lige", "netop", "bare", "kun", "alene", "sammen",
        "tilbage", "frem", "ind", "ud", "op", "ned", "over", "under", "mellem",
        "gennem", "bag", "foran", "ved", "hos", "blandt", "omkring", "overfor", "imod",
        "uden", "inden", "efter", "før", "siden", "mens", "når", "da", "hvis",
        "fordi", "derfor", "hvordan", "hvorfor", "hvem", "hvad", "hvilken", "hvilket",
        "hvor", "hvornår", 
    ]

def get_combined_stopwords() -> List[str]:
    """Return combined unique list of spaCy and custom stopwords."""
    spacy_sw = load_spacy_stopwords()
    custom_sw = get_custom_stopwords()
    return sorted(list(set(spacy_sw + custom_sw)))
