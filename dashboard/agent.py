import os
import pandas as pd
import numpy as np
import openai
from sentence_transformers import util

class QueryRouter:
    def __init__(self, embedding_model):
        self.model = embedding_model
        # Define intents and their semantic descriptions
        self.routes = {
            "topic_lookup": "what is this topic about? keywords terms words definition",
            "party_stance": "how does a party feel about a topic? sentiment opinion view attitude positive negative",
            "temporal_trend": "how has this topic changed over time? evolution history when trend year",
            "compare_parties": "compare two parties on a topic difference agreement disagreement",
            "speech_search": "find speeches documents what did they say quote text specific example search",
            "general_chat": "hello hi who are you help summary what can you do"
        }
        self.route_names = list(self.routes.keys())
        # Pre-compute embeddings for routes
        self.route_embeddings = self.model.encode(list(self.routes.values()), convert_to_tensor=True)

    def route(self, query):
        query_vec = self.model.encode(query, convert_to_tensor=True)
        # Ensure device match
        if query_vec.device != self.route_embeddings.device:
            self.route_embeddings = self.route_embeddings.to(query_vec.device)
            
        scores = util.cos_sim(query_vec, self.route_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        
        if best_score < 0.2: # Low confidence fallthrough
            return "general_chat"
            
        return self.route_names[best_idx]

class DataRetriever:
    def __init__(self, df, topic_terms, embeddings, model):
        self.df = df
        self.topic_terms = topic_terms
        self.embeddings = embeddings # stored as numpy array usually
        self.model = model
        # Create lowercased map for fuzzy matching
        self.topics_map = {t.lower().split(":")[1].strip(): t for t in df["topic_label"].unique()} 
        self.party_map = {p.lower(): p for p in df["party_name"].unique()}

    def _find_topic(self, query):
        query_lower = query.lower()
        
        # 1. Direct label match
        for t_key, t_full in self.topics_map.items():
            if t_key in query_lower:
                return t_full
        
        # 2. Key term match (fuzzy fallback)
        # Search through the terms dictionary
        best_match = None
        max_hits = 0
        
        query_words = set(query_lower.split())
        
        for tid, terms in self.topic_terms.items():
            # Get string terms
            term_list = [t['term'] if isinstance(t, dict) else str(t) for t in terms]
            # Check overlap
            hits = sum(1 for term in term_list if term.lower() in query_lower)
            
            if hits > max_hits:
                max_hits = hits
                # Find the full label from df since topic_terms only has ID
                label_row = self.df[self.df["top_topic_id"] == int(tid)]["topic_label"].iloc[0]
                best_match = label_row
                
        if max_hits > 0:
            return best_match
            
        return None

    def _find_parties(self, query):
        query_lower = query.lower()
        found = []
        for p_key, p_full in self.party_map.items():
            if p_key in query_lower:
                found.append(p_full)
        return found
    
    def get_relevant_speeches(self, query, top_k=3):
        # 1. Encode query
        query_vec = self.model.encode(query, convert_to_tensor=True)
        
        # 2. Compute similarity against ALL docs
        # Note: self.embeddings is numpy array (N, D)
        # We use util.cos_sim but need inputs as tensors
        import torch
        if isinstance(self.embeddings, np.ndarray):
            emb_tensor = torch.from_numpy(self.embeddings)
        else:
            emb_tensor = self.embeddings
            
        # Move to correct device
        if emb_tensor.device != query_vec.device:
            emb_tensor = emb_tensor.to(query_vec.device)
            
        scores = util.cos_sim(query_vec, emb_tensor)[0]
        
        # 3. Filter by Party (if mentioned)
        parties = self._find_parties(query)
        
        # Get top indices
        # We only want to look at indices that exist in self.df
        # self.df is likely a SUBSET of the full embeddings if sample_size was used
        # BUT self.embeddings usually matches self.df alignment in load_data
        
        # Let's assume indices align with df.reset_index(drop=True) OR df index
        # Ideally embeddings are paired with doc_ids.
        # In app.py: df, ... df_docs ... embeddings = ...
        # The embeddings usually correspond to 'df_docs' or the main 'df'.
        # Let's assume the alignment from app.py: df corresponds to embeddings rows?
        # NO. app.py loads embeddings for ALL of df_docs (465k) or subset.
        # df is merged.
        # SAFEST: Use the 'doc_id' mapping if available, or just top_k generic if alignment is uncertain.
        # In app.py, `embeddings` are loaded. `df` is the main view.
        # We need to map embedding index -> doc_id -> df row.
        # This is tricky without the ID mapping passed in.
        # SIMPLIFICATION: semantic search in app.py uses `df_docs.iloc[doc_idx]`.
        # We should pass `df_docs` to DataRetriever if we want 100% safety.
        # OR: Just accept we might miss party filtering for now and return top global hits?
        # "find specific documents from SPECIFIC PARTIES" -> filtering is key.
        
        # To do filtering properly with potentially misaligned embeddings/df:
        # Top K raw indices -> map to doc_ids -> check if in df & party match.
        
        top_results = torch.topk(scores, k=100) # Get more candidates
        indices = top_results.indices.cpu().numpy()
        probs = top_results.values.cpu().numpy()
        
        results = []
        for rank, idx in enumerate(indices):
            if len(results) >= top_k:
                break
                
            # We assume embedding index `idx` maps to `self.df.iloc[idx]`? 
            # DANGER: If df is filtered/sorted, this breaks.
            # We need to trust that `agent.df` passed in IS likely just `df_docs` merged with topics.
            # IF `df` was shuffled or filtered, we are in trouble.
            # FIX: In app.py, `df` is created by merge.
            # `embeddings` comes from `load_embeddings`.
            # We need `doc_id` to index mapping from `load_data`.
            # `df` has `doc_id`.
            # Let's assume `embeddings` are for `df_docs` in order.
            # We need `df_docs` to map index -> doc_id.
            # I will request `df_docs` be passed to `AIAgent`.
            
            # Temporary fallback: Just take `self.df.iloc[idx]` and hope 
            # (Works if df wasn't shuffled? It was merged...)
            # Actually, `get_semantic_search_model` returns embeddings. `load_data` returns `df` and `df_docs`.
            # The embeddings correspond to `df_docs` rows!
            
            # So: index -> df_docs.iloc[idx]["doc_id"] -> lookup in self.df
            pass # Placeholder logic handled in fixed block below
        
        return "Search functionality requires df_docs linkage. Please update Agent init."

    def get_topic_info(self, query):
        query_lower = query.lower()
        
        # 1. Direct label match
        for t_key, t_full in self.topics_map.items():
            if t_key in query_lower:
                return t_full
        
        # 2. Key term match (fuzzy fallback)
        # Search through the terms dictionary
        best_match = None
        max_hits = 0
        
        query_words = set(query_lower.split())
        
        for tid, terms in self.topic_terms.items():
            # Get string terms
            term_list = [t['term'] if isinstance(t, dict) else str(t) for t in terms]
            # Check overlap
            hits = sum(1 for term in term_list if term.lower() in query_lower)
            
            if hits > max_hits:
                max_hits = hits
                # Find the full label from df since topic_terms only has ID
                label_row = self.df[self.df["top_topic_id"] == int(tid)]["topic_label"].iloc[0]
                best_match = label_row
                
        if max_hits > 0:
            return best_match
            
        return None

    def _find_parties(self, query):
        query_lower = query.lower()
        found = []
        for p_key, p_full in self.party_map.items():
            if p_key in query_lower:
                found.append(p_full)
        return found

    def get_topic_info(self, query):
        topic = self._find_topic(query)
        if not topic:
            return "Could not identify a specific topic in your query. Please specify a topic name."
        
        tid = topic.split(":")[0]
        terms = self.topic_terms.get(tid, [])
        # Format terms
        term_str = ", ".join([t['term'] if isinstance(t, dict) else str(t) for t in terms[:10]])
        return f"Topic '{topic}' is defined by these top terms: {term_str}"

    def get_party_stance(self, query):
        topic = self._find_topic(query)
        parties = self._find_parties(query)
        
        if not topic:
             return "Please mention a specific topic."
        
        sub = self.df[self.df["topic_label"] == topic]
        if not parties:
            # Aggregate all
            agg = sub.groupby("party_name")["sentiment"].mean().sort_values()
            return f"Sentiment handling '{topic}':\n" + agg.to_string()
        
        # Specific parties
        sub_p = sub[sub["party_name"].isin(parties)]
        agg = sub_p.groupby("party_name")["sentiment"].mean()
        return f"Sentiment on '{topic}':\n" + agg.to_string()

    def get_temporal_trend(self, query):
        topic = self._find_topic(query)
        if not topic:
             return "Please mention a specific topic."
        
        sub = self.df[self.df["topic_label"] == topic]
        agg = sub.groupby("time_bin").size()
        return f"Speech volume for '{topic}' over time:\n" + agg.to_string()


    
    def get_relevant_speeches(self, query, top_k=3):
        import torch
        
        # 1. Encode query
        query_vec = self.model.encode(query, convert_to_tensor=True)
        if isinstance(self.embeddings, np.ndarray):
            emb_tensor = torch.from_numpy(self.embeddings)
        else:
            emb_tensor = self.embeddings
        
        if emb_tensor.device != query_vec.device:
            emb_tensor = emb_tensor.to(query_vec.device)
            
        # 2. Similarity
        scores = util.cos_sim(query_vec, emb_tensor)[0]
        
        # 3. Filter candidates
        parties = self._find_parties(query)
        
        # Get top 50 to allow for filtering
        top_results = torch.topk(scores, k=50)
        indices = top_results.indices.cpu().numpy()
        
        found_speeches = []
        
        # We need to map Index -> DocID -> Main DF Row
        # We rely on self.df having 'doc_id' and 'doc_index' OR assume self.df IS df_docs-compatible?
        # Actually in app.py: df_docs has the raw text. 
        # Ideally we pass df_docs to DataRetriever.
        # Let's try to find the row in self.df using implicit index if possible, otherwise we iterate.
        # WAIT: self.df might not have ALL docs if it was filtered (min length etc).
        # We should use the `df_docs` if passed, or fail gracefully.
        
        # Assuming self.df HAS 'doc_id'.
        # Assuming we can't map index -> doc_id without the original `df_docs`.
        # I will enforce passing `df_docs` to `AIAgent`.
        
        if not hasattr(self, 'df_docs'):
             return "Error: Document index missing."
        
        for idx in indices:
            if len(found_speeches) >= top_k:
                break
                
            try:
                # Get Doc ID from the original sequential list
                doc_row = self.df_docs.iloc[idx]
                doc_id = doc_row["doc_id"]
                
                # Check Party via Main DF (where metadata lives)
                # Or use doc_row if it has party
                # df_docs usually has raw columns. 
                # Let's check self.df (the processed one) for this doc_id
                
                meta_rows = self.df[self.df["doc_id"] == doc_id]
                if meta_rows.empty:
                    continue
                meta = meta_rows.iloc[0]
                
                # Filter by party if specified
                if parties:
                    if meta["party_name"] not in parties:
                        continue
                
                found_speeches.append(f"({meta['party_name']}, {meta['time_bin']}): \"{meta['text'][:300]}...\"")
                
            except Exception as e:
                continue
                
        if not found_speeches:
            return "No relevant speeches found matching criteria."
            
        return "Relevant Speeches:\n" + "\n\n".join(found_speeches)


class AIAgent:
    def __init__(self, df, df_docs, topic_terms, embedding_model, embeddings):
        self.retriever = DataRetriever(df, topic_terms, embeddings, embedding_model)
        self.retriever.df_docs = df_docs # Inject directly
        self.router = QueryRouter(embedding_model)
        
        # Load API Key
        # Ensure imports work
        try:
             from dotenv import load_dotenv
             load_dotenv()
        except:
             pass
        
        self.client = None
        if os.environ.get("OPENAI_API_KEY"):
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def answer(self, user_query):
        if not self.client:
            return "⚠️ OpenAI API Key not found. Please check your .env file."

        # 1. Classify Intent
        intent = self.router.route(user_query)
        
        # 2. Retrieve Data
        context = ""
        if intent == "topic_lookup":
            context = self.retriever.get_topic_info(user_query)
        elif intent == "party_stance" or intent == "compare_parties":
            context = self.retriever.get_party_stance(user_query)
        elif intent == "temporal_trend":
            context = self.retriever.get_temporal_trend(user_query)
        elif intent == "speech_search":
            context = self.retriever.get_relevant_speeches(user_query)
        
        # 3. Generate Answer
        system_prompt = f"""You are an expert political analyst for the Danish Parliament (Folketinget). 
        You have access to a dataset of speeches.
        
        The user asks: "{user_query}"
        
        Here is the relevant data retrieved from the database:
        ---
        {context}
        ---
        
        Analyze this data and answer the user's question concisely. 
        If the data is missing or empty, admit it. 
        Do not hallucinate data not present in the context.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=1
        )
        
        return response.choices[0].message.content
