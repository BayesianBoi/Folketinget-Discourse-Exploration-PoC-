import os
import re
import pandas as pd
import numpy as np
import openai
from sentence_transformers import util


class QueryRouter:
    def __init__(self, embedding_model):
        self.model = embedding_model
        # Define intents and their semantic descriptions
        self.routes = {
            "topic_lookup": "what is this topic about? keywords terms words definition meaning",
            "party_stance": "how does a party feel about a topic? sentiment opinion view attitude positive negative",
            "temporal_trend": "how has this topic changed over time? evolution history trend over the years",
            "compare_parties": "compare two parties on a topic difference agreement disagreement between",
            "temporal_comparison": "compare different years how did X talk about Y in year vs year 2008 2018 change over time",
            "speech_search": "find speeches documents what did they say quote text specific example search",
            "general_chat": "hello hi who are you help summary what can you do list topics parties"
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
        
        if best_score < 0.2:  # Low confidence fallthrough
            return "general_chat"
            
        return self.route_names[best_idx]


class DataRetriever:
    def __init__(self, df, topic_terms, embeddings, model):
        self.df = df
        self.topic_terms = topic_terms
        self.embeddings = embeddings
        self.model = model
        
        # Create lowercased map for fuzzy matching
        self.topics_map = {t.lower().split(":")[1].strip(): t for t in df["topic_label"].unique()}
        self.party_map = {p.lower(): p for p in df["party_name"].unique()}
        
        # Pre-compute topic label embeddings for semantic matching
        self.topic_labels = list(df["topic_label"].unique())
        self._topic_embeddings = None

    def _get_topic_embeddings(self):
        """Lazily compute topic label embeddings."""
        if self._topic_embeddings is None:
            # Create rich descriptions for each topic using labels + terms
            descriptions = []
            for label in self.topic_labels:
                tid = label.split(":")[0]
                terms = self.topic_terms.get(tid, [])
                term_str = " ".join([t['term'] if isinstance(t, dict) else str(t) for t in terms[:15]])
                # Combine label and terms for better matching
                desc = f"{label} {term_str}"
                descriptions.append(desc)
            self._topic_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        return self._topic_embeddings

    def _extract_years(self, query: str) -> list:
        """Extract year mentions from query (e.g., '2008 vs 2018' -> [2008, 2018])."""
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        return sorted(set(int(y) for y in years))

    def _find_topic(self, query):
        """Find topic by exact match first, then semantic fallback."""
        query_lower = query.lower()
        
        # 1. Direct label match (exact substring)
        for t_key, t_full in self.topics_map.items():
            if t_key in query_lower:
                return t_full
        
        # 2. Key term match
        best_match = None
        max_hits = 0
        
        for tid, terms in self.topic_terms.items():
            term_list = [t['term'] if isinstance(t, dict) else str(t) for t in terms]
            hits = sum(1 for term in term_list if term.lower() in query_lower)
            
            if hits > max_hits:
                max_hits = hits
                try:
                    label_row = self.df[self.df["top_topic_id"] == int(tid)]["topic_label"].iloc[0]
                    best_match = label_row
                except (IndexError, ValueError):
                    pass
                
        if max_hits > 0:
            return best_match
        
        # 3. Semantic matching as last resort
        return self._find_topic_semantic(query)

    def _find_topic_semantic(self, query: str, threshold: float = 0.3):
        """Use embeddings to find the most semantically similar topic."""
        query_vec = self.model.encode(query, convert_to_tensor=True)
        topic_embs = self._get_topic_embeddings()
        
        if query_vec.device != topic_embs.device:
            topic_embs = topic_embs.to(query_vec.device)
        
        scores = util.cos_sim(query_vec, topic_embs)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        
        if best_score >= threshold:
            return self.topic_labels[best_idx]
        return None

    def _find_parties(self, query):
        query_lower = query.lower()
        found = []
        for p_key, p_full in self.party_map.items():
            if p_key in query_lower:
                found.append(p_full)
        return found

    def get_available_data_summary(self) -> str:
        """Return a summary of available topics and parties."""
        # Get topic list (without numbers for readability)
        topics = [t.split(":")[1].strip() for t in self.topic_labels[:20]]
        parties = list(self.party_map.values())
        
        # Get year range
        years = self.df["year"].dropna().astype(int)
        year_range = f"{years.min()}-{years.max()}" if len(years) > 0 else "Unknown"
        
        return f"""Available Data:
- Topics (showing 20 of {len(self.topic_labels)}): {', '.join(topics)}...
- Parties: {', '.join(parties)}
- Year range: {year_range}
- Total speeches: {len(self.df):,}"""

    def get_topic_info(self, query):
        topic = self._find_topic(query)
        if not topic:
            # Provide helpful context when topic not found
            return f"Could not identify a specific topic. {self.get_available_data_summary()}"
        
        tid = topic.split(":")[0]
        terms = self.topic_terms.get(tid, [])
        term_str = ", ".join([t['term'] if isinstance(t, dict) else str(t) for t in terms[:15]])
        
        # Add some basic stats
        topic_df = self.df[self.df["topic_label"] == topic]
        speech_count = len(topic_df)
        avg_sentiment = topic_df["sentiment"].mean() if "sentiment" in topic_df.columns else None
        
        result = f"Topic '{topic}'\n"
        result += f"- Top terms: {term_str}\n"
        result += f"- Total speeches: {speech_count:,}\n"
        if avg_sentiment is not None:
            result += f"- Average sentiment: {avg_sentiment:.3f}\n"
        
        return result

    def get_party_stance(self, query):
        topic = self._find_topic(query)
        parties = self._find_parties(query)
        years = self._extract_years(query)
        
        if not topic:
            return f"Could not find a matching topic. {self.get_available_data_summary()}"
        
        sub = self.df[self.df["topic_label"] == topic]
        
        # Filter by years if specified
        if years:
            sub = sub[sub["year"].isin(years)]
            if sub.empty:
                return f"No speeches found for topic '{topic}' in years {years}."
        
        if not parties:
            # All parties
            agg = sub.groupby("party_name").agg(
                count=("doc_id", "count"),
                sentiment=("sentiment", "mean")
            ).sort_values("sentiment")
            return f"All parties on '{topic}'" + (f" (years: {years})" if years else "") + f":\n{agg.to_string()}"
        
        # Specific parties
        sub_p = sub[sub["party_name"].isin(parties)]
        if sub_p.empty:
            return f"No speeches found for {parties} on topic '{topic}'."
        
        agg = sub_p.groupby("party_name").agg(
            count=("doc_id", "count"),
            sentiment=("sentiment", "mean")
        )
        return f"Sentiment on '{topic}'" + (f" (years: {years})" if years else "") + f":\n{agg.to_string()}"

    def get_temporal_trend(self, query):
        topic = self._find_topic(query)
        parties = self._find_parties(query)
        
        if not topic:
            return f"Could not find a matching topic. {self.get_available_data_summary()}"
        
        sub = self.df[self.df["topic_label"] == topic]
        
        # Filter by party if specified
        if parties:
            sub = sub[sub["party_name"].isin(parties)]
        
        # Get counts and sentiment by year
        agg = sub.groupby("time_bin").agg(
            count=("doc_id", "count"),
            sentiment=("sentiment", "mean")
        ).sort_index()
        
        party_note = f" (filtered to: {', '.join(parties)})" if parties else ""
        return f"Temporal trend for '{topic}'{party_note}:\n{agg.to_string()}"

    def get_temporal_comparison(self, query):
        """Compare a topic across different years, optionally filtered by party."""
        topic = self._find_topic(query)
        parties = self._find_parties(query)
        years = self._extract_years(query)
        
        if not topic:
            return f"Could not find a matching topic. {self.get_available_data_summary()}"
        
        sub = self.df[self.df["topic_label"] == topic]
        
        # Filter by party if specified
        if parties:
            sub = sub[sub["party_name"].isin(parties)]
            party_label = ", ".join(parties)
        else:
            party_label = "All parties"
        
        if not years:
            # If no specific years, show overall trend
            years = sorted(sub["year"].dropna().unique())[-10:]  # Last 10 years
        
        # Filter to specified years
        sub_years = sub[sub["year"].isin(years)]
        
        if sub_years.empty:
            return f"No speeches found for '{topic}' by {party_label} in years {years}."
        
        # Aggregate by year
        agg = sub_years.groupby("year").agg(
            speech_count=("doc_id", "count"),
            avg_sentiment=("sentiment", "mean")
        ).sort_index()
        
        # Get sample speeches per year (for context)
        samples = []
        for year in years:
            year_df = sub_years[sub_years["year"] == year]
            if not year_df.empty:
                sample = year_df.sample(min(2, len(year_df)))
                for _, row in sample.iterrows():
                    text_preview = row["text"][:200] if isinstance(row["text"], str) else ""
                    samples.append(f"  [{int(year)}, {row['party_name']}]: \"{text_preview}...\"")
        
        result = f"Comparison of '{topic}' for {party_label}\n"
        result += f"Years analyzed: {years}\n\n"
        result += "Statistics by year:\n"
        result += agg.to_string() + "\n\n"
        result += "Sample speeches:\n" + "\n".join(samples[:6])
        
        return result

    def get_relevant_speeches(self, query, top_k=5):
        import torch
        
        topic = self._find_topic(query)
        parties = self._find_parties(query)
        years = self._extract_years(query)
        
        # Base filter
        sub = self.df.copy()
        
        if topic:
            sub = sub[sub["topic_label"] == topic]
        if parties:
            sub = sub[sub["party_name"].isin(parties)]
        if years:
            sub = sub[sub["year"].isin(years)]
        
        if sub.empty:
            return f"No speeches found matching filters. {self.get_available_data_summary()}"
        
        # If we have embeddings and df_docs, do semantic search within filtered set
        if hasattr(self, 'df_docs') and self.embeddings is not None:
            query_vec = self.model.encode(query, convert_to_tensor=True)
            if isinstance(self.embeddings, np.ndarray):
                emb_tensor = torch.from_numpy(self.embeddings)
            else:
                emb_tensor = self.embeddings
            
            if emb_tensor.device != query_vec.device:
                emb_tensor = emb_tensor.to(query_vec.device)
            
            # Get doc_ids in our filtered set
            valid_doc_ids = set(sub["doc_id"].values)
            
            # Score all embeddings
            scores = util.cos_sim(query_vec, emb_tensor)[0].cpu().numpy()
            
            # Find best matches within our filter
            found_speeches = []
            sorted_indices = np.argsort(-scores)
            
            for idx in sorted_indices:
                if len(found_speeches) >= top_k:
                    break
                try:
                    doc_row = self.df_docs.iloc[idx]
                    doc_id = doc_row["doc_id"]
                    
                    if doc_id not in valid_doc_ids:
                        continue
                    
                    meta_rows = self.df[self.df["doc_id"] == doc_id]
                    if meta_rows.empty:
                        continue
                    meta = meta_rows.iloc[0]
                    
                    text = meta['text'][:400] if isinstance(meta['text'], str) else ""
                    found_speeches.append(
                        f"[{meta['party_name']}, {meta['time_bin']}, score={scores[idx]:.2f}]\n\"{text}...\""
                    )
                except Exception:
                    continue
            
            if found_speeches:
                return f"Found {len(sub)} matching speeches. Top {len(found_speeches)} by relevance:\n\n" + "\n\n".join(found_speeches)
        
        # Fallback: just sample from filtered data
        sample = sub.sample(min(top_k, len(sub)))
        speeches = []
        for _, row in sample.iterrows():
            text = row['text'][:400] if isinstance(row['text'], str) else ""
            speeches.append(f"[{row['party_name']}, {row['time_bin']}]\n\"{text}...\"")
        
        return f"Found {len(sub)} matching speeches. Random sample:\n\n" + "\n\n".join(speeches)


class AIAgent:
    def __init__(self, df, df_docs, topic_terms, embedding_model, embeddings):
        self.retriever = DataRetriever(df, topic_terms, embeddings, embedding_model)
        self.retriever.df_docs = df_docs
        self.router = QueryRouter(embedding_model)
        
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
        
        # 2. Retrieve Data based on intent
        context = ""
        if intent == "topic_lookup":
            context = self.retriever.get_topic_info(user_query)
        elif intent == "party_stance" or intent == "compare_parties":
            context = self.retriever.get_party_stance(user_query)
        elif intent == "temporal_trend":
            context = self.retriever.get_temporal_trend(user_query)
        elif intent == "temporal_comparison":
            context = self.retriever.get_temporal_comparison(user_query)
        elif intent == "speech_search":
            context = self.retriever.get_relevant_speeches(user_query)
        elif intent == "general_chat":
            context = self.retriever.get_available_data_summary()
        
        # Always append available data summary if context seems limited
        if len(context) < 200 or "Could not" in context:
            context += "\n\n" + self.retriever.get_available_data_summary()
        
        # 3. Generate Answer with enhanced system prompt
        system_prompt = f"""You are an expert political analyst for the Danish Parliament (Folketinget).
You have access to a comprehensive dataset of parliamentary speeches with topics, parties, years, and sentiment scores.

The user asks: "{user_query}"

Here is the relevant data retrieved from the database:
---
{context}
---

Instructions:
1. Analyze the data and answer the user's question based ONLY on what's provided above.
2. If the data doesn't contain what the user asked for, explain what IS available and suggest related queries.
3. Format numbers nicely and explain what sentiment scores mean (-1 = negative, 0 = neutral, +1 = positive).
4. If comparing years/parties, highlight key differences in the data.
5. Be honest if data is missing or insufficient - suggest what might help.
6. Keep your response concise but informative.

IMPORTANT: Do NOT make up data. Only use what's in the context above."""

        response = self.client.chat.completions.create(
            model="gpt-5-search-mini",  # Web search enabled model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
