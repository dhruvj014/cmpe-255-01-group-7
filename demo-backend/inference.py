import math
import re


class FakeReviewPipeline:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        if not self.use_mock:
            # Uncomment to load real models
            # import joblib
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # self.l5_model = joblib.load('models/l5_random_forest.pkl')
            # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            # self.l3_model = AutoModelForSequenceClassification.from_pretrained("models/deberta_finetuned")
            pass

    def predict(self, data: dict) -> dict:
        if self.use_mock:
            return self._mock_predict(data)
        # Real inference would go here
        return self._mock_predict(data)

    def _text_features(self, text: str) -> dict:
        words = text.split()
        caps = sum(1 for c in text if c.isupper())
        exclamations = text.count('!')
        caps_ratio = caps / max(len(text), 1)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

        spam_phrases = [
            "best ever", "absolutely love", "highly recommend", "terrible",
            "worst", "amazing", "perfect", "disgusting", "scam", "fraud",
            "do not buy", "waste of money", "best product", "must buy"
        ]
        phrase_hits = sum(1 for p in spam_phrases if p in text.lower())

        return {
            "word_count": len(words),
            "caps_ratio": caps_ratio,
            "exclamations": exclamations,
            "avg_word_len": avg_word_len,
            "phrase_hits": phrase_hits,
        }

    def _mock_predict(self, data: dict) -> dict:
        text = data["text"]
        rating = data["rating"]
        tenure_days = data["tenure_days"]
        seller_concentration = data["seller_concentration"]
        burst_score = data["burst_score"]
        review_count = data["review_count"]

        tf = self._text_features(text)

        # ---- L1: ETL / Rule-based signals ----
        l1_score = 0.0
        l1_factors = []
        if rating in [1, 5]:
            l1_score += 0.15
        if tenure_days < 30:
            l1_score += 0.2
            l1_factors.append(f"L1 Rule: Account only {tenure_days} days old — new-account bias detected")
        if seller_concentration > 0.8:
            l1_score += 0.2
            l1_factors.append(f"L1 Rule: Seller concentration {seller_concentration:.0%} — single-seller review pattern")

        # ---- L2: FP-Growth behavioral patterns ----
        l2_score = 0.0
        l2_factors = []
        if burst_score >= 5:
            l2_score += 0.2
            l2_factors.append(f"L2 FP-Growth: Burst score {burst_score} — review flooding pattern matched")
        if review_count < 3 and seller_concentration > 0.6:
            l2_score += 0.15
            l2_factors.append("L2 FP-Growth: Low review count + high concentration ↔ coordinated-bot itemset")

        # ---- L3: DeBERTa NLP signals ----
        l3_score = 0.0
        l3_factors = []
        if tf["caps_ratio"] > 0.15:
            l3_score += 0.1
            l3_factors.append(f"L3 DeBERTa: High capitalisation ratio ({tf['caps_ratio']:.0%}) — aggressive tone")
        if tf["exclamations"] >= 2:
            l3_score += 0.1
            l3_factors.append(f"L3 DeBERTa: {tf['exclamations']} exclamation marks — synthetic enthusiasm")
        if tf["phrase_hits"] >= 2:
            l3_score += 0.15
            l3_factors.append(f"L3 DeBERTa: {tf['phrase_hits']} spam-phrase matches — scripted language")
        if tf["word_count"] < 8:
            l3_score += 0.1
            l3_factors.append("L3 DeBERTa: Review too short — low information content")

        # ---- L4: K-Means / DBSCAN cluster risk ----
        l4_score = 0.0
        l4_factors = []
        if tenure_days < 60 and burst_score >= 3:
            l4_score += 0.15
            l4_factors.append("L4 Clustering: Assigned to high-risk cluster (new + burst)")
        if seller_concentration > 0.7 and review_count < 5:
            l4_score += 0.1
            l4_factors.append("L4 Clustering: Outlier profile detected by DBSCAN isolation")

        # ---- L5: MLP / Random Forest final score ----
        base = l1_score + l2_score + l3_score + l4_score + 0.05
        # Sigmoid-like squashing
        prob = 1 / (1 + math.exp(-6 * (base - 0.45)))
        prob = round(min(max(prob, 0.03), 0.99), 4)

        # Aggregate top factors
        all_factors = l1_factors + l2_factors + l3_factors + l4_factors
        if not all_factors:
            all_factors = ["All signals nominal — no suspicious pattern detected"]

        tier = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")

        return {
            "is_spam": prob > 0.425,
            "spam_probability": prob,
            "behavioral_risk_tier": tier,
            "top_factors": all_factors[:5],
            "layer_scores": {
                "L1_rules": round(l1_score, 3),
                "L2_fpgrowth": round(l2_score, 3),
                "L3_deberta": round(l3_score, 3),
                "L4_clustering": round(l4_score, 3),
                "L5_ensemble": round(prob, 3),
            },
        }


pipeline = FakeReviewPipeline(use_mock=True)
