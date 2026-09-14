import json
import math
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel



class RankedProduct(BaseModel):
    """
    One product returned by the Ranking Agent.
    """

    rank: int
    parent_asin: str
    title: str
    average_rating: float
    retrieval_score: float
    final_score: float
    reason: str


class RankingOutput(BaseModel):
    """
    Complete output returned by the Ranking Agent.
    """

    products: list[RankedProduct]


class RankingExplanation(BaseModel):
    """
    LLM-generated explanation for one ranked product.
    """

    parent_asin: str
    reason: str


class RankingExplanationOutput(BaseModel):
    """
    Structured explanations returned by the LLM.
    """

    products: list[RankingExplanation]




def safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Safely convert a value to a finite float.

    Handles:
    - None
    - invalid strings
    - NaN
    - positive infinity
    - negative infinity
    """

    try:
        if value is None:
            return default

        numeric_value = float(value)

        if math.isnan(numeric_value):
            return default

        if math.isinf(numeric_value):
            return default

        return numeric_value

    except (TypeError, ValueError):
        return default


def normalise_text(
    value: Any
) -> str:
    """
    Convert a value into normalized lowercase searchable text.
    """

    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return " ".join(
            normalise_text(item)
            for item in value
        )

    if isinstance(value, dict):
        return " ".join(
            normalise_text(item)
            for item in value.values()
        )

    return " ".join(
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )


def safe_list(
    value: Any
) -> list[Any]:
    """
    Convert a value into a list safely.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return [value]


def truncate_text(
    value: Any,
    max_characters: int
) -> str:
    """
    Convert a value to text and limit its character length.

    This reduces prompt size and therefore reduces LLM latency.
    """

    text = str(
        value or ""
    ).strip()

    if len(text) <= max_characters:
        return text

    return text[
        :max_characters
    ].rstrip()




def sentiment_score(
    sentiment: str
) -> float:
    """
    Convert the Review Agent sentiment label into a
    deterministic numeric value.
    """

    value = normalise_text(
        sentiment
    )

    mapping = {
        "positive": 1.0,
        "mostly positive": 0.8,
        "mixed": 0.5,
        "neutral": 0.5,
        "mostly negative": 0.25,
        "negative": 0.0,
        "unknown": 0.5
    }

    return mapping.get(
        value,
        0.5
    )



FEATURE_ALIASES: dict[str, list[str]] = {
    "good camera": [
        "good camera",
        "great camera",
        "excellent camera",
        "camera quality",
        "photo quality",
        "picture quality",
        "image quality",
        "high quality camera",
        "advanced camera",
        "pro grade camera",
        "pro camera",
        "quad camera",
        "triple camera",
        "dual camera",
        "8k camera",
        "sharp photos",
        "clear photos",
        "excellent camera capabilities",
        "excellent camera quality",
        "good camera capabilities",
        "good camera quality",
        "video quality"
    ],

    "long battery life": [
        "long battery life",
        "long lasting battery",
        "all day battery",
        "all day intelligent battery",
        "extended battery life",
        "battery lasts all day",
        "excellent battery life",
        "good battery life",
        "strong battery life",
        "long battery backup"
    ],

    "fast performance": [
        "fast performance",
        "good performance",
        "excellent performance",
        "smooth performance",
        "high performance",
        "lag free performance",
        "no lag",
        "fast processor",
        "responsive performance"
    ],

    "gaming": [
        "gaming",
        "gaming performance",
        "game performance",
        "high frame rate",
        "smooth gaming",
        "gaming phone"
    ],

    "good display": [
        "good display",
        "great display",
        "excellent display",
        "display quality",
        "screen quality",
        "bright display",
        "bright screen",
        "vivid display",
        "amoled display",
        "oled display"
    ],

    "lightweight": [
        "lightweight",
        "light weight",
        "easy to carry",
        "compact design",
        "portable design"
    ],

    "noise cancellation": [
        "noise cancellation",
        "noise cancelling",
        "noise canceling",
        "active noise cancellation",
        "active noise cancelling",
        "anc"
    ],

    "fitness tracking": [
        "fitness tracking",
        "activity tracking",
        "workout tracking",
        "step tracking",
        "heart rate tracking",
        "health tracking"
    ],

    "water resistant": [
        "water resistant",
        "water resistance",
        "waterproof",
        "ip67",
        "ip68"
    ],

    "large storage": [
        "large storage",
        "ample storage",
        "high storage capacity",
        "128gb",
        "256gb",
        "512gb",
        "1tb"
    ],

    "good sound quality": [
        "good sound quality",
        "great sound quality",
        "excellent sound quality",
        "audio quality",
        "clear sound",
        "stereo speakers"
    ],

    "fast charging": [
        "fast charging",
        "quick charging",
        "rapid charging",
        "super fast charging"
    ]
}


def get_feature_aliases(
    feature: str
) -> list[str]:
    """
    Return normalized aliases for a requested feature.

    For unknown features, only the complete feature phrase is
    returned. Individual word matching is intentionally avoided
    because it can create false-positive feature matches.
    """

    feature_text = normalise_text(
        feature
    )

    if not feature_text:
        return []

    aliases = FEATURE_ALIASES.get(
        feature_text
    )

    if aliases is None:
        return [
            feature_text
        ]

    normalized_aliases = []

    for alias in aliases:
        normalized_alias = normalise_text(
            alias
        )

        if normalized_alias:
            normalized_aliases.append(
                normalized_alias
            )

    return normalized_aliases



def build_product_evidence(
    product: dict[str, Any]
) -> str:
    """
    Combine product-listing and customer-review evidence into
    one normalized text string.

    Only evidence supplied in the product dictionary is used.
    """

    evidence_fields = [
        product.get("title"),
        product.get("product_text"),
        product.get("description"),
        product.get("product_description"),
        product.get("features"),
        product.get("highlights"),
        product.get("categories"),
        product.get("review_summary"),
        product.get("summary"),
        product.get("pros"),
        product.get("cons"),
        product.get("recommended_for"),
        product.get("avoid_if")
    ]

    return normalise_text(
        evidence_fields
    )




def get_feature_match_details(
    query_features: list[str] | None,
    product: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """
    Return matched and unsupported requested features.
    """

    requested_features = (
        query_features or []
    )

    if not requested_features:
        return [], []

    evidence = build_product_evidence(
        product
    )

    matched_features: list[str] = []
    unsupported_features: list[str] = []

    for feature in requested_features:

        aliases = get_feature_aliases(
            feature
        )

        feature_found = any(
            alias in evidence
            for alias in aliases
            if alias
        )

        if feature_found:
            matched_features.append(
                str(feature)
            )

        else:
            unsupported_features.append(
                str(feature)
            )

    return (
        matched_features,
        unsupported_features
    )


def feature_match_score(
    query_features: list[str] | None,
    product: dict[str, Any]
) -> float:
    """
    Calculate the proportion of requested features supported
    by the supplied product evidence.
    """

    requested_features = (
        query_features or []
    )

    if not requested_features:
        return 1.0

    matched_features, _ = (
        get_feature_match_details(
            query_features=requested_features,
            product=product
        )
    )

    return (
        len(matched_features)
        / len(requested_features)
    )



def compute_final_score(
    query: Any,
    product: dict[str, Any]
) -> float:
    """
    Compute the deterministic product-ranking score.

    Weights:
    - Retrieval relevance: 50%
    - Requested-feature match: 20%
    - Customer sentiment: 10%
    - Average rating: 10%
    - Rating-volume popularity: 10%
    """

    retrieval = safe_float(
        product.get(
            "retrieval_score",
            0
        )
    )

    retrieval = min(
        max(
            retrieval,
            0.0
        ),
        1.0
    )

    average_rating = safe_float(
        product.get(
            "average_rating",
            0
        )
    )

    normalized_rating = min(
        max(
            average_rating / 5.0,
            0.0
        ),
        1.0
    )

    rating_number = max(
        safe_float(
            product.get(
                "rating_number",
                0
            )
        ),
        0.0
    )

    popularity = min(
        math.log10(
            rating_number + 1
        ) / 5.0,
        1.0
    )

    sentiment = sentiment_score(
        product.get(
            "overall_sentiment",
            "Unknown"
        )
    )

    query_features = getattr(
        query,
        "features",
        []
    ) or []

    feature_match = feature_match_score(
        query_features=query_features,
        product=product
    )

    # ---------------------------------------
    # Budget penalty
    # ---------------------------------------

    budget_penalty = 0.0

    raw_budget = getattr(
        query,
        "budget",
        None
    )

    budget = (
        safe_float(
            raw_budget,
            default=0.0
        )
        if raw_budget is not None
        else 0.0
    )

    price = safe_float(
        product.get("price"),
        default=0.0
    )

   
    if budget > 0:

        if price == 0:
        
            budget_penalty = 0.10

        elif price > budget:
   
            budget_penalty = 0.25

    # ---------------------------------------
    # Final score
    # ---------------------------------------

    final_score = (
        retrieval * 0.50
        + feature_match * 0.20
        + sentiment * 0.10
        + normalized_rating * 0.10
        + popularity * 0.10
    )

    final_score -= budget_penalty

    return round(
        max(final_score, 0),
        6
    )


# =====================================================
# 8. DETERMINISTIC FALLBACK EXPLANATION
# =====================================================

def create_fallback_reason(
    query: Any,
    product: dict[str, Any]
) -> str:
    """
    Generate a grounded deterministic explanation.

    This explanation is used:
    - for products outside the LLM explanation limit
    - when LLM explanation generation is disabled
    - when the LLM request fails
    - when an explanation is missing from the LLM output
    """

    query_features = getattr(
        query,
        "features",
        []
    ) or []

    matched_features, unsupported_features = (
        get_feature_match_details(
            query_features=query_features,
            product=product
        )
    )

    reason_parts = [
        (
            "Ranked using retrieval relevance, requested-feature "
            "evidence, customer sentiment, average rating and "
            "rating-volume popularity."
        )
    ]

    if matched_features:
        reason_parts.append(
            "Supported requested features: "
            + ", ".join(
                matched_features
            )
            + "."
        )

    if unsupported_features:
        reason_parts.append(
            "Could not verify from the available evidence: "
            + ", ".join(
                unsupported_features
            )
            + "."
        )

    review_count = int(
        max(
            safe_float(
                product.get(
                    "review_count",
                    0
                )
            ),
            0
        )
    )

    if review_count == 0:
        reason_parts.append(
            "No customer-review evidence was available."
        )

    else:
        sentiment = normalise_text(
            product.get(
                "overall_sentiment",
                "Unknown"
            )
        )

        if sentiment and sentiment != "unknown":
            reason_parts.append(
                f"Available customer-review sentiment was {sentiment}."
            )

    average_rating = safe_float(
        product.get(
            "average_rating",
            0
        )
    )

    raw_budget = getattr(
        query,
        "budget",
        None
    )

    budget = (
        safe_float(
            raw_budget,
            default=0.0
        )
        if raw_budget is not None
        else 0.0
    )

    if budget > 0:

        price = safe_float(
            product.get("price"),
            default=0.0
        )

        if price == 0:
            reason_parts.append(
                "Price could not be verified."
            )

        elif price <= budget:
            reason_parts.append(
                "Price is within the requested budget."
            )

        else:
            reason_parts.append(
                "Price exceeds the requested budget."
            )

    # Add rating information whether or not a budget was requested
    if average_rating > 0:
        reason_parts.append(
            f"The average rating was {average_rating:.1f} out of 5."
        )

    return " ".join(reason_parts)


def create_compact_product_payload(
    product: dict[str, Any]
) -> dict[str, Any]:
    """
    Create a compact product representation for the LLM.

    Only the fields required to explain the rank are included.
    Long text fields are truncated to reduce tokens and latency.
    """

    product_text = truncate_text(
        product.get(
            "product_text",
            ""
        ),
        max_characters=350
    )

    review_summary = truncate_text(
        product.get(
            "review_summary",
            ""
        ),
        max_characters=200
    )

    pros = [
        truncate_text(
            item,
            max_characters=120
        )
        for item in safe_list(
            product.get(
                "pros"
            )
        )[:2]
    ]

    cons = [
        truncate_text(
            item,
            max_characters=120
        )
        for item in safe_list(
            product.get(
                "cons"
            )
        )[:1]
    ]

    return {
        "rank": int(
            safe_float(
                product.get(
                    "rank",
                    0
                )
            )
        ),
        "parent_asin": str(
            product.get(
                "parent_asin",
                ""
            )
        ),
        "title": truncate_text(
            product.get(
                "title",
                ""
            ),
            max_characters=220
        ),
        "average_rating": safe_float(
            product.get(
                "average_rating",
                0
            )
        ),
        "rating_number": int(
            max(
                safe_float(
                    product.get(
                        "rating_number",
                        0
                    )
                ),
                0
            )
        ),
        "review_count": int(
            max(
                safe_float(
                    product.get(
                        "review_count",
                        0
                    )
                ),
                0
            )
        ),
        "retrieval_score": safe_float(
            product.get(
                "retrieval_score",
                0
            )
        ),
        "final_score": safe_float(
            product.get(
                "final_score",
                0
            )
        ),
        "overall_sentiment": str(
            product.get(
                "overall_sentiment",
                "Unknown"
            )
        ),
        "pros": pros,
        "cons": cons,
        "review_summary": review_summary,
        "product_evidence": product_text
    }


class RankingAgent:
    """
    Deterministic product-ranking agent.

    Numeric scoring and ordering are performed by Python.

    The LLM is used only to explain a limited number of
    top-ranked products. Remaining products receive deterministic
    fallback explanations to reduce latency and token usage.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        explanation_limit: int = 3,
        use_llm_explanations: bool = True
    ):
        """
        Initialize the Ranking Agent.

        Parameters
        ----------
        llm:
            ChatOpenAI model used only for ranking explanations.

        explanation_limit:
            Maximum number of top-ranked products sent to the LLM.
            Products after this limit receive deterministic
            fallback explanations.

            Recommended:
            - 3 for balanced quality and latency
            - 2 for lower latency
            - 0 to disable LLM explanations

        use_llm_explanations:
            When False, the LLM is not called and all reasons are
            generated deterministically.
        """

        self.explanation_limit = max(
            int(explanation_limit),
            0
        )

        self.use_llm_explanations = bool(
            use_llm_explanations
        )

        self.explanation_llm = (
            llm.with_structured_output(
                RankingExplanationOutput
            )
        )

    def _generate_llm_explanations(
        self,
        query: Any,
        scored_products: list[dict[str, Any]]
    ) -> dict[str, str]:
        """
        Generate grounded explanations for only the highest-ranked
        products.

        Returns a dictionary in this form:

        {
            "parent_asin": "explanation"
        }

        An empty dictionary is returned when:
        - LLM explanations are disabled
        - explanation_limit is zero
        - there are no products
        - the LLM call fails
        """

        if not self.use_llm_explanations:
            return {}

        if self.explanation_limit <= 0:
            return {}

        if not scored_products:
            return {}

        products_for_explanation = scored_products[
            :self.explanation_limit
        ]

        compact_products = [
            create_compact_product_payload(
                product
            )
            for product in products_for_explanation
        ]

        if not compact_products:
            return {}

        query_brand = getattr(
            query,
            "brand",
            None
        )

        query_product_type = getattr(
            query,
            "product_type",
            None
        )

        query_budget = getattr(
            query,
            "budget",
            None
        )

        query_features = getattr(
            query,
            "features",
            []
        ) or []

        compact_query = {
            "brand": query_brand,
            "product_type": query_product_type,
            "budget": query_budget,
            "requested_features": query_features
        }

        # Compact JSON without indentation reduces prompt tokens.
        query_json = json.dumps(
            compact_query,
            default=str,
            ensure_ascii=False
        )

        products_json = json.dumps(
            compact_products,
            default=str,
            ensure_ascii=False
        )

        explanation_prompt = f"""
You explain ranks already calculated by a deterministic
e-commerce ranking system.

Do not change the product order, ranks, scores, titles or
identifiers.

QUERY:
{query_json}

RANKED PRODUCTS:
{products_json}

Return one concise explanation for every supplied parent_asin.

Rules:

1. Use only the supplied data. Do not use outside knowledge.

2. Never mention a hardware specification such as megapixels,
processor, RAM, battery capacity, screen size, charging wattage
or storage capacity unless the exact claim appears in the
supplied product evidence.

3. Mention a requested feature only when it is directly supported
by product_evidence, review_summary, pros or cons.

4. If a requested feature is unsupported, state that it could not
be verified from the available evidence.

5. A retrieval score is a relevance signal, not customer evidence.

6. An average rating or rating count does not prove a requested
feature.

7. If review_count is 0, include exactly:
"No customer-review evidence was available."

8. Missing customer reviews indicate limited evidence, not poor
product quality.

9. Do not make unsupported comparisons between products.

10. Do not claim that a product is within or outside budget unless
the supplied evidence contains a verifiable price and comparable
currency.

11. Explain the existing rank rather than suggesting a different
rank.

12. Keep every reason under 80 words.
"""

        try:
            explanation_output = (
                self.explanation_llm.invoke(
                    explanation_prompt
                )
            )

            explanation_map: dict[str, str] = {}

            for item in explanation_output.products:

                parent_asin = str(
                    item.parent_asin
                ).strip()

                reason = str(
                    item.reason
                ).strip()

                if parent_asin and reason:
                    explanation_map[
                        parent_asin
                    ] = reason

            return explanation_map

        except Exception:
            # Ranking should still succeed when the explanation
            # model fails. Deterministic fallback explanations
            # will be used instead.
            return {}

    def rank_products(
        self,
        query: Any,
        products: list[dict[str, Any]]
    ) -> RankingOutput:
        """
        Score, sort and explain enriched product dictionaries.

        Workflow:
        1. Calculate a deterministic final score.
        2. Sort products by final score.
        3. Assign deterministic ranks.
        4. Send only the top N compact products to the LLM.
        5. Use deterministic explanations for all other products.
        """

        if not products:
            return RankingOutput(
                products=[]
            )

        scored_products: list[
            dict[str, Any]
        ] = []

        for product in products:

            if not isinstance(
                product,
                dict
            ):
                continue

            enriched_product = (
                product.copy()
            )

            enriched_product[
                "final_score"
            ] = compute_final_score(
                query=query,
                product=enriched_product
            )

            scored_products.append(
                enriched_product
            )

        if not scored_products:
            return RankingOutput(
                products=[]
            )

        scored_products.sort(
            key=lambda item: (
                safe_float(
                    item.get(
                        "final_score",
                        0
                    )
                ),
                safe_float(
                    item.get(
                        "retrieval_score",
                        0
                    )
                ),
                safe_float(
                    item.get(
                        "average_rating",
                        0
                    )
                ),
                safe_float(
                    item.get(
                        "rating_number",
                        0
                    )
                )
            ),
            reverse=True
        )


        for index, product in enumerate(
            scored_products,
            start=1
        ):
            product["rank"] = index

    
        explanation_map = (
            self._generate_llm_explanations(
                query=query,
                scored_products=scored_products
            )
        )


        ranked_products: list[
            RankedProduct
        ] = []

        for product in scored_products:

            parent_asin = str(
                product.get(
                    "parent_asin",
                    ""
                )
            ).strip()

            reason = explanation_map.get(
                parent_asin,
                ""
            ).strip()

            # Products outside the LLM limit automatically reach
            # this fallback path because they do not appear in
            # explanation_map.
            if not reason:
                reason = create_fallback_reason(
                    query=query,
                    product=product
                )

            ranked_product = RankedProduct(
                rank=int(
                    safe_float(
                        product.get(
                            "rank",
                            0
                        )
                    )
                ),
                parent_asin=parent_asin,
                title=str(
                    product.get(
                        "title",
                        ""
                    )
                ),
                average_rating=safe_float(
                    product.get(
                        "average_rating",
                        0
                    )
                ),
                retrieval_score=safe_float(
                    product.get(
                        "retrieval_score",
                        0
                    )
                ),
                final_score=safe_float(
                    product.get(
                        "final_score",
                        0
                    )
                ),
                reason=reason
            )

            ranked_products.append(
                ranked_product
            )

        return RankingOutput(
            products=ranked_products
        )