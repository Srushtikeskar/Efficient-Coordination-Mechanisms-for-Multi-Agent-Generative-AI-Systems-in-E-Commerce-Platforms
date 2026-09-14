import json
import re
import math
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# 1. OUTPUT MODELS
# ============================================================

class ProductVerification(BaseModel):
    """Verification result for one ranked product."""

    rank: int
    parent_asin: str
    title: str

    status: Literal[
        "passed",
        "passed_with_warning",
        "failed"
    ]

    brand_match: bool
    product_type_match: bool

    feature_status: Literal[
        "full_match",
        "partial_match",
        "no_match",
        "not_requested"
    ]

    feature_match_ratio: float
    matched_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)

    budget_verifiable: bool
    budget_match: bool | None = None

    evidence_strength: Literal[
        "strong",
        "moderate",
        "weak"
    ]

    warnings: list[str] = Field(default_factory=list)
    verification_reason: str


class VerifierOutput(BaseModel):
    """Final output returned by the Verifier Agent."""

    overall_status: Literal[
        "passed",
        "passed_with_warnings",
        "failed"
    ]

    confidence: Literal[
        "high",
        "medium",
        "low"
    ]

    verified_products: list[ProductVerification] = Field(
        default_factory=list
    )

    recommended_product_asin: str | None = None
    recommended_product_title: str | None = None
    summary: str


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalise_price(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return None
        return numeric_value

    value_text = str(value).strip()
    if not value_text:
        return None

    cleaned_value = re.sub(
        r"[^\d.\-]",
        "",
        value_text.replace(",", "")
    )

    if cleaned_value in {"", ".", "-", "-."}:
        return None

    try:
        return float(cleaned_value)
    except (TypeError, ValueError):
        return None


def parse_budget(value: Any) -> float | None:
    return normalise_price(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def model_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


# ============================================================
# 3. BRAND VERIFICATION
# ============================================================

def check_brand_match(
    expected_brand: str | None,
    title: str
) -> bool:
    expected_brand_text = normalise_text(expected_brand)
    title_text = normalise_text(title)

    if not expected_brand_text:
        return True

    return expected_brand_text in title_text


# ============================================================
# 4. PRODUCT TYPE VERIFICATION
# ============================================================

def check_product_type_match(
    product_type: str | None,
    title: str
) -> bool:
    requested_type = normalise_text(product_type)
    title_text = normalise_text(title)

    if not requested_type:
        return True

    product_type_aliases = {
        "phone": [
            "phone", "smartphone", "mobile", "cell phone",
            "cellphone", "iphone", "galaxy", "pixel"
        ],
        "smartphone": [
            "phone", "smartphone", "mobile", "cell phone",
            "cellphone", "iphone", "galaxy", "pixel"
        ],
        "mobile": [
            "phone", "smartphone", "mobile", "cell phone",
            "cellphone", "iphone", "galaxy", "pixel"
        ],
        "laptop": ["laptop", "notebook", "chromebook", "macbook"],
        "tablet": ["tablet", "ipad", "galaxy tab", "surface pro"],
        "headphones": ["headphone", "headphones", "headset", "over-ear", "on-ear"],
        "earbuds": ["earbud", "earbuds", "buds", "in-ear", "airpods"],
        "smartwatch": ["smartwatch", "smart watch", "apple watch", "galaxy watch"],
        "camera": ["camera", "dslr", "mirrorless", "digital camera"],
        "speaker": ["speaker", "bluetooth speaker", "wireless speaker"],
        "television": ["television", "smart tv", "led tv", "oled tv", "qled tv"],
        "tv": ["television", "smart tv", "led tv", "oled tv", "qled tv"]
    }

    accessory_keywords = [
        "case", "cover", "screen protector", "tempered glass", "protector",
        "charger", "charging cable", "charging cord", "usb cable", "adapter",
        "power adapter", "stylus pen", "touch pen", "digital pen",
        "camera lens", "phone lens", "telephoto lens", "lens kit", "lens protector",
        "stand", "holder", "mount", "tripod", "remote", "selfie stick",
        "selfie remote", "replacement battery", "battery case", "replacement screen",
        "replacement part", "skin", "wallet case", "dock", "docking station",
        "strap", "band", "sleeve", "pouch", "accessory", "accessories",
        "ring light", "camera light", "flash fill light", "armband", "arm bag",
        "bluetooth earpiece", "wireless earpiece", "battery replacement",
        "spare battery", "voip phone", "sip phone", "desk phone", "landline phone"
    ]

    aliases = product_type_aliases.get(
        requested_type,
        [requested_type]
    )

    contains_product_type = any(
        alias in title_text
        for alias in aliases
    )

    contains_accessory = any(
        accessory in title_text
        for accessory in accessory_keywords
    )

    if contains_accessory:
        return False

    return contains_product_type


# ============================================================
# 5. FEATURE VERIFICATION
# ============================================================

def build_feature_evidence(
    product: dict[str, Any]
) -> str:
    evidence_parts: list[str] = []

    single_value_fields = [
        "title", "review_summary", "summary", "overall_sentiment",
        "recommended_for", "avoid_if", "description", "features",
        "highlights", "product_text", "product_description"
    ]

    list_value_fields = [
        "pros", "cons", "categories"
    ]

    for field in single_value_fields:
        value = product.get(field)
        if value is not None:
            evidence_parts.append(str(value))

    for field in list_value_fields:
        value = product.get(field)
        if isinstance(value, list):
            evidence_parts.extend(str(item) for item in value)
        elif value is not None:
            evidence_parts.append(str(value))

    return normalise_text(" ".join(evidence_parts))


def normalise_requested_feature_list(
    requested_features: list[str] | None
) -> list[str]:
    requested_features = requested_features or []

    canonical_map = {
        "camera": "camera",
        "good camera": "camera",
        "great camera": "camera",
        "excellent camera": "camera",
        "camera quality": "camera",
        "good camera quality": "camera",
        "great camera quality": "camera",
        "excellent camera quality": "camera",
        "long battery life": "long battery life",
        "good battery life": "long battery life",
        "great battery life": "long battery life",
        "excellent battery life": "long battery life",
        "fast performance": "fast performance",
        "good performance": "fast performance",
        "great performance": "fast performance",
        "excellent performance": "fast performance",
        "large storage": "large storage",
        "high storage": "large storage",
        "good sound": "good sound quality",
        "good sound quality": "good sound quality",
        "great sound quality": "good sound quality",
        "excellent sound quality": "good sound quality",
        "gaming": "gaming",
        "photography": "photography",
        "removable battery": "removable battery",
        "stylus support": "stylus support",
        "fast charging": "fast charging",
        "wireless charging": "wireless charging",
        "water resistant": "water resistant",
        "waterproof": "water resistant"
    }

    normalised = []

    for feature in requested_features:
        cleaned = normalise_text(feature)
        canonical = canonical_map.get(cleaned, cleaned)

        if canonical and canonical not in normalised:
            normalised.append(canonical)

    return normalised


def get_feature_aliases(feature: str) -> list[str]:
    feature_text = normalise_text(feature)

    feature_aliases = {
        "camera": [
            "camera", "good camera", "great camera", "excellent camera",
            "camera quality", "good camera quality", "great camera quality",
            "excellent camera quality", "photo quality", "picture quality",
            "image quality", "high quality camera", "advanced camera",
            "pro grade camera", "pro-grade camera", "sharp photos", "clear photos",
            "excellent camera capabilities", "good camera capabilities"
        ],
        "good camera": [
            "good camera", "great camera", "excellent camera", "camera quality",
            "good camera quality", "great camera quality", "excellent camera quality",
            "photo quality", "picture quality", "image quality", "high quality camera",
            "advanced camera", "pro grade camera", "pro-grade camera",
            "sharp photos", "clear photos", "excellent camera capabilities",
            "good camera capabilities"
        ],
        "photography": [
            "photography", "camera", "camera quality", "photo quality",
            "picture quality", "image quality", "portrait", "sharp photos",
            "clear photos"
        ],
        "long battery life": [
            "long battery life", "long-lasting battery", "long lasting battery",
            "all-day battery", "all day battery", "all-day intelligent battery",
            "all day intelligent battery", "extended battery life",
            "battery lasts all day", "excellent battery life", "great battery life",
            "good battery life", "strong battery life", "long battery backup",
            "5000mah", "6000mah", "6600mah", "7000mah", "8000mah"
        ],
        "fast performance": [
            "fast performance", "good performance", "excellent performance",
            "smooth performance", "high performance", "lag-free performance",
            "lag free performance", "responsive performance", "fast processor", "no lag"
        ],
        "gaming": [
            "gaming", "gaming performance", "game performance", "smooth gaming",
            "gaming phone", "high frame rate", "144hz", "120hz", "snapdragon 870"
        ],
        "good display": [
            "good display", "great display", "excellent display", "display quality",
            "screen quality", "bright display", "bright screen", "vivid display",
            "amoled display", "oled display"
        ],
        "lightweight": [
            "lightweight", "light weight", "easy to carry", "portable design", "compact design"
        ],
        "noise cancellation": [
            "noise cancellation", "noise cancelling", "noise canceling",
            "active noise cancellation", "active noise cancelling", "anc"
        ],
        "fitness tracking": [
            "fitness tracking", "activity tracking", "workout tracking",
            "step tracking", "heart rate tracking", "health tracking"
        ],
        "water resistant": [
            "water resistant", "water-resistant", "water resistance", "waterproof", "ip67", "ip68"
        ],
        "large storage": [
            "large storage", "ample storage", "high storage capacity", "128gb", "256gb", "512gb", "1tb"
        ],
        "good sound quality": [
            "good sound quality", "great sound quality", "excellent sound quality",
            "sound quality", "audio quality", "clear sound"
        ],
        "fast charging": [
            "fast charging", "quick charging", "rapid charging", "super fast charging", "flash charge"
        ],
        "removable battery": [
            "removable battery", "replaceable battery", "user removable battery",
            "user-replaceable battery", "user replaceable battery"
        ],
        "stylus support": [
            "stylus support", "supports stylus", "stylus compatible", "s pen", "s-pen", "pen support"
        ],
        "wireless charging": [
            "wireless charging", "qi charging", "qi wireless"
        ]
    }

    if not feature_text:
        return []

    return feature_aliases.get(
        feature_text,
        [feature_text]
    )


def check_feature_match(
    requested_features: list[str] | None,
    product: dict[str, Any]
) -> tuple[
    Literal[
        "full_match",
        "partial_match",
        "no_match",
        "not_requested"
    ],
    float,
    list[str],
    list[str]
]:
    requested_features = normalise_requested_feature_list(
        requested_features
    )

    if not requested_features:
        return (
            "not_requested",
            1.0,
            [],
            []
        )

    evidence = build_feature_evidence(product)

    matched_features: list[str] = []
    missing_features: list[str] = []

    for feature in requested_features:
        aliases = get_feature_aliases(feature)

        feature_found = any(
            normalise_text(alias) in evidence
            for alias in aliases
            if normalise_text(alias)
        )

        if feature_found:
            matched_features.append(feature)
        else:
            missing_features.append(feature)

    match_ratio = (
        len(matched_features)
        / len(requested_features)
    )

    if match_ratio == 1.0:
        feature_status = "full_match"
    elif match_ratio > 0:
        feature_status = "partial_match"
    else:
        feature_status = "no_match"

    return (
        feature_status,
        match_ratio,
        matched_features,
        missing_features
    )


# ============================================================
# 6. REVIEW-EVIDENCE STRENGTH
# ============================================================

def determine_evidence_strength(
    product: dict[str, Any]
) -> Literal[
    "strong",
    "moderate",
    "weak"
]:
    review_count = safe_int(
        product.get("review_count", 0)
    )

    review_summary = normalise_text(
        product.get(
            "review_summary",
            product.get("summary", "")
        )
    )

    pros = product.get("pros", []) or []
    cons = product.get("cons", []) or []

    recommended_for = normalise_text(
        product.get("recommended_for", "")
    )

    avoid_if = normalise_text(
        product.get("avoid_if", "")
    )

    no_review_messages = {
        "",
        "no customer reviews available.",
        "no customer reviews available",
        "unknown"
    }

    has_real_summary = (
        review_summary not in no_review_messages
    )

    has_review_content = bool(
        has_real_summary
        or pros
        or cons
        or recommended_for
        or avoid_if
    )

    if review_count >= 10 and has_review_content:
        return "strong"

    if review_count >= 3 and has_review_content:
        return "moderate"

    return "weak"


# ============================================================
# 7. RANKING CONSISTENCY
# ============================================================

def verify_rank_order(
    ranked_products: list[dict[str, Any]]
) -> bool:
    if len(ranked_products) <= 1:
        return True

    scores = [
        safe_float(product.get("final_score", 0))
        for product in ranked_products
    ]

    return all(
        scores[index] >= scores[index + 1]
        for index in range(len(scores) - 1)
    )


# ============================================================
# 8. CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(
    verified_products: list[ProductVerification],
    rank_order_valid: bool
) -> Literal[
    "high",
    "medium",
    "low"
]:
    if not verified_products:
        return "low"

    if not rank_order_valid:
        return "low"

    total_products = len(verified_products)

    failed_count = sum(
        product.status == "failed"
        for product in verified_products
    )

    passed_count = sum(
        product.status == "passed"
        for product in verified_products
    )

    strong_evidence_count = sum(
        product.evidence_strength == "strong"
        for product in verified_products
    )

    full_feature_count = sum(
        product.feature_status in {
            "full_match",
            "not_requested"
        }
        for product in verified_products
    )

    if failed_count == total_products:
        return "low"

    passed_ratio = passed_count / total_products
    strong_evidence_ratio = strong_evidence_count / total_products
    full_feature_ratio = full_feature_count / total_products

    if (
        failed_count == 0
        and passed_ratio >= 0.6
        and strong_evidence_ratio >= 0.5
        and full_feature_ratio >= 0.5
    ):
        return "high"

    return "medium"


# ============================================================
# 9. VERIFIER AGENT
# ============================================================

class VerifierAgent:
    """
    Final quality-assurance agent for the recommendation system.

    Verification decisions are deterministic.
    The LLM is used only to generate the final readable summary.
    """

    def __init__(
        self,
        llm: Any,
        use_llm_summary: bool = False
    ):
        self.llm = llm
        self.use_llm_summary = bool(use_llm_summary)

    def verify(
        self,
        query: Any,
        ranking_output: Any,
        original_products: list[dict[str, Any]]
    ) -> VerifierOutput:
        ranked_products = getattr(
            ranking_output,
            "products",
            []
        ) or []

        if not ranked_products:
            return VerifierOutput(
                overall_status="failed",
                confidence="low",
                verified_products=[],
                recommended_product_asin=None,
                recommended_product_title=None,
                summary=(
                    "Verification could not be completed "
                    "because no ranked products were available."
                )
            )

        original_product_map = {
            str(product.get("parent_asin", "")): product
            for product in original_products
        }

        ranked_product_dicts = [
            model_to_dict(product)
            for product in ranked_products
        ]

        rank_order_valid = verify_rank_order(
            ranked_product_dicts
        )

        verified_products: list[ProductVerification] = []

        for ranked_product in ranked_products:
            ranked_product_data = model_to_dict(ranked_product)

            parent_asin = str(
                ranked_product_data.get("parent_asin", "")
            )

            evidence_product = original_product_map.get(
                parent_asin,
                {}
            )

            combined_product = {
                **evidence_product,
                **ranked_product_data
            }

            rank = safe_int(
                ranked_product_data.get("rank", 0)
            )

            title = str(
                ranked_product_data.get("title", "")
            )

            brand_match = check_brand_match(
                getattr(query, "brand", None),
                title
            )

            product_type_match = check_product_type_match(
                getattr(query, "product_type", None),
                title
            )

            (
                feature_status,
                feature_match_ratio,
                matched_features,
                missing_features
            ) = check_feature_match(
                getattr(query, "features", []),
                combined_product
            )

            price = normalise_price(
                combined_product.get("price")
            )

            budget = parse_budget(
                getattr(query, "budget", None)
            )

            budget_was_requested = (
                budget is not None
            )

            budget_verifiable = (
                budget_was_requested
                and price is not None
            )

            budget_match: bool | None = None

            if budget_verifiable:
                budget_match = (
                    price <= budget
                )

            evidence_strength = determine_evidence_strength(
                combined_product
            )

            warnings: list[str] = []

            if not brand_match:
                warnings.append(
                    "The product does not clearly match "
                    "the requested brand."
                )

            if not product_type_match:
                warnings.append(
                    "The product may not match the requested "
                    "product type or may be an accessory."
                )

            if feature_status == "partial_match":
                warnings.append(
                    "Only some of the requested features "
                    "were supported by the available evidence."
                )

            elif feature_status == "no_match":
                warnings.append(
                    "None of the requested features could "
                    "be verified from the available evidence."
                )

            if missing_features:
                warnings.append(
                    "Missing feature evidence: "
                    + ", ".join(missing_features)
                )

            if (
                budget_was_requested
                and not budget_verifiable
            ):
                warnings.append(
                    "Budget compliance could not be verified "
                    "because the product price was unavailable."
                )

            elif budget_match is False:
                warnings.append(
                    "The product exceeds the requested budget."
                )

            if evidence_strength == "weak":
                warnings.append(
                    "The recommendation has limited review "
                    "evidence."
                )

            if not rank_order_valid:
                warnings.append(
                    "The ranking order is inconsistent with "
                    "descending final scores."
                )

            # Hard failures:
            # 1) wrong brand
            # 2) wrong product type/accessory
            # 3) explicit budget exceeded
            # 4) none of the requested features supported
            critical_failure = (
                not brand_match
                or not product_type_match
                or (
                    budget_verifiable
                    and budget_match is False
                )
                or (
                    feature_status == "no_match"
                )
            )

            # Warnings:
            # 1) partial feature support
            # 2) weak review evidence
            # 3) budget requested but price unavailable
            # 4) ranking order inconsistency
            has_warning = (
                feature_status == "partial_match"
                or evidence_strength == "weak"
                or (
                    budget_was_requested
                    and not budget_verifiable
                )
                or not rank_order_valid
            )

            if critical_failure:
                status = "failed"
            elif has_warning:
                status = "passed_with_warning"
            else:
                status = "passed"

            verification_reason = (
                f"Brand match: {brand_match}. "
                f"Product type match: {product_type_match}. "
                f"Feature status: {feature_status}. "
                f"Feature match ratio: {feature_match_ratio:.2f}. "
                f"Evidence strength: {evidence_strength}. "
                f"Price available: {price is not None}. "
                f"Budget requested: {budget_was_requested}. "
                f"Budget match: {budget_match}."
            )

            verified_product = ProductVerification(
                rank=rank,
                parent_asin=parent_asin,
                title=title,
                status=status,
                brand_match=brand_match,
                product_type_match=product_type_match,
                feature_status=feature_status,
                feature_match_ratio=round(
                    feature_match_ratio,
                    3
                ),
                matched_features=matched_features,
                missing_features=missing_features,
                budget_verifiable=budget_verifiable,
                budget_match=budget_match,
                evidence_strength=evidence_strength,
                warnings=warnings,
                verification_reason=verification_reason
            )

            verified_products.append(
                verified_product
            )

        valid_products = [
            product
            for product in verified_products
            if product.status != "failed"
        ]

        failed_products = [
            product
            for product in verified_products
            if product.status == "failed"
        ]

        warning_products = [
            product
            for product in verified_products
            if product.status == "passed_with_warning"
        ]

        if not valid_products:
            overall_status = "failed"
        elif failed_products or warning_products:
            overall_status = "passed_with_warnings"
        else:
            overall_status = "passed"

        confidence = calculate_confidence(
            verified_products=verified_products,
            rank_order_valid=rank_order_valid
        )

        query_budget = parse_budget(
            getattr(query, "budget", None)
        )

        budget_requested = (
            query_budget is not None
        )

        fully_matched_products = [
            product
            for product in verified_products
            if (
                product.status != "failed"
                and product.feature_status in {
                    "full_match",
                    "not_requested"
                }
                and (
                    not budget_requested
                    or product.budget_match is True
                )
            )
        ]

        fully_matched_products.sort(
            key=lambda product: product.rank
        )

        if fully_matched_products:
            recommended_product = (
                fully_matched_products[0]
            )
        else:
            budget_valid_products = [
                product
                for product in verified_products
                if (
                    product.status != "failed"
                    and (
                        not budget_requested
                        or product.budget_match is True
                    )
                )
            ]

            budget_valid_products.sort(
                key=lambda product: product.rank
            )

            recommended_product = (
                budget_valid_products[0]
                if budget_valid_products
                else None
            )

        if self.use_llm_summary:
            summary = self._generate_summary(
                query=query,
                overall_status=overall_status,
                confidence=confidence,
                verified_products=verified_products,
                rank_order_valid=rank_order_valid,
                recommended_product=recommended_product
            )
        else:
            summary = self._fallback_summary(
                overall_status=overall_status,
                confidence=confidence,
                verified_products=verified_products,
                recommended_product=recommended_product
            )

        return VerifierOutput(
            overall_status=overall_status,
            confidence=confidence,
            verified_products=verified_products,
            recommended_product_asin=(
                recommended_product.parent_asin
                if recommended_product
                else None
            ),
            recommended_product_title=(
                recommended_product.title
                if recommended_product
                else None
            ),
            summary=summary
        )

    # ========================================================
    # 10. LLM SUMMARY
    # ========================================================

    def _generate_summary(
        self,
        query: Any,
        overall_status: str,
        confidence: str,
        verified_products: list[ProductVerification],
        rank_order_valid: bool,
        recommended_product: ProductVerification | None
    ) -> str:
        query_features = getattr(
            query,
            "features",
            []
        ) or []

        query_details = {
            "brand": getattr(query, "brand", None),
            "product_type": getattr(query, "product_type", None),
            "budget": getattr(query, "budget", None),
            "features": query_features
        }

        verification_data = [
            product.model_dump()
            for product in verified_products
        ]

        recommended_title = (
            recommended_product.title
            if recommended_product
            else None
        )

        prompt = f"""
You are the explanation component of an e-commerce
recommendation verification system.

All verification decisions have already been calculated
deterministically.

Do not change product ranks, scores, statuses, warnings,
feature results, budget results or confidence.

Customer query:
{json.dumps(query_details, indent=2, default=str)}

Overall status:
{overall_status}

Confidence:
{confidence}

Ranking order valid:
{rank_order_valid}

Recommended product:
{recommended_title}

Product verification results:
{json.dumps(verification_data, indent=2, default=str)}

Write a concise summary of approximately 3 to 5 sentences.

Rules:
- Do not invent product specifications.
- Do not invent feature evidence.
- First check whether the customer supplied a budget.
- If no budget was supplied, state:
  "No budget constraint was specified, so budget compliance was not evaluated."
- Only discuss whether a product is within budget when the customer supplied a budget.
- If a budget was supplied but the product price is unavailable, state that budget compliance could not be verified.
- Mention partial or missing feature evidence where relevant.
- Mention weak review evidence where relevant.
- State the highest-ranked valid recommendation.
- If no product passed verification, state that clearly.
- Do not change the deterministic verification results.
"""

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", "")

            if isinstance(content, str):
                content = content.strip()

            if content:
                return content

        except Exception:
            pass

        return self._fallback_summary(
            overall_status=overall_status,
            confidence=confidence,
            verified_products=verified_products,
            recommended_product=recommended_product
        )

    # ========================================================
    # 11. FALLBACK SUMMARY
    # ========================================================

    def _fallback_summary(
        self,
        overall_status: str,
        confidence: str,
        verified_products: list[ProductVerification],
        recommended_product: ProductVerification | None
    ) -> str:
        if not verified_products:
            return (
                "No products were available for verification."
            )

        failed_count = sum(
            product.status == "failed"
            for product in verified_products
        )

        warning_count = sum(
            len(product.warnings)
            for product in verified_products
        )

        if recommended_product is None:
            return (
                f"Verification finished with status "
                f"'{overall_status}' and {confidence} "
                f"confidence. All {failed_count} products "
                f"failed one or more critical checks, so "
                f"no product could be recommended."
            )

        return (
            f"Verification finished with status "
            f"'{overall_status}' and {confidence} confidence. "
            f"The highest-ranked valid recommendation is "
            f"{recommended_product.title}. "
            f"The verifier identified {failed_count} failed "
            f"product(s) and {warning_count} warning(s) "
            f"across the ranked results."
        )