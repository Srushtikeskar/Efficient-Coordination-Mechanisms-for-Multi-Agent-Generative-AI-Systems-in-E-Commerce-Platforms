# =====================================================
# RETRIEVAL AGENT
# =====================================================

import math
import re
from typing import Iterable

from pydantic import BaseModel, Field


# =====================================================
# QUERY SCHEMA
# =====================================================

class ProductQuery(BaseModel):

    product_type: str

    brand: str | None = None

    budget: float | str | None = None

    features: list[str] = Field(
        default_factory=list
    )


# =====================================================
# PRODUCT-TYPE NORMALISATION
# =====================================================

PRODUCT_TYPE_ALIASES = {

    # Phones
    "phone": "phone",
    "phones": "phone",
    "smartphone": "phone",
    "smartphones": "phone",
    "smart phone": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "cell phone": "phone",
    "cellphone": "phone",

    # Headsets
    "headset": "headset",
    "headsets": "headset",
    "headphone": "headset",
    "headphones": "headset",
    "earphone": "headset",
    "earphones": "headset",
    "earbud": "headset",
    "earbuds": "headset",
    "airpods": "headset",

    # Laptops
    "laptop": "laptop",
    "laptops": "laptop",
    "notebook": "laptop",
    "notebooks": "laptop",
    "macbook": "laptop",
    "chromebook": "laptop",

    # Tablets
    "tablet": "tablet",
    "tablets": "tablet",
    "ipad": "tablet",

    # Smartwatches
    "smartwatch": "smartwatch",
    "smart watch": "smartwatch",
    "smartwatches": "smartwatch",
    "watch": "smartwatch",
    "fitness watch": "smartwatch",

    # Cameras
    "camera": "camera",
    "digital camera": "camera",
    "dslr": "camera",
    "mirrorless camera": "camera"
}


# =====================================================
# SEARCH ALIASES
# =====================================================
#
# IMPORTANT:
# These aliases describe the PRODUCT TYPE only.
#
# We intentionally do NOT insert every known brand
# into every search query.
#
# Example:
# Samsung query should not automatically include
# Apple, Pixel, Black Shark, etc.
# =====================================================

SEARCH_ALIASES = {

    "phone": (
        "phone smartphone mobile cell phone "
        "cellphone android phone"
    ),

    "headset": (
        "headset headphones headphone "
        "earphones earphone earbuds earbud "
        "wireless audio bluetooth"
    ),

    "laptop": (
        "laptop notebook macbook "
        "chromebook ultrabook computer"
    ),

    "tablet": (
        "tablet ipad android tablet galaxy tab"
    ),

    "smartwatch": (
        "smartwatch smart watch fitness watch wearable"
    ),

    "camera": (
        "camera digital camera dslr mirrorless"
    )
}


# =====================================================
# PRODUCT-TYPE TERMS
# =====================================================

PRODUCT_TYPE_TERMS = {

    "phone": [
        "phone",
        "smartphone",
        "smart phone",
        "mobile",
        "cell phone",
        "cellphone",
        "android phone",
        "iphone",
        "galaxy",
        "pixel",
        "oneplus",
        "black shark",
        "redmagic",
        "rog phone"
    ],

    "headset": [
        "headset",
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "earbud",
        "earbuds",
        "airpods"
    ],

    "laptop": [
        "laptop",
        "notebook",
        "macbook",
        "chromebook",
        "ultrabook"
    ],

    "tablet": [
        "tablet",
        "ipad",
        "galaxy tab"
    ],

    "smartwatch": [
        "smartwatch",
        "smart watch",
        "apple watch",
        "galaxy watch",
        "fitness watch",
        "wearable"
    ],

    "camera": [
        "camera",
        "digital camera",
        "dslr",
        "mirrorless"
    ]
}


# =====================================================
# ACCESSORY / NON-PRODUCT FILTERS
# =====================================================

NEGATIVE_KEYWORDS = {

    "phone": [
        "case",
        "cover",
        "screen protector",
        "protector",
        "charger",
        "adapter",
        "cable",

        "stylus pen",
        "touch pen",
        "digital pen",

        "tripod",
        "holder",
        "mount",
        "holster",
        "belt clip",
        "pouch",
        "phone grip",

        "camera lens",
        "phone lens",
        "telephoto lens",
        "fisheye lens",
        "macro lens",
        "lens kit",

        "replacement battery",
        "battery cover",
        "battery case",

        "replacement screen",
        "replacement part",

        "selfie stick",
        "selfie remote",

        "game controller",
        "gaming controller",

        "phone cooler",
        "cooling case",

        "screen magnifier",

        "vr headset",
        "virtual reality headset",

        "power bank",

        "purse",

        "accessory",
        "accessories",

        "ring light",
        "selfie ring light",
        "flash fill light",
        "camera light",
        "led camera light",
        "battery for",
        "replacement for",
        "compatible with iphone",
        "compatible with samsung",

        "armband",
        "running armband",
        "arm bag",
        "armband case",

        "bluetooth earpiece",
        "wireless earpiece",
        "phone earpiece",
        "handsfree earpiece",
        "headset for cell phone",

        "battery replacement",
        "battery replaccement",
        "spare battery",

        "voip phone",
        "sip phone",
        "desk phone",
        "office phone",
        "landline phone",

                # Wearable / carrying accessories
        "armband",
        "arm bag",
        "armbag",

        # Audio accessories
        "bluetooth earpiece",
        "wireless earpiece",
        "phone earpiece",
        "headset for cell phone",

        # Replacement batteries
        "battery replacement",
        "battery replaccement",
        "spare battery",
        "replacement battery",

        # Non-mobile telephone products
        "voip phone",
        "sip phone",
        "desk phone",
        "landline phone",
    ],

    "headset": [
        "adapter",
        "jack",
        "connector",
        "converter",
        "replacement cable",
        "charging cable",
        "case",
        "cover",
        "holder",
        "stand",
        "ear pads",
        "ear cushions"
    ],

    "laptop": [
        "charger",
        "keyboard cover",
        "screen protector",
        "case",
        "replacement battery",
        "adapter",
        "sleeve",
        "stand",
        "cooling pad"
    ],

    "tablet": [
        "case",
        "cover",
        "screen protector",
        "stylus pen",
        "keyboard case",
        "stand",
        "holder",
        "charger",
        "adapter"
    ],

    "smartwatch": [
        "band",
        "strap",
        "case",
        "cover",
        "screen protector",
        "charger",
        "charging cable",
        "stand"
    ],

    "camera": [
        "camera bag",
        "case",
        "cover",
        "tripod",
        "lens cap",
        "lens filter",
        "strap",
        "battery charger",
        "memory card",
        "remote",
        "adapter"
    ]
}


# =====================================================
# KNOWN BRANDS
# =====================================================

KNOWN_BRANDS = [
    "apple",
    "samsung",
    "google",
    "oneplus",
    "xiaomi",
    "realme",
    "oppo",
    "vivo",
    "motorola",
    "nokia",
    "sony",
    "asus",
    "black shark",
    "redmagic",
    "lenovo",
    "huawei",
    "honor",
    "nothing",
    "blu",
    "cubot",
    "oukitel",
    "ulefone",
    "kxd",
    "jbl",
    "boat",
    "beats",
    "bose",
    "sennheiser",
    "anker",
    "acer",
    "dell",
    "hp",
    "msi"
]


# =====================================================
# FEATURE NORMALISATION
# =====================================================

FEATURE_ALIASES = {

    "camera": [
        "camera",
        "good camera",
        "great camera",
        "excellent camera",
        "camera quality",
        "excellent camera quality",
        "good camera quality",
        "great camera quality",
        "pro grade camera",
        "pro camera",
        "camera phone",
        "8k camera"
    ],

    "long battery life": [
        "long battery life",
        "good battery life",
        "great battery life",
        "excellent battery life",
        "all day battery",
        "all day intelligent battery",
        "large battery"
    ],

    "fast performance": [
        "fast performance",
        "good performance",
        "great performance",
        "excellent performance",
        "smooth performance",
        "high performance",
        "fast processor"
    ],

    "gaming": [
        "gaming",
        "gaming phone",
        "gaming performance",
        "game performance"
    ],

    "photography": [
        "photography",
        "mobile photography",
        "camera photography"
    ],

    "removable battery": [
        "removable battery",
        "replaceable battery",
        "user removable battery"
    ],

    "large storage": [
        "large storage",
        "high storage",
        "ample storage",
        "128gb",
        "256gb",
        "512gb",
        "1tb"
    ],

    "stylus support": [
        "stylus support",
        "supports stylus",
        "stylus compatible"
    ],

    "fast charging": [
        "fast charging",
        "quick charging",
        "rapid charging"
    ],

    "wireless charging": [
        "wireless charging"
    ],

    "water resistant": [
        "water resistant",
        "water resistance",
        "waterproof",
        "ip67",
        "ip68"
    ],

    "good sound quality": [
        "good sound quality",
        "great sound quality",
        "excellent sound quality",
        "sound quality",
        "audio quality",
        "clear sound"
    ]
}


# =====================================================
# BASIC HELPERS
# =====================================================

def normalise_text(
    value
) -> str:

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip().lower()


def canonical_product_type(
    product_type: str
) -> str:

    cleaned = normalise_text(
        product_type
    )

    return PRODUCT_TYPE_ALIASES.get(
        cleaned,
        cleaned
    )


def safe_float(
    value,
    default: float = 0.0
) -> float:

    try:

        if value is None:
            return default

        if isinstance(
            value,
            str
        ):

            value = (
                value
                .replace(",", "")
                .strip()
            )

            if not value:
                return default

        result = float(
            value
        )

        if (
            math.isnan(result)
            or math.isinf(result)
        ):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):
        return default


def contains_any(
    text: str,
    terms: Iterable[str]
) -> bool:

    return any(
        term in text
        for term in terms
    )


# =====================================================
# BRAND EXTRACTION
# =====================================================

def extract_brand(
    title: str
) -> str | None:

    cleaned_title = normalise_text(
        title
    )

    # Longest brand names first.
    # Example:
    # "black shark" before shorter tokens.
    for brand in sorted(
        KNOWN_BRANDS,
        key=len,
        reverse=True
    ):

        pattern = (
            rf"\b{re.escape(brand)}\b"
        )

        if re.search(
            pattern,
            cleaned_title
        ):
            return brand

    return None


# =====================================================
# REQUESTED FEATURE NORMALISATION
# =====================================================

def canonical_feature(
    feature: str
) -> str:

    cleaned = normalise_text(
        feature
    )

    if not cleaned:
        return ""

    for canonical_name, aliases in (
        FEATURE_ALIASES.items()
    ):

        if cleaned == canonical_name:
            return canonical_name

        if cleaned in aliases:
            return canonical_name

    return cleaned


def normalise_requested_features(
    features: list[str] | None
) -> list[str]:

    normalised = []

    for feature in (
        features or []
    ):

        canonical = canonical_feature(
            feature
        )

        if (
            canonical
            and canonical
            not in normalised
        ):
            normalised.append(
                canonical
            )

    return normalised


# =====================================================
# SEARCH QUERY BUILDER
# =====================================================

def build_search_query(
    query: ProductQuery
) -> str:

    canonical_type = (
        canonical_product_type(
            query.product_type
        )
    )

    aliases = SEARCH_ALIASES.get(
        canonical_type,
        canonical_type
    )

    brand_text = normalise_text(
        query.brand
    )

    requested_features = (
        normalise_requested_features(
            query.features
        )
    )

    feature_text = " ".join(
        requested_features
    )

    # Product type
    # + requested brand
    # + requested features
    #
    # We intentionally avoid injecting
    # unrelated brands.
    query_parts = [
        aliases,
        brand_text,
        feature_text
    ]

    return " ".join(
        part
        for part in query_parts
        if part
    ).strip()


# =====================================================
# ACTUAL PRODUCT VALIDATION
# =====================================================

def is_actual_product(
    title: str,
    product_type: str
) -> bool:

    cleaned_title = normalise_text(
        title
    )

    canonical_type = (
        canonical_product_type(
            product_type
        )
    )

    positive_terms = (
        PRODUCT_TYPE_TERMS.get(
            canonical_type,
            [canonical_type]
        )
    )

    negative_terms = (
        NEGATIVE_KEYWORDS.get(
            canonical_type,
            []
        )
    )

    # Must contain evidence that it is
    # actually the requested type.
    if not contains_any(
        cleaned_title,
        positive_terms
    ):
        return False

    # Reject obvious accessories.
    if contains_any(
        cleaned_title,
        negative_terms
    ):
        return False

    return True


# =====================================================
# PRODUCT FILTERING
# =====================================================

def filter_products(
    results,
    query: ProductQuery,
    min_rating: float = 0.0
):

    filtered = []

    canonical_type = (
        canonical_product_type(
            query.product_type
        )
    )

    requested_brand = normalise_text(
        query.brand
    )

    positive_terms = (
        PRODUCT_TYPE_TERMS.get(
            canonical_type,
            [canonical_type]
        )
    )

    negative_terms = (
        NEGATIVE_KEYWORDS.get(
            canonical_type,
            []
        )
    )

    budget = safe_float(
        query.budget,
        default=0.0
    )

    for (
        document,
        score
    ) in results:

        title = normalise_text(
            document.metadata.get(
                "title",
                ""
            )
        )

        categories = normalise_text(
            document.metadata.get(
                "categories",
                ""
            )
        )

        page_text = normalise_text(
            getattr(
                document,
                "page_content",
                ""
            )
        )

        searchable_text = (
            f"{title} "
            f"{categories} "
            f"{page_text}"
        )

        if not title:
            continue

        # =================================================
        # STRICT BRAND FILTER
        # =================================================
        #
        # A Samsung query should not return Black Shark
        # simply because "Samsung" appears somewhere in
        # reviews/category/product text.
        # =================================================

        if requested_brand:

            predicted_brand = extract_brand(
        title
    )

    # Brand should identify the actual product,
    # not merely appear in accessory compatibility text.
            extracted_brand_match = (
                predicted_brand
                == requested_brand
            )

            title_starts_with_brand = (
                title.startswith(
                    requested_brand
                )
            )

            if not (
                extracted_brand_match
                or title_starts_with_brand
            ):
                continue
            
        # =================================================
        # PRODUCT TYPE FILTER
        # =================================================

        if not contains_any(
            searchable_text,
            positive_terms
        ):
            continue

        # =================================================
        # STRONG TITLE VALIDATION
        # =================================================

        if not is_actual_product(
            title=title,
            product_type=canonical_type
        ):
            continue

        # =================================================
        # ACCESSORY EXCLUSION
        # =================================================

        if contains_any(
            title,
            negative_terms
        ):
            continue

        # =================================================
        # BUDGET FILTER
        # =================================================

        price = safe_float(
            document.metadata.get(
                "price"
            ),
            default=0.0
        )

        # Only enforce budget when both:
        # - user supplied a budget
        # - product has usable price data
        if (
            budget > 0
            and price > 0
            and price > budget
        ):
            continue

        # =================================================
        # OPTIONAL MINIMUM RATING
        # =================================================

        rating = safe_float(
            document.metadata.get(
                "average_rating"
            ),
            default=0.0
        )

        # Missing rating does NOT automatically
        # reject a product.
        if (
            min_rating > 0
            and rating > 0
            and rating < min_rating
        ):
        
            continue

        filtered.append(
            (
                document,
                score
            )
        )

    return filtered


# =====================================================
# REMOVE DUPLICATES
# =====================================================

def remove_duplicates(
    results
):

    seen = set()

    unique = []

    for (
        document,
        score
    ) in results:

        asin = normalise_text(
            document.metadata.get(
                "parent_asin"
            )
        )

        if not asin:
            continue

        if asin in seen:
            continue

        seen.add(
            asin
        )

        unique.append(
            (
                document,
                score
            )
        )

    return unique


# =====================================================
# FEATURE MATCHING
# =====================================================

def calculate_feature_match(
    document,
    requested_features: list[str]
) -> float:

    cleaned_features = (
        normalise_requested_features(
            requested_features
        )
    )

    if not cleaned_features:
        return 0.0

    product_text = " ".join(
        [
            normalise_text(
                document.metadata.get(
                    "title",
                    ""
                )
            ),

            normalise_text(
                document.metadata.get(
                    "categories",
                    ""
                )
            ),

            normalise_text(
                getattr(
                    document,
                    "page_content",
                    ""
                )
            )
        ]
    )

    matches = 0

    for feature in cleaned_features:

        aliases = FEATURE_ALIASES.get(
            feature,
            [feature]
        )

        feature_found = any(
            normalise_text(alias)
            in product_text
            for alias in aliases
        )

        if feature_found:
            matches += 1

    return (
        matches
        / len(cleaned_features)
    )


# =====================================================
# RETRIEVAL AGENT
# =====================================================

class RetrievalAgent:

    def __init__(
        self,
        vector_db,
        min_rating: float = 0.0,
        overfetch_factor: int = 5,
        minimum_candidate_pool: int = 50
    ):

        self.vector_db = (
            vector_db
        )

        self.min_rating = max(
            safe_float(
                min_rating,
                default=0.0
            ),
            0.0
        )

        self.overfetch_factor = max(
            int(
                overfetch_factor
            ),
            1
        )

        self.minimum_candidate_pool = max(
            int(
                minimum_candidate_pool
            ),
            1
        )

    def retrieve(
        self,
        query: ProductQuery,
        k: int = 10
    ):

        if query is None:
            raise ValueError(
                "Query cannot be None."
            )

        if not query.product_type:
            raise ValueError(
                "Query product_type cannot be empty."
            )

        requested_k = max(
            int(k),
            1
        )

        # ---------------------------------------------
        # Build semantic query
        # ---------------------------------------------

        search_query = (
            build_search_query(
                query
            )
        )

        # ---------------------------------------------
        # Over-fetch BEFORE filtering
        # ---------------------------------------------
        #
        # FAISS may return accessories in the first few
        # positions.
        #
        # Therefore retrieve a larger candidate pool,
        # filter it, then return Top-k.
        # ---------------------------------------------

        raw_k = max(
            requested_k
            * self.overfetch_factor,

            self.minimum_candidate_pool
        )

        raw_results = (
            self.vector_db
            .similarity_search_with_score(
                search_query,
                k=raw_k
            )
        )

        # ---------------------------------------------
        # Apply constraints
        # ---------------------------------------------

        filtered_results = (
            filter_products(
                results=raw_results,
                query=query,
                min_rating=self.min_rating
            )
        )

        # ---------------------------------------------
        # Remove duplicate ASINs
        # ---------------------------------------------

        unique_results = (
            remove_duplicates(
                filtered_results
            )
        )

        # ---------------------------------------------
        # Re-score valid products
        # ---------------------------------------------

        scored_results = []

        requested_features = (
            normalise_requested_features(
                query.features
            )
        )

        for (
            document,
            similarity_distance
        ) in unique_results:

            distance = max(
                safe_float(
                    similarity_distance,
                    default=0.0
                ),
                0.0
            )

            similarity = (
                1
                / (
                    1
                    + distance
                )
            )

            rating = safe_float(
                document.metadata.get(
                    "average_rating"
                ),
                default=0.0
            )

            rating_number = max(
                safe_float(
                    document.metadata.get(
                        "rating_number"
                    ),
                    default=0.0
                ),
                0.0
            )

            popularity = min(
                math.log10(
                    rating_number
                    + 1
                )
                / 5,
                1.0
            )

            feature_match = (
                calculate_feature_match(
                    document=document,
                    requested_features=(
                        requested_features
                    )
                )
            )

            # =================================================
            # RETRIEVAL SCORE
            # =================================================
            #
            # Semantic similarity remains dominant.
            #
            # Explicit feature support gets enough weight
            # to distinguish products when a user asks
            # for specific requirements.
            # =================================================

            retrieval_score = (

                similarity
                * 0.60

                + (
                    rating
                    / 5
                )
                * 0.15

                + popularity
                * 0.10

                + feature_match
                * 0.15
            )

            canonical_type = (
                canonical_product_type(
                    query.product_type
                )
            )

            # ---------------------------------------------
            # Store evidence for downstream agents
            # ---------------------------------------------

            document.metadata[
                "similarity"
            ] = similarity

            document.metadata[
                "retrieval_score"
            ] = retrieval_score

            document.metadata[
                "feature_match"
            ] = feature_match

            document.metadata[
                "brand"
            ] = extract_brand(
                document.metadata.get(
                    "title",
                    ""
                )
            )

            document.metadata[
                "product_type"
            ] = canonical_type

            document.metadata[
                "requested_features"
            ] = requested_features

            document.metadata[
                "budget"
            ] = query.budget

            scored_results.append(
                (
                    document,
                    retrieval_score
                )
            )

        # ---------------------------------------------
        # Highest score first
        # ---------------------------------------------

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # ---------------------------------------------
        # Final Top-k
        # ---------------------------------------------

        return scored_results[
            :requested_k
        ]


# =====================================================
# OPTIONAL LOCAL TEST
# =====================================================

if __name__ == "__main__":

    print(
        "RetrievalAgent module loaded successfully."
    )