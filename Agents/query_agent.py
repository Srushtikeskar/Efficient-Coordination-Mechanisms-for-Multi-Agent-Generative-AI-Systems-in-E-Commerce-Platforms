
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class ProductQuery(BaseModel):

    product_type: str = Field(
        description=(
            "The type of product requested by the user, "
            "such as smartphone, headphones, laptop, "
            "tablet, smartwatch, or camera."
        )
    )

    brand: str | None = Field(
        default=None,
        description=(
            "The requested brand, or null if no brand "
            "is explicitly mentioned."
        )
    )

    budget: float | None = Field(
        default=None,
        description=(
            "Maximum budget as a numeric value only. "
            "Return null if no budget is mentioned."
        )
    )

    features: list[str] = Field(
        default_factory=list,
        description=(
            "Include only product features or use cases "
            "explicitly mentioned by the user. "
            "Do not infer technical specifications that "
            "the user did not request. "
            "For example, 'smartphone for gaming' should "
            "include 'gaming', while 'best Samsung phone' "
            "should return an empty feature list."
        )
    )


class QueryAgent:

    def __init__(
        self,
        model: str = "gpt-4o-mini"
    ):

        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )

        self.structured_llm = (
            self.llm.with_structured_output(
                ProductQuery
            )
        )

    def parse(
        self,
        user_query: str
    ) -> ProductQuery:

        # -------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------

        if (
            not user_query
            or not user_query.strip()
        ):
            raise ValueError(
                "User query cannot be empty."
            )

        # -------------------------------------------------
        # LLM STRUCTURED EXTRACTION
        # -------------------------------------------------

        prompt = f"""
        Extract the product request from the user query.

        Rules:

        1. Identify the requested product type.

        2. Do not invent a brand.
           If no brand is explicitly mentioned,
           return null.

        3. If no budget is mentioned,
           return null.

        4. If a budget is mentioned,
           return only its numeric value.

        5. Include product features explicitly
           mentioned by the user.

        6. Preserve explicit use-case terms such as:
           gaming,
           photography,
           travel,
           work,
           office use,
           fitness,
           studying,
           content creation.

        7. Do not expand a use case into inferred
           technical specifications.

           Example:
           "smartphone for gaming"

           Include:
           ["gaming"]

           Do NOT automatically add:
           "high RAM",
           "high refresh rate",
           "powerful processor",
           etc.

        8. Do not add features the user did not request.

        9. Examples:

           Query:
           "Suggest a smartphone for gaming."

           Expected:
           product_type = smartphone
           features = ["gaming"]

           Query:
           "Recommend a Samsung phone with a good camera,
           long battery life and fast performance."

           Expected:
           product_type = smartphone
           brand = Samsung
           features should preserve all three
           explicitly requested requirements.

           Query:
           "I need something good for photography."

           Expected:
           product_type = smartphone
           features = ["photography"]

           Query:
           "Recommend the best Samsung phone."

           Expected:
           product_type = smartphone
           brand = Samsung
           features = []

        User query:
        {user_query}
        """

        structured_query = (
            self.structured_llm.invoke(
                prompt
            )
        )

        # -------------------------------------------------
        # NORMALIZED ORIGINAL QUERY
        # -------------------------------------------------

        query_text = (
            user_query
            .strip()
            .lower()
        )

        explicit_feature_patterns = {

            # ---------------------------------------------
            # CAMERA / PHOTOGRAPHY
            # ---------------------------------------------

            "good camera":
                "camera",

            "great camera":
                "camera",

            "excellent camera":
                "camera",

            "excellent camera quality":
                "camera",

            "good camera quality":
                "camera",

            "great camera quality":
                "camera",

            "camera quality":
                "camera",

            # ---------------------------------------------
            # BATTERY
            # ---------------------------------------------

            "long battery life":
                "long battery life",

            "good battery life":
                "long battery life",

            "great battery life":
                "long battery life",

            "excellent battery life":
                "long battery life",

            "long lasting battery":
                "long battery life",

            "removable battery":
                "removable battery",

            # ---------------------------------------------
            # PERFORMANCE
            # ---------------------------------------------

            "fast performance":
                "fast performance",

            "good performance":
                "performance",

            "great performance":
                "performance",

            "excellent performance":
                "performance",

            "high performance":
                "performance",

            # ---------------------------------------------
            # STORAGE
            # ---------------------------------------------

            "large storage":
                "large storage",

            "high storage":
                "large storage",

            "more storage":
                "large storage",

            # ---------------------------------------------
            # DISPLAY
            # ---------------------------------------------

            "large screen":
                "large screen",

            "big screen":
                "large screen",

            "good display":
                "good display",

            "great display":
                "good display",

            "excellent display":
                "good display",

            # ---------------------------------------------
            # CHARGING
            # ---------------------------------------------

            "fast charging":
                "fast charging",

            "quick charging":
                "fast charging",

            "wireless charging":
                "wireless charging",

            # ---------------------------------------------
            # DURABILITY
            # ---------------------------------------------

            "water resistant":
                "water resistant",

            "water-resistant":
                "water resistant",

            "waterproof":
                "water resistant",

            # ---------------------------------------------
            # AUDIO
            # ---------------------------------------------

            "good sound quality":
                "good sound quality",

            "great sound quality":
                "good sound quality",

            "excellent sound quality":
                "good sound quality",

            "sound quality":
                "good sound quality",

            # ---------------------------------------------
            # INPUT / ACCESSORY SUPPORT
            # ---------------------------------------------

            "stylus support":
                "stylus support",

            "stylus":
                "stylus support",

            # ---------------------------------------------
            # PORTABILITY
            # ---------------------------------------------

            "lightweight":
                "lightweight",

            "light weight":
                "lightweight",

            # ---------------------------------------------
            # USE CASES
            # ---------------------------------------------

            "gaming":
                "gaming",

            "photography":
                "photography",

            "travel":
                "travel",

            "office use":
                "work",

            "work":
                "work",

            "fitness":
                "fitness",

            "content creation":
                "content creation",

            "content creator":
                "content creation",

            "studying":
                "studying",

            "study":
                "studying"
        }



        existing_features = []

        for feature in (
            structured_query.features
            or []
        ):

            cleaned_feature = (
                str(feature)
                .strip()
                .lower()
            )

            if (
                cleaned_feature
                and cleaned_feature
                not in existing_features
            ):
                existing_features.append(
                    cleaned_feature
                )


        for (
            phrase,
            feature_name
        ) in explicit_feature_patterns.items():

            if (
                phrase in query_text
                and feature_name
                not in existing_features
            ):
                existing_features.append(
                    feature_name
                )


        photography_phrases = [
            "good for photography",
            "great for photography",
            "best for photography",
            "for photography"
        ]

        if any(
            phrase in query_text
            for phrase in photography_phrases
        ):

            if (
                "photography"
                not in existing_features
            ):
                existing_features.append(
                    "photography"
                )



        structured_query.features = (
            existing_features
        )

 

        product_type = (
            str(
                structured_query.product_type
            )
            .strip()
            .lower()
        )

        product_type_aliases = {

            # Phones
            "phone":
                "smartphone",

            "phones":
                "smartphone",

            "mobile":
                "smartphone",

            "mobile phone":
                "smartphone",

            "cell phone":
                "smartphone",

            "cellphone":
                "smartphone",

            "smart phone":
                "smartphone",

            "smartphone":
                "smartphone",

            # Headsets / headphones
            "headphone":
                "headset",

            "headphones":
                "headset",

            "earphone":
                "headset",

            "earphones":
                "headset",

            "earbud":
                "headset",

            "earbuds":
                "headset",

            "headset":
                "headset",

            # Laptops
            "notebook":
                "laptop",

            "macbook":
                "laptop",

            "chromebook":
                "laptop",

            "laptop":
                "laptop",

            # Tablets
            "ipad":
                "tablet",

            "tablet":
                "tablet",

            # Smartwatches
            "smart watch":
                "smartwatch",

            "watch":
                "smartwatch",

            "smartwatch":
                "smartwatch",

            # Cameras
            "digital camera":
                "camera",

            "dslr":
                "camera",

            "mirrorless camera":
                "camera",

            "camera":
                "camera"
        }

        structured_query.product_type = (
            product_type_aliases.get(
                product_type,
                product_type
            )
        )

        smartphone_context_terms = {
            "gaming",
            "photography",
            "camera phone",
            "mobile photography",
            "phone",
            "smartphone",
            "mobile phone",
            "cell phone"
        }

        ambiguous_product_types = {
            "",
            "something",
            "something good",
            "something good for photography",
            "device",
            "product",
            "mobile device"
        }

        if any(
            term in query_text
            for term in smartphone_context_terms
        ):

            if (
                structured_query.product_type
                in ambiguous_product_types
            ):
                structured_query.product_type = (
                    "smartphone"
                )

        if structured_query.brand:

            structured_query.brand = (
                str(
                    structured_query.brand
                )
                .strip()
            )

        structured_query.features = list(
            dict.fromkeys(
                structured_query.features
            )
        )

        return structured_query


if __name__ == "__main__":

    query_agent = QueryAgent()

    test_queries = [

        (
            "I need a Samsung phone under 30000 "
            "with good camera and long battery life."
        ),

        "Suggest a smartphone for gaming.",

        "I need something good for photography.",

        "Recommend the best Samsung phone.",

        (
            "Recommend a Samsung phone with a good camera, "
            "long battery life and fast performance."
        ),

        (
            "Recommend a Samsung phone with a removable "
            "battery and an excellent camera."
        ),

        (
            "Recommend an Apple phone with stylus support."
        ),

        (
            "Recommend an Apple phone with a good camera "
            "and large storage."
        )
    ]

    for test_query in test_queries:

        response = query_agent.parse(
            test_query
        )

        print("=" * 100)
        print("Query:", test_query)
        print(
            response.model_dump()
        )
        print()