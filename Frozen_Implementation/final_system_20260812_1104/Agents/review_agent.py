import pandas as pd

from pydantic import BaseModel
from langchain_openai import ChatOpenAI


# =====================================================
# Structured Output Schema
# =====================================================

class ReviewSummary(BaseModel):
    pros: list[str]
    cons: list[str]
    overall_sentiment: str
    recommended_for: str
    avoid_if: str
    summary: str


# =====================================================
# Review Agent
# =====================================================

class ReviewAgent:

    def __init__(self, llm: ChatOpenAI, reviews_df: pd.DataFrame):

        self.llm = llm
        self.reviews_df = reviews_df
        self.cache = {}

        self.structured_llm = llm.with_structured_output(
            ReviewSummary
        )

    def summarize_reviews(
        self,
        parent_asin: str,
        max_reviews: int = 20
    ) -> dict:

        # ----------------------------------------
        # Return cached result
        # ----------------------------------------

        if parent_asin in self.cache:
            return self.cache[parent_asin]

        # ----------------------------------------
        # Retrieve reviews
        # ----------------------------------------

        product_reviews = self.reviews_df[
            self.reviews_df["parent_asin"] == parent_asin
        ]

        # ----------------------------------------
        # No reviews available
        # ----------------------------------------

        if product_reviews.empty:

            result = {
                "parent_asin": parent_asin,
                "title": "Unknown Product",
                "review_count": 0,
                "average_rating": 0,
                "pros": [],
                "cons": [],
                "overall_sentiment": "Unknown",
                "recommended_for": "",
                "avoid_if": "",
                "summary": "No customer reviews available.",
                "sample_reviews": ""
            }

            self.cache[parent_asin] = result

            return result

        # ----------------------------------------
        # Review statistics
        # ----------------------------------------

        product_reviews = product_reviews.sort_values(
            by="rating",
            ascending=False
        )

        average_rating = round(
            product_reviews["rating"].mean(),
            2
        )

        review_count = len(product_reviews)

        title = product_reviews.iloc[0].get(
            "product_title",
            "Unknown Product"
        )

        reviews_text = "\n\n".join(

            product_reviews["text"]

            .dropna()

            .head(max_reviews)

            .tolist()

        )

        # ----------------------------------------
        # Prompt
        # ----------------------------------------

        prompt = f"""
You are an expert ecommerce product analyst.

Product Title:
{title}

Average Rating:
{average_rating}

Number of Reviews:
{review_count}

Read the customer reviews below.

Identify:

• Pros
• Cons
• Overall Sentiment
• Recommended For
• Avoid If
• Short Summary

Customer Reviews:

{reviews_text}
"""

        # ----------------------------------------
        # LLM Summary
        # ----------------------------------------

        try:

            summary = self.structured_llm.invoke(prompt)

        except Exception:

            summary = ReviewSummary(

                pros=[],

                cons=[],

                overall_sentiment="Unknown",

                recommended_for="",

                avoid_if="",

                summary="Review summarization failed."

            )

        # ----------------------------------------
        # Final Output
        # ----------------------------------------

        result = {

            "parent_asin": parent_asin,

            "title": title,

            "review_count": review_count,

            "average_rating": average_rating,

            "pros": summary.pros,

            "cons": summary.cons,

            "overall_sentiment": summary.overall_sentiment,

            "recommended_for": summary.recommended_for,

            "avoid_if": summary.avoid_if,

            "summary": summary.summary,

            "sample_reviews": reviews_text

        }

        # ----------------------------------------
        # Cache Result
        # ----------------------------------------

        self.cache[parent_asin] = result

        return result