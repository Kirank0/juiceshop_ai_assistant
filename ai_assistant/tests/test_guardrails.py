import pytest

from app.guardrails import is_recipe_request


@pytest.mark.parametrize("text", [
    "Give me a recipe for apple juice",
    "What's the recipe of the Eggfruit Juice?",
    "How is the apple juice prepared?",
    "How do you make orange juice?",
    "how to make lemon juice",
    "How can I brew this at home?",
    "Can you explain the production process of the juice?",
    "Step by step guide please",
    "Is there a homemade version I can do myself?",
    "I want to make it myself at home",
    "List the ingredients and their quantities",
    "How are your juices manufactured?",
])
def test_blocks_recipe_requests(text):
    assert is_recipe_request(text)


@pytest.mark.parametrize("text", [
    "How much is the Apple Juice?",
    "What is the deluxe price of Orange Juice?",
    "What does 3 bottles of Lemon Juice cost?",
    "Which juices do you have?",
    "What is the cheapest product?",
    "Tell me about the Eggfruit Juice",
    "How many products do you sell?",
    "Is the OWASP hoodie available in your shop?",
])
def test_allows_product_and_pricing_questions(text):
    assert not is_recipe_request(text)
