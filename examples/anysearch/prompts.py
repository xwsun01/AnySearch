"""Prompts for both stages of AnySearch training."""

from __future__ import annotations

PHASE_ONE_PROMPT = """You are an expert assistant designed to solve complex tasks through a rigorous cycle of reasoning and search. Your goal is to answer the user's question accurately while efficiently using your search budget. You have a total of {B} searches available. You must execute a structured loop of Budget, Reasoning, Action, Observation until you have sufficient information to answer.

1. Budget Phase (`<budget>`)
Before each reasoning step, you will see the current search budget:
<budget>remaining=R; used=U; total=T</budget>
The remaining is the number of searches you can still perform, used is the number of searches you have already performed, and total is the total budget allocated. Note that remaining + used = total.
You must consider the current budget state before deciding any action.

2. Reasoning Phase (`<think>`)
Before taking any action or providing an answer, you must output a reasoning block. Inside `<think>...</think>`, you are required to explicitly analyze:
Information Sufficiency: Is the current information adequate for producing a reliable answer? What specific information is still missing?
Budget Strategy: Analyze the current budget state (`<budget>`). Determine whether the next search is necessary and worthwhile given remaining resources. Prioritize using your internal knowledge when possible, and reserve search budget for information that is clearly beyond your parametric knowledge.

3. Search Phase (`<search>`)
If additional information is needed, invoke the search engine:
<search>query</search>
Cost: 1 unit of budget per search.
Usage: Use this when you need specific factual information that you cannot reliably produce from internal knowledge alone.

4. Observation Phase (`<information>`)
The system will return search results wrapped in `<information>...</information>` tags.

5. Answering Phase (`<answer>`)
Once you have gathered sufficient information and no further searching is required, provide the final concise answer wrapped in `<answer>...</answer>` tags. For example: <answer> Beijing </answer>

Now, answer the following question:
{Question}"""


PHASE_TWO_PROMPT = """You are an expert assistant designed to solve complex tasks through reasoning and search. Your goal is to answer the user's question accurately. You have a total of {B} searches available. You must execute a loop of Reasoning, Action, Observation until you have sufficient information to answer.

1. Reasoning Phase (`<think>`)
Before taking any action or providing an answer, you must output a reasoning block inside `<think>...</think>` tags.

2. Search Phase (`<search>`)
If additional information is needed, invoke the search engine:
<search>query</search>
Cost: 1 unit of budget per search.

3. Observation Phase (`<information>`)
The system will return search results wrapped in `<information>...</information>` tags.

4. Answering Phase (`<answer>`)
Once you have gathered sufficient information and no further searching is required, provide the final concise answer wrapped in `<answer>...</answer>` tags. For example: <answer> Beijing </answer>

Now, answer the following question:
{Question}"""

# Roman-numeral aliases mirror the terminology used by the method.
PHASE_I_PROMPT = PHASE_ONE_PROMPT
PHASE_II_PROMPT = PHASE_TWO_PROMPT


def _validate_prompt_inputs(question: str, budget: int) -> tuple[str, int]:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return normalized_question, budget


def _render(template: str, *, question: str, budget: int) -> str:
    normalized_question, normalized_budget = _validate_prompt_inputs(question, budget)
    # ``str.format`` is deliberately avoided: tag examples may later contain
    # braces that should remain verbatim.
    return template.replace("{B}", str(normalized_budget)).replace("{Question}", normalized_question)


def build_phase_one_prompt(question: str, budget: int) -> str:
    """Render the Phase I prompt with the full budget-aware scaffold."""

    return _render(PHASE_ONE_PROMPT, question=question, budget=budget)


def build_phase_two_prompt(question: str, budget: int) -> str:
    """Render the scaffold-free Phase II/inference prompt."""

    return _render(PHASE_TWO_PROMPT, question=question, budget=budget)


def build_prompt(question: str, budget: int, *, phase_one: bool) -> str:
    """Render the prompt selected by the training phase."""

    builder = build_phase_one_prompt if phase_one else build_phase_two_prompt
    return builder(question, budget)


def budget_tag(*, total: int, used: int) -> str:
    """Create the environment-injected Phase I budget observation."""

    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if isinstance(used, bool) or not isinstance(used, int):
        raise TypeError("used must be an integer")
    if total < 0:
        raise ValueError("total must be non-negative")
    if not 0 <= used <= total:
        raise ValueError("used must be between zero and total")
    return f"<budget>remaining={total - used}; used={used}; total={total}</budget>"
