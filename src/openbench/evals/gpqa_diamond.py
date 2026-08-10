import random
from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import SIMPLE_EVALS_SYSTEM_MESSAGE
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


# Options are shuffled deterministically PER QUESTION (seeded by the question
# text) rather than with a single global seed. A single global seed reshuffles
# the same fixed initial order identically for every record, pinning the correct
# answer to one letter across the dataset; per-question seeding keeps the shuffle
# reproducible while spreading the correct answer across A/B/C/D, matching
# OpenRouter's "shuffles by index" convention (see Auto Exacto docs). Note: the
# prompts are not reshuffled across epochs.
def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a GQPQA Diamond record to an openbench MCQSample."""
    rng = random.Random(record["Question"])
    options = [
        record["Correct Answer"],
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    rng.shuffle(options)
    # Get index of correct answer and convert to A, B, C, D
    correct_index = options.index(record["Correct Answer"])
    correct_letter = "ABCD"[correct_index]
    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["Question"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        ),
        target=correct_letter,
    )


@task
def gpqa_diamond() -> Task:
    """Evaluate the GQPQA Diamond dataset (MCQ Abstracted)."""
    return MCQEval(
        name="gpqa_diamond",
        dataset_path="nmayorga7/gpqa_diamond",
        record_to_mcq_sample=record_to_mcq_sample,
        split="train",  # only train split available
        auto_id=True,
        prompt_template=SIMPLE_EVALS_SYSTEM_MESSAGE,
        config=GenerateConfig(temperature=0.5),
        epochs=Epochs(10),
    )
