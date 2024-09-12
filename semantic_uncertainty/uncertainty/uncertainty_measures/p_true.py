"""Compute p_true uncertainty metric."""

import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(".env")


def construct_few_shot_prompt(
    *,
    model,
    dataset,
    indices,
    prompt,
    brief,
    brief_always,
    make_prompt,
    num_generations,
    metric,
):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    it = -1  # Initialize it outside the loop

    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ["\n"]
        prompt_candidate += ["Question: " + question]
        prompt_candidate += ["\nBrainstormed Answers: "]
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info("P_TRUE >> Current Question: ".ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):
            temperature = 0.1 if j == 0 else 1.0

            if "gpt" in model:
                openai_api_key = os.environ["OPENAI_API_KEY"]
                client = OpenAI(api_key=openai_api_key)
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. Help me with my task!",
                            },
                            {"role": "user", "content": local_prompt},
                        ],
                        temperature=temperature,
                        seed=42,
                    )
                    response = completion.choices[0].message.content
                except Exception as e:
                    print(f"Error in OpenAI API call: {e}")
                    response = "Error: Failed to respond"

            logging.info("P_TRUE >> Current Response: ".ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f"{response.strip()} \n"]
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example["answers"]["text"]]
                logging.info(
                    "P_TRUE >> LOW-T >> true answer: ".ljust(35) + str(answers)
                )
                logging.info("P_TRUE >> LOW-T >> acc: ".ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses,
            most_likely_response=most_likely_response,
            is_correct=is_correct,
        )

        prompt_candidate += ["Possible answer: " + most_likely_response + "\n"]
        prompt_candidate += ["Is the possible answer:\n"]
        prompt_candidate += ["A) True\n"]
        prompt_candidate += ["B) False\n"]
        prompt_candidate += ["The possible answer is:"]
        prompt_candidate += [" A" if is_correct else " B"]

        prompt_len = len(few_shot_prompt + prompt_candidate)
        max_new_tokens = 128
        token_limit = 4096
        max_input_len = prompt_len + num_generations * max_new_tokens + 200

        if max_input_len < token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning("Cutting off p_true prompt at length %d.", it)
            break

    # If no examples were processed, add a placeholder
    if it == -1:
        few_shot_prompt.append("No examples available.\n\n")
        logging.warning("No examples were processed for p_true prompt.")

    return "".join(few_shot_prompt), all_responses, it + 1


def get_p_true(args, input_data, model):
    """Get the probability of the model answering A (True) for the given input."""

    input_data += " A"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Help me with my task!",
                },
                {"role": "user", "content": input_data},
            ],
            seed=args.random_seed,
            max_tokens=1,
            logprobs=True,
            top_p=1,
            temperature=0,
        )
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        result.append("Error: Failed to respond")

    # Get the log probability of the last token (which should be 'A')
    logprobs = completion.choices[0].logprobs.content[0].top_logprobs

    # Find the log probability for 'A' or ' A' (space + 'A')
    a_logprob = next((lp.logprob for lp in logprobs if lp.token in ["A", " A"]), None)

    if a_logprob is not None:
        p_true = exp(a_logprob)
    else:
        # If 'A' is not in the top logprobs, it's very unlikely
        p_true = 0.0

    return p_true


def calculate_p_true(
    args,
    model,
    question,
    most_probable_answer,
    brainstormed_answers,
    few_shot_prompt,
    hint=False,
):
    """Calculate p_true uncertainty metric."""

    if few_shot_prompt:
        prompt = few_shot_prompt + "\n"
    else:
        prompt = ""

    prompt += "Question: " + question
    prompt += "\nBrainstormed Answers: "
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + "\n"
    prompt += "Possible answer: " + most_probable_answer + "\n"
    if not hint:
        prompt += "Is the possible answer:\n"
        prompt += "A) True\n"
        prompt += "B) False\n"
        prompt += "The possible answer is:"
    else:
        prompt += "Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:"

    log_prob = get_p_true(args, prompt, model)

    return log_prob
