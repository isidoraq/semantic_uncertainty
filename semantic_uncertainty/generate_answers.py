"""Sample answers from LLMs on QA task."""

import gc
import os
import logging
import random
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import torch
import wandb
import logging
import sys
from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


# utils.setup_logger()

from datetime import datetime


def setup_logging(args):
    # Create a logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name based on the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{current_time}.log")

    # Set up logging to file only
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    logger = logging.getLogger(__name__)

    # Log the script arguments
    logger.info(f"Script started with arguments: {args}")

    return logger, log_file


import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main(args):

    # Setup run.
    if args.dataset == "svamp":
        if not args.use_context:
            logging.info("Forcing `use_context=True` for svamp dataset.")
            print("Forcing `use_context=True` for svamp dataset.")
            args.use_context = True
    elif args.dataset == "squad":
        if not args.answerable_only:
            logging.info("Forcing `answerable_only=True` for squad dataset.")
            print("Forcing `answerable_only=True` for squad dataset.")
            args.answerable_only = True

    experiment_details = {"args": args}
    random.seed(args.random_seed)
    user = os.environ["USER"]
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")
    wandb.init(
        entity=args.entity,
        project=(
            "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
        ),
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
    )
    logging.info("Finished wandb init.")
    print("Finished wandb init.")

    # Get accuracy metric.
    # metric, best_iou,  best_corr = utils.get_metric(args.metric)
    metric = utils.get_metric(args, args.metric)

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, seed=args.random_seed  # add_options=args.use_mc_options,
    )
    if args.ood_train_dataset is not None:
        logging.warning(
            "Using OOD dataset %s to construct few-shot prompts and train p_ik.",
            args.ood_train_dataset,
        )
        print("Using OOD dataset %s to construct few-shot prompts and train p_ik.")
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(
            args.ood_train_dataset, add_options=args.use_mc_options
        )
    if not isinstance(train_dataset, list):
        logging.info("Train dataset: %s", train_dataset)
        print("Train dataset: %s", train_dataset)
    # Handle the case where train_dataset is None
    if train_dataset is None:
        logger.info("No training dataset available. Skipping training data processing.")
        print("No training dataset available. Skipping training data processing.")
        answerable_indices = []
        unanswerable_indices = []
    else:
        answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
        logger.info(f"Answerable indices: {len(answerable_indices)}")
        print(f"Answerable indices: {len(answerable_indices)}")
        logger.info(f"Unanswerable indices: {len(unanswerable_indices)}")
        print(f"Unanswerable indices: {len(unanswerable_indices)}")

    # Get indices of answerable and unanswerable questions and construct prompt.
    # answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    # Adjust few-shot sampling
    if len(answerable_indices) < args.num_few_shot:
        logger.warning(
            f"Not enough examples for few-shot (requested {args.num_few_shot}, available {len(answerable_indices)}). Using all available examples."
        )
        print(
            f"Not enough examples for few-shot (requested {args.num_few_shot}, available {len(answerable_indices)}). Using all available examples."
        )
        prompt_indices = answerable_indices
    else:
        prompt_indices = random.sample(answerable_indices, args.num_few_shot)

    experiment_details["prompt_indices"] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt
    )
    experiment_details["prompt"] = prompt
    experiment_details["BRIEF"] = BRIEF
    logging.info("Prompt is: %s", prompt)
    print("Prompt is: %s", prompt)

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80 * "#")
        print(80 * "#")
        logging.info("Constructing few-shot prompt for p_true.")
        print("Constructing few-shot prompt for p_true.")

        # p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        # Adjust sampling for p_true_indices
        if len(answerable_indices) < args.p_true_num_fewshot:
            logger.warning(
                f"Not enough examples for p_true (requested {args.p_true_num_fewshot}, available {len(answerable_indices)}). Using all available examples."
            )
            print(
                f"Not enough examples for p_true (requested {args.p_true_num_fewshot}, available {len(answerable_indices)}). Using all available examples."
            )
            p_true_indices = answerable_indices
        else:
            p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)

        # Print a few examples
        for i in range(min(5, len(validation_dataset))):
            example = validation_dataset[i]
            logger.info(f"Example {i}:")
            logger.info(f"  Question: {example['question']}")
            logger.info(f"  Answer: {example['answers']['text']}")
            logger.info(
                f"  Context: {example['context'][:100]}..."
            )  # Print first 100 chars of context
            print(f"Example {i}:")
            print(f"  Question: {example['question']}")
            print(f"  Answer: {example['answers']['text']}")
            print(
                f"  Context: {example['context'][:100]}..."
            )  # Print first 100 chars of context

        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = (
            p_true_utils.construct_few_shot_prompt(
                model=model,
                dataset=train_dataset,
                indices=p_true_indices,
                prompt=prompt,
                brief=BRIEF,
                brief_always=args.brief_always and args.enable_brief,
                make_prompt=make_prompt,
                num_generations=args.num_generations,
                metric=metric,
            )
        )
        wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details["p_true_indices"] = p_true_indices
        experiment_details["p_true_responses"] = p_true_responses
        experiment_details["p_true_few_shot_prompt"] = p_true_few_shot_prompt
        logging.info("Finished constructing few-shot prompt for p_true.")
        logging.info(80 * "#")
        logging.info("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")
        print(80 * "#")
        print("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        print(80 * "#")
        print("Finished constructing few-shot prompt for p_true.")
        print(80 * "#")

    # Start answer generation.
    logging.info(80 * "=")
    logging.info("Generating answers: ")
    logging.info(80 * "=")
    print(80 * "=")
    print("Generating answers: ")
    print(80 * "=")
    for dataset_split in ["validation"]:
        logging.info(80 * "x")
        logging.info("Starting with dataset_split %s.", dataset_split)
        logging.info(80 * "x")
        print(80 * "x")
        print("Starting with dataset_split %s.", dataset_split)
        print(80 * "x")

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        ious, corrs = [], []

        if dataset_split == "train":
            if not args.get_training_set_generations:
                logging.info("Skip training data.")
                print("Skip training data.")
                continue
            dataset = train_dataset
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices)
            )

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {"indices": indices}

        if args.num_samples > len(dataset):
            logging.warning(
                "Not enough samples in dataset. Using all %d samples.", len(dataset)
            )
            print("Not enough samples in dataset. Using all %d samples.", len(dataset))
        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example["context"]
            generations[example["id"]] = {"question": question, "context": context}
            correct_answer = example["answers"]["text"]

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief
            )
            local_prompt = prompt + current_input

            logging.info("Current input: ".ljust(15) + current_input)
            print("Current input: ".ljust(15) + current_input)

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if (
                dataset_split == "train"
                and args.get_training_set_generations_most_likely_only
            ):
                num_generations = 1
            else:
                # num_generations = args.num_generations + 1
                num_generations = 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                # temperature = 0.1 if i == 0 else args.temperature
                temperature = 0

                # predicted_answer, token_log_likelihoods, embedding = model.predict(
                # local_prompt, temperature)
                # embedding = embedding.cpu() if embedding is not None else None
                try:
                    openai_api_key = os.environ["OPENAI_API_KEY"]
                    client = OpenAI(api_key=openai_api_key)
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
                        seed=args.random_seed,
                    )
                    response = completion.choices[0].message.content
                except Exception as e:
                    logging.error(f"Error generating answer: {e}")
                    print(f"Error generating answer: {e}")
                    response = ""

                predicted_answer = response

                # Only compute accuracy if question is answerable.
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                    iou, corr = utils.model_based_metric_iou_corr(
                        predicted_answer, example
                    )
                else:
                    acc = 0.0  # pylint: disable=invalid-name
                    iou, corr = utils.model_based_metric_iou_corr(
                        predicted_answer, example
                    )

                if i == 0:
                    logging.info("Iteration " + str(it) + ":  " + 80 * "#")
                    print("Iteration " + str(it) + ":  " + 80 * "#")
                    if args.use_context:
                        logging.info("context: ".ljust(15) + str(context))
                        print("context: ".ljust(15) + str(context))
                    logging.info("question: ".ljust(15) + question)
                    logging.info("low-t prediction: ".ljust(15) + predicted_answer)
                    logging.info("correct answer: ".ljust(15) + str(correct_answer))
                    logging.info("accuracy: ".ljust(15) + str(acc))
                    logging.info("iou: ".ljust(15) + str(iou))
                    logging.info("corr: ".ljust(15) + str(corr))
                    print("question: ".ljust(15) + question)
                    print("low-t prediction: ".ljust(15) + predicted_answer)
                    print("correct answer: ".ljust(15) + str(correct_answer))
                    print("accuracy: ".ljust(15) + str(acc))
                    print("iou: ".ljust(15) + str(iou))
                    print("corr: ".ljust(15) + str(corr))
                    print(80 * "#")

                    accuracies.append(acc)
                    ious.append(iou)
                    corrs.append(corr)
                    most_likely_answer_dict = {
                        "response": predicted_answer,
                        #'token_log_likelihoods': token_log_likelihoods,
                        #'embedding': embedding,
                        "accuracy": acc,
                        "iou": iou,
                        "corr": corr,
                    }
                    generations[example["id"]].update(
                        {
                            "most_likely_answer": most_likely_answer_dict,
                            "reference": utils.get_reference(example),
                        }
                    )

                else:
                    logging.info(
                        "high-t prediction ".ljust(15)
                        + str(i)
                        + " : "
                        + predicted_answer
                    )
                    print(
                        "high-t prediction ".ljust(15)
                        + str(i)
                        + " : "
                        + predicted_answer
                    )
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (
                            predicted_answer,
                            acc,
                            iou,
                            corr,
                        )  # token_log_likelihoods, embedding,
                    )

            # Append all predictions for this example to `generations`.
            generations[example["id"]]["responses"] = full_responses

            if args.compute_p_true and dataset_split == "validation":
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    args,
                    model,
                    question,
                    most_likely_answer_dict["response"],
                    [r[0] for r in full_responses],
                    p_true_few_shot_prompt,
                    hint=args.p_true_hint,
                )
                p_trues.append(p_true)
                logging.info("p_true: %s", p_true)
                print("p_true: %s", p_true)

        # Save generations for that split.
        utils.save(generations, f"{dataset_split}_generations.pkl")

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        mean_iou = np.mean(ious)
        mean_corr = np.mean(corrs)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        print(f"Overall {dataset_split} split iou: {mean_iou}")
        print(f"Overall {dataset_split} split corr: {mean_corr}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})
        wandb.log({f"{dataset_split}_iou": mean_iou})
        wandb.log({f"{dataset_split}_corr": mean_corr})

        if dataset_split == "validation":
            if args.compute_p_true:
                results_dict["uncertainty_measures"] = {
                    "p_false": [1 - p for p in p_trues],
                    "p_false_fixed": [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, "uncertainty_measures.pkl")

    utils.save(experiment_details, "experiment_details.pkl")
    logging.info("Run complete.")
    print("Run complete.")
    del model


if __name__ == "__main__":
    args = utils.get_parser().parse_args()
    logger, log_file = setup_logging(args)

    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_file, "a")
    sys.stderr = open(log_file, "a")

    try:
        logging.info("Starting new run with args: %s", args)
        print("Starting new run with args: %s", args)

        if args.compute_uncertainties:
            args.assign_new_wandb_id = False

        # First sample generations from LLM.
        logging.info("STARTING `generate_answers`!")
        print("STARTING `generate_answers`!")
        main(args)
        logging.info("FINISHED `generate_answers`!")
        print("FINISHED `generate_answers`!")

        if args.compute_uncertainties:
            # Follow with uncertainty calculation script by default.
            args.assign_new_wandb_id = False
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(50 * "#X")
            print(50 * "#X")
            # logging.info("STARTING `compute_uncertainty_measures`!")
            # print("STARTING `compute_uncertainty_measures`!")
            # main_compute(args)
            # logging.info("FINISHED `compute_uncertainty_measures`!")
            # print("FINISHED `compute_uncertainty_measures`!")

    except Exception as e:
        logger.exception("An error occurred during script execution:")
        print("An error occurred during script execution:")
    finally:
        logger.info("Script execution completed.")
        print("Script execution completed.")
        # Close the redirected stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
