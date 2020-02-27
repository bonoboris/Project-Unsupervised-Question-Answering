
import os
import random
import timeit
# import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    CamembertConfig,
    CamembertForQuestionAnswering,
    CamembertTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

ALL_MODELS = tuple(CamembertConfig.pretrained_config_archive_map.keys())
# ('camembert-base', 'umberto-commoncrawl-cased-v1', 'umberto-wikipedia-uncased-v1')

MODEL_CLASS = (CamembertConfig, CamembertForQuestionAnswering, CamembertTokenizer)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "camembert"
MODEL_NAME_OR_PATH = ALL_MODELS[0]
OUTPUT_DIR = "./camembert_model_output/"
DATA_DIR = "./data/"
TRAIN_FILE = "camembert_train_set.json"
PREDICT_FILE = "camembert_eval_set.json"
DO_TRAIN = True
DO_EVAL = True
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0
LEARNING_RATE = 5e-5
ADAM_EPSILON = 1e-8
LOGGING_STEPS = 500
SAVE_STEPS = 500
WITH_NEGATIVE = False
MAX_SEQ_LENGTH = 384  # "The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded."
DOC_STRIDE = 128  # When splitting up a long document into chunks, how much stride to take between chunks.
MAX_QUERY_LENGTH = 64  # The maximum number of tokens for the question.
# Questions longer than this will be truncated to this length.


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(_train_dataset, _model, _tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    print("Load train data set")
    train_sampler = RandomSampler(_train_dataset)
    train_dataloader = DataLoader(_train_dataset, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

    t_total = len(train_dataloader) // NUM_TRAIN_EPOCHS

    print("Prepare optimizer")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in _model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {"params": [p for n, p in _model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )

    # Train!
    print("***** Running training *****")
    print("Num examples = %d" % (len(_train_dataset)))
    print("Num Epochs = %d" % NUM_TRAIN_EPOCHS)
    print("batch size = %d" % TRAIN_BATCH_SIZE)
    print("Total optimization steps = %d" % t_total)

    _global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    _tr_loss, logging_loss = 0.0, 0.0
    _model.zero_grad()
    train_iterator = trange(epochs_trained, NUM_TRAIN_EPOCHS, desc="Epoch", disable=False)
    # Added here for reproductibility
    set_seed()

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            _model.train()
            batch = tuple(t.to(DEVICE) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = _model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()

            _tr_loss += loss.item()
            if step + 1 == 0:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), 1)

                optimizer.step()
                _model.zero_grad()
                _global_step += 1

                # Log metrics
                if _global_step % LOGGING_STEPS == 0:
                    results = evaluate(_model, _tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, _global_step)
                        print("{0} : {1}".format(key, (_tr_loss - logging_loss) / LOGGING_STEPS))
                    tb_writer.add_scalar("loss", (_tr_loss - logging_loss) / LOGGING_STEPS, _global_step)

                    logging_loss = _tr_loss

                # Save model checkpoint
                if _global_step % SAVE_STEPS == 0:
                    output_dir = os.path.join(OUTPUT_DIR, "checkpoint-{}".format(_global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    _model.save_pretrained(output_dir)
                    _tokenizer.save_pretrained(output_dir)

                    print("Saving model checkpoint to %s" % output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    print("Saving optimizer and scheduler states to %s" % output_dir)

    tb_writer.close()

    return _global_step, _tr_loss / _global_step


def evaluate(model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(tokenizer, do_evaluate=True, output_examples=True)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len(dataset))

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }
            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            start_logits, end_logits = [to_list(output[i]) for output in outputs]
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print("  Evaluation done in total %f secs (%f sec per example)" % (evalTime, evalTime / len(dataset)))

    # Compute predictions
    output_prediction_file = os.path.join(OUTPUT_DIR, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(OUTPUT_DIR, "nbest_predictions_{}.json".format(prefix))

    if WITH_NEGATIVE:
        output_null_log_odds_file = os.path.join(OUTPUT_DIR, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(examples, features, all_results, n_best_size=20,
                                             max_answer_length=30, do_lower_case=True,
                                             output_prediction_file=output_prediction_file,
                                             output_nbest_file=output_nbest_file,
                                             output_null_log_odds_file=output_null_log_odds_file,
                                             version_2_with_negative=WITH_NEGATIVE, null_score_diff_threshold=0,
                                             tokenizer=tokenizer, verbose_logging=False)

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(tokenizer, do_evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = DATA_DIR
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if do_evaluate else "train",
            list(filter(None, MODEL_NAME_OR_PATH.split("/"))).pop(),
            str(MAX_SEQ_LENGTH),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s" % cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print("Creating features from dataset file at %s" % input_dir)
        if not DATA_DIR and ((do_evaluate and not PREDICT_FILE) or (not do_evaluate and not TRAIN_FILE)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if WITH_NEGATIVE:
                print("tensorflow_datasets does not handle SQuAD with plausible answers.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=do_evaluate)
        else:
            processor = SquadV2Processor() if WITH_NEGATIVE else SquadV1Processor()
            if do_evaluate:
                examples = processor.get_dev_examples(DATA_DIR, filename=PREDICT_FILE)
            else:
                examples = processor.get_train_examples(DATA_DIR, filename=TRAIN_FILE)

        features, dataset = squad_convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                                               max_seq_length=MAX_SEQ_LENGTH,
                                                               doc_stride=DOC_STRIDE,
                                                               max_query_length=MAX_QUERY_LENGTH,
                                                               is_training=not do_evaluate,
                                                               return_dataset="pt", threads=1,
                                                               )
        print("Saving features into cached file %s" % cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    if DOC_STRIDE >= MAX_SEQ_LENGTH - MAX_QUERY_LENGTH:
        print(
            "WARNING - You've set a doc stride which may be superior to the document length in some " +
            "examples. This could result in errors when building features from the examples. Please reduce the doc " +
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    if (
            os.path.exists(OUTPUT_DIR)
            and os.listdir(OUTPUT_DIR)
            and DO_TRAIN
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                OUTPUT_DIR
            )
        )

    set_seed()
    config_class, model_class, tokenizer_class = MODEL_CLASS
    config = config_class.from_pretrained(MODEL_NAME_OR_PATH, cache_dir=None)
    tokenizer = tokenizer_class.from_pretrained(MODEL_NAME_OR_PATH, do_lower_case=True, cache_dir=None)
    model = model_class.from_pretrained(MODEL_NAME_OR_PATH, from_tf=False, config=config, cache_dir=None)
    model.to(DEVICE)

    print("Training/evaluation parameters: ", )

    # Training
    if DO_TRAIN:
        train_dataset = load_and_cache_examples(tokenizer, do_evaluate=False, output_examples=False)
        global_step, tr_loss = train(train_dataset, model, tokenizer)
        print(" global_step = %s, average loss = %s" % (global_step, tr_loss))

    # Save the trained model and the tokenizer
    if DO_TRAIN:
        # Create output directory if needed
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        print("Saving model to %s" % OUTPUT_DIR)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(OUTPUT_DIR)  # , force_download=True)
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=True)
        model.to(DEVICE)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if DO_EVAL:
        if DO_TRAIN:
            print("Loading checkpoints saved during training for evaluation")
            checkpoints = [OUTPUT_DIR]
            # if args.eval_all_checkpoints:
            #     checkpoints = list(
            #         os.path.dirname(c)
            #         for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            #     )
        else:
            print("Loading checkpoint %s for evaluation" % MODEL_NAME_OR_PATH)
            checkpoints = [MODEL_NAME_OR_PATH]

        print("Evaluate the following checkpoints: %s" % checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(DEVICE)

            # Evaluate
            result = evaluate(model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    print("Results: {}".format(results))


if __name__ == '__main__':
    main()
