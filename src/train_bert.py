from __future__ import annotations

import logging
import argparse
from copy import deepcopy
from pathlib import Path
import yaml
import wandb
import torch

import torch.nn.functional as F

import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EvalPrediction, EarlyStoppingCallback, TrainerCallback
from transformers.utils import logging as hf_logging

from data_processing.dataset_loader import get_dataset
from metrics import calculate_metrics

hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', help='input folder', type=str, default="/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active")
parser.add_argument('--out_folder', help='output folder', type=str, default="models/pretrained/historical/active_rand")

opts = parser.parse_args()
print(opts)

out_folder = opts.out_folder

# Define your custom classification head
class CustomClassifier(torch.nn.Module):
    def __init__(self, original_classifier, num_additional_layers, hidden_dim, num_labels):
        super(CustomClassifier, self).__init__()
        self.original_classifier = original_classifier
        self.num_additional_layers = num_additional_layers
        self.hidden_dim = hidden_dim
        
        # Define additional layers
        self.additional_layers = torch.nn.ModuleList()
        for _ in range(num_additional_layers):
            self.additional_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer for classification
        self.classifier_presoft = torch.nn.Linear(hidden_dim, num_labels)
        self.classifier = torch.nn.Softmax(dim=1)

    def forward(self, input_ids):
        outputs = self.original_classifier(input_ids)
        pooled_output = outputs.pooled_output  # Get the pooled output from the base model
        
        # Apply additional layers
        for layer in self.additional_layers:
            pooled_output = torch.nn.functional.relu(layer(pooled_output))
        
        # Final classification layer
        raw_logits = self.classifier_presoft(pooled_output)
        logits = self.classifier(raw_logits)
        return nn.Sigmoid()(logits)




# Define the additional layers
class CustomHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CustomHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)  # You can adjust the dropout rate as needed

    def forward(self, x):
        x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.dropout(x)d
        return nn.Sigmoid()(x)




class SingleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        #self.fc2 = nn.Linear(256, output_size)
    def forward(self, x):
        hidden =  F.relu(self.fc1(x))
        return hidden
        #return self.fc2(hidden)
        
# Stacking 'n' neural networks
class StackedNN(nn.Module):
    def __init__(self, input_size, num_nets, output_size):
        super(StackedNN, self).__init__()
        self.nets = nn.ModuleList([SingleNN(input_size, output_size) for _ in range(num_nets)])

    def forward(self, x):
        outputs = [F.softmax(net(x), dim=1) for net in self.nets]
        ret = torch.stack(outputs, dim=1).squeeze(2)
        print(ret.shape)
        return ret


    
    
# Now you can use this modified model for sequence classification



def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_preds

    # Convert logits into probabilities between 0 and 1. Logits are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(torch.Tensor(logits))

    return calculate_metrics(predictions=predictions, labels=labels)


def load_config(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config["learning_rate"] = float(config["learning_rate"])
    config["per_device_train_batch_size"] = int(config["per_device_train_batch_size"])
    config["per_device_eval_batch_size"] = int(config["per_device_eval_batch_size"])
    config["num_train_epochs"] = int(config["num_train_epochs"])
    config["weight_decay"] = float(config["weight_decay"])

    return config


class TrainingMetricsCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")

            return control_copy


def train(data_dir: Path, model_name_or_path: str, output_dir: Path, config_file: Path,
          tokenizer_path: str | None = None):
    """For local models, the tokenizer paths needs to be given."""
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print('loading data from')
    print(data_dir)
    _, encoded_dataset, id2label, label2id = get_dataset(data_dir, tokenizer)
    
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(id2label),
                                                               id2label=id2label,
                                                               label2id=label2id)
                                                                model.classifier = nn.Sequential(
                        nn.Linear(input_size = 768, output_size = 256),
                        nn.ReLU(),
                        nn.Linear(input_size = 256, output_size = len(id2label),
                        F.Softmax(dim = 1)
                                 )
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(id2label),
                                                               id2label=id2label,
                                                               label2id=label2id)
    
   
    '''
    StackedNN(input_size = 768, num_nets = len(id2label), output_size = 1)
    
    print(model.classifier)
    
    print(model)
    
    model.classifier = CustomHead(input_dim = 768, hidden_dim = len(id2label), num_classes = len(id2label))
    
    print(model.classifier)
    

    model.classifier = CustomClassifier(model.classifier, num_additional_layers=2, hidden_dim=512, num_labels = len(id2label)) 
    '''
    
    config = load_config(config_file)
    config["output_dir"] = output_dir

    train_args = TrainingArguments(**config)

    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=100)],
    )
    trainer.add_callback(TrainingMetricsCallback(trainer))

    logger.info(f"Training/evaluation parameters {train_args}")

    # Training
    if train_args.do_train:
        trainer.train()

    # Evaluation
    if train_args.do_eval:
        results = {}
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = Path(train_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info(f"{key} = {value}")
                writer.write(f"{key} = {value}\n")

            results.update(result)


def main():

    #wandb.init(mode="disabled")
    wandb.init(project="AI4-resident-education")
    

    opts = parser.parse_args()
    
    inp_path = opts.in_folder

    
    data_dir = Path(inp_path).resolve()
    data_dir = Path("data/splits/").resolve()
    config_file = Path("experiments/exp_0001.yaml").resolve()

    # Model from the HUB
    '''
    model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer_path = None
    output_dir = Path("models/finetuned/Bio_ClinicalBERT-finetuned")
    '''

    model_name_or_path = "/gpfs/data/chopralab/ad6489/pathology-extractor-bert/models/pretrained/nyutron_small-checkpoint-736000"
    tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/src/tmp/nyutron-big"
    #output_dir = Path("/gpfs/data/chopralab/ad6489/pathology-extractor-bert/models/finetuned/nyutron_small-finetuned/new_runs/config1-frozenbert")

    # Local model
    #smodel_name_or_path = "/gpfs/home/lt2504/pathology-extractor-bert/tmp/test-mlm/epoch_28/"
    #model_name_or_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/finetunted/"
    #tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/tmp/test-mlm/epoch_28/"
    output_dir = Path(out_folder)

    train(data_dir, model_name_or_path, output_dir, config_file, tokenizer_path)


if __name__ == "__main__":
    main()
