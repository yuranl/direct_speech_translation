from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup
from translationData import translationDataset
from transformers import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
import torch
import numpy as np
#from sacrebleu.metrics import BLEU

split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/*.json', 'validation': './translation_validation/*.json'})
# translationTrain = translationDataset('./transcription_translation/')
# translationValidation = translationDataset('./translation_validation/')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    for i in range(len(decoded_labels)):
        if (decoded_labels[i] == ['']):
            decoded_labels[i] = ['This sentence was corrupted. Developer notes']
            print(labels[i])
    return decoded_preds, decoded_labels


def pretrained_performance():
    model.eval()
    loss_eval_cumu = 0
    i = 0
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            i += 1
            loss_eval_cumu += loss.item()

    loss_eval_cumu /= i
    print(f"Before training, Validation Loss: {loss_eval_cumu:.2f}")

    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    results = metric.compute()
    print(f"Before training, BLEU score: {results['score']:.2f}")

def preprocess_function(examples):
    inputs = [ex for ex in examples["transcript"]]
    targets = [ex for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
#print(model)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")

class AugmentedEncoder(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(AugmentedEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(100, 512)
        self.encoder = model.model.encoder

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        linear1_out = self.linear1(x)
        encoder_out = self.encoder(linear1_out)
        return encoder_out

# Create random Tensors to hold inputs and outputs
x = torch.randn(10, 100)
x.long()
y = torch.randn(10, 512)

# Construct our model by instantiating the class defined above
aug_model = AugmentedEncoder()

aug_criterion = torch.nn.MSELoss(reduction='sum')
aug_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = aug_model(x)

    # Compute and print loss
    loss = aug_criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    aug_optimizer.zero_grad()
    loss.backward()
    aug_optimizer.step()