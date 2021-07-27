## Distillation Pipeline

A general distillation pipeline which could easily plugin a teacher that has to be distilled to a student. Currently only supports models from the Huggingface transformer library for the Language Modelling Problem.

## File details
- `train.py` : entry file called from training the student
- `utils.py` : contains utility functions used by the trainer
- `distiller.py` : main file holding the distillation model structure
- `preprocess_data.py`, `grouped_batch_sampler.py`, `lm_seqs_dataset` : files for preprocessing the raw input into binarized and tokenized text called by the `train.py` file.

## How to Train Distillation Models

A specific example of using the code for training a distillation for the masked language modelling task.
```
pip3 install -r requirements.txt
python3 train.py \
    --student_name distilroberta-base \
    --teacher_name roberta-base \
    --teacher_pretrained trained_Roberta_checkpoint \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --dump_path output/train1 \
    --data_file data/dump.txt \
    --force --n_gpu 0
```
The dump path argument holds the following files after training:
 - `checkpoint.pth` : The latest saved checkpoint for the model
 - `model_epoch_<epoch_no>.pth` : Checkpoint saved after the completion of the respective epoch
 - `config.json` : Configuration file for your student
 - `parmeters.json` : Parameters used for distillation
 - `logs` : Directory holding the tensorboard logs
 - `<teacher_name>.pickle`* : Binarized data file created using the input file
 - `<teacher_name>.token_counts.pickle`* : Token Counts for the input file used in the MLM smoothing task

*These files take time to be created during the first run, hence it is advisable to save them in a different directory and use them directly in the following runs by using arguments `preprocessed_data_file` and `preprocessed_token_counts`.

For any help in understanding the arguments needed by the train.py file use:
```
python3 train.py -h
```

## Loading and Using the trained student model
In the dump path location `config.json` and `checkpoint.pth` will be used to load then student. Run the following commands to get the student loaded:
```
from transformers inport AutoConfig, AutoModelForMaskedLM

stu_config = AutoConfig.from_pretrained('config.json')
stu_config.output_hidden_states = True
student_model = AutoModelForMaskedLM.from_config(stu_config)
student_model.load_state_dict(torch.load("checkpoint.pth", map_location = device))
```
Here `device` refers to the device either cpu or gpu where this model needs to get loaded.

## Experiments 
(https://drive.google.com/drive/folders/12PmvrGTB6WWjridWUsDbI6JV7Tj3SDSs?usp=sharing)<br>
This repository was used to distill the knowledge of RoBERTa and XLM-RoBERTa fine tuned on twitter datasets to the DistilRoBERTa student model. The different models distilled are:
 - RoBERTa finetuned on English Tweets
 - XLM-RoBERTa finetuned on Hindi Tweets
 - XLM-RoBERTa finetuned on Latin Tweets

Some Empirical results where the distilled model is able to capture twitter specific lingo better than the base RoBERTa model is given as follows, where the words define the closest cosine distanced words in the respective models:

| Word name | fine-tuned-model| base-model |
| :---: | :----: |  :---: |
| ma |my (0.99755), anna (0.9975)|son (0.99207), ji (0.99207)|
|af |f\*\*k (0.99625), ma (0.99607), |if (0.99241), ash (0.99198)|
|lmao|Lmao (0.99693), Lmaoooo (0.99401)|Lmao (0.94968), lady (0.94861)|
|Yeah|Okay (0.99899), Yo (0.99889)|Yes (0.98318), yeah (0.97725)|

## Acknowledgement
This project was part of Sprinklr Inc. ML internship 2021.<br>
Adapted in part from HuggingFace DistilBert training model (https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation)<br> 
**Author :** Mayank Musaddi (mayankmusaddi1997@gmail.com)<br>
**Mentor :** Ratnesh Jamidar (ratnesh.jamidar@sprinklr.com)<br>
