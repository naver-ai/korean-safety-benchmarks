# Pipeline for SQuARe

## Sensitive Questions Generation

### 1. Question Generation
Before generating questions, you need:

- **HyperCLOVA API url** and **API key**
- Sensitive topic dataset reflecting your own culture or society
  + Topics we used are in `pipeline/square/square_topic.json`
- Initial demonstration pool for prompting
  + Ours is in `pipeline/square/demo/square_question_iter0.json`

Here is a code for generating questions: 

```
python pipeline/square/generate_question.py \
    --api_url {API_url} \
    --api_key {API_key} \
    --demo_fpath pipeline/square/demo/square_question_iter0.json \
    --demo_pool_category {contentious,ethical,predictive} \
    --topic_fpath pipeline/square/square_topic.json \
    --output_dir data/generations \
    --output_fname output_questions.json
```

### 2. Filtering: Remove Objective Questions
To remove objective questions, you need a finetuned classifier, as mentioned in the paper.
If there is no classifier (e.g., at the first iteration), skip the argument
`--filter_model_ckpt`.

```
python pipeline/square/filter_question.py \
    --input_fname data/generations/output_questions.json \
    --filter_model_ckpt {filter_model.ckpt} \
    --output_dir data/generations \
    --output_fname filtered_questions.json
```

### 3. Human Annotation
After annotated from human annotators, each data points should contain 3 labels:

- `subjective? = {0,1}` : whether the question is subjective
- `sensitive? = {0,1}` : whether the question is sensitive
- `category` : category of sensitive question

You can see `sensitive?` and `category` in our *SQuARe* dataset.
We remove `subjective?` label from our dataset because all questions are subjective.

### 4. Training the Filter Model
Before training your own filter model, you need to split annotated dataset into
train/valid/test set.

Make sure that `--task` should include `square_subj_q_classify`.

```
python src/train.py \
    --task square_subj_q_classify \
    --wandb_project {wandb_prj} \
    --wandb_entity {wandb_entity} \
    --do_train_only True \
    --checkpoint_dirpath data/models/square_question \
    --trn_pth {train.json} \
    --val_pth {valid.json} \
    --tst_pth {test.json} \
    --epoch {epoch} \
    --lr {lr} \
    --batch_size {bs} \
    --gradient_clip_val {gc_val}
```

### 5. Augmenting Demonstrations Pool
With annotated dataset, we augment the demonstrations pool. To be used as a demonstration,
each data should contain `sensitive?` and `category` labels.

Note that, at the first iteration, we *replace* the initial demonstrations with the annotated dataset, not augment it. (i.e., skipping `--prev_demo_fname`)

```
python pipeline/square/augment_demo_question.py \
    --input_fname data/generations/filtered_questions.json \
    --prev_demo_fname {prev_demo.json} \
    --output_dir pipeline/square/demo \
    --output_demo_fname square_question_iter1.json
```

## Non-/Acceptable Responses Generation

### 1. Response Generation
Before generating responses, you need:

- **HyperCLOVA API url** and **API key** (as same as in question generation)
- Annotated questions dataset
- Initial demonstration pool for prompting
  + Ours is in `pipeline/square/demo/square_response_iter0.json`

Here is a code for generating responses: 

```
python pipeline/square/generate_response.py \
    --api_url {API_url} \
    --api_key {API_key} \
    --demo_fpath pipeline/square/demo/square_response_iter0.json \
    --question_fpath data/generations/filtered_questions.json \
    --output_dir data/generations \
    --output_fname output_responses.json
```

### 2. Filtering: Select Ambiguous Data
To make our benchmark more challenging, we select ambiguous datapoints among generated. In the paper, we used 5 checkpoints and calculated estimated max variability.

If there are no classifiers (e.g., at the first iteration), skip the argument `--filter_model_ckpt_dir`.

```
python pipeline/square/filter_response.py \
    --input_fname data/generations/output_responses.json \
    --filter_model_ckpt_dir {filter_ckpt_dir} \
    --filter_model_train_epochs 5 \
    --output_dir data/generations \
    --output_fname filtered_responses.json
```

### 3. Human Annotation
After annotated from human annotators, each data points should contain 2 labels:

- `acceptable? = {0,1}` : whether the response is acceptable
- `category` : category of non-/acceptable response

You can see examples in our *SQuARe* dataset.

### 4. Training the Filter Model
Before training your own filter model, you must split annotated dataset into train/valid/test sets.

It is recommended to set `checkpoint_save_top_k` == `epoch` and `earlystop_patience` == `epoch` to save all checkpoints. These checkpoints will be used for variability calculation.

Make sure that `--task` should include `square_accept_r_classify`.

```
python src/train.py \
    --task square_accept_r_classify \
    --wandb_project {wandb_prj} \
    --wandb_entity {wandb_entity} \
    --do_train_only True \
    --checkpoint_dirpath data/models/square_response \
    --trn_pth {train.json} \
    --val_pth {valid.json} \
    --tst_pth {test.json} \
    --epoch {epoch} \
    --lr {lr} \
    --batch_size {bs} \
    --gradient_clip_val {gc_val} \
    --checkpoint_save_top_k {epoch} \
    --earlystop_patience {epoch}
```

### 5. Augmenting Demonstrations Pool
With annotated dataset, we augment the demonstrations pool. To be used as a demonstration, each data should contain an `acceptable?` label.

As mentioned in the paper, we updated the demonstrations pool with the annotated dataset, to which all annotators agree. Therefore, raw annotations are also needed at this step. Examples of raw annotations can be found in `data/SQuARe/with_raw_annotations`.

Note that, at the first iteration, we *replace* the initial demonstrations with the annotated dataset, not augment it. (i.e., skipping `--prev_demo_fname`)

```
python pipeline/square/augment_demo_response.py \
    --input_fname data/generations/filtered_responses.json \
    --prev_demo_fname {prev_demo.json} \
    --filter_model_ckpt_dir data/models/square_response \
    --filter_model_train_epochs 5 \
    --output_dir pipeline/square/demo \
    --output_demo_fname square_response_iter1.json \
    --most_ambiguous_ratio 0.25
```