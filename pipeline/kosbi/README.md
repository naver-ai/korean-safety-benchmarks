# Pipeline for KoSBi

## Neutral Contexts Generation

### 1. Context Generation
Before generating contexts, you need:

- **HyperCLOVA API url** and **API key**
- Targeted demographic category and groups
  + used in KoSBi is in `pipeline/kosbi/kosbi_demographic_groups.json`
  + As mentioned in the paper, we generated dataset for **72** targeted demographic groups across **15** categories.
- Initial demonstration pool for prompting
  + Ours is in `pipeline/kosbi/demo/kosbi_context_iter0.json`

We used 10 demonstration samples of contexts when generating KoSBi. Among them, 5 demonstrations are from the pool with the same demographic category as what we are generating, and 3 are from the pool with the same demographic group. Of course, you can change these numbers in your generation pipeline.

Here is a code for generating contexts: 

```
python pipeline/kosbi/generate_contexts.py \
    --api_url {API_url} \
    --api_key {API_key} \
    --demo_fpath pipeline/kosbi/demo/kosbi_context_iter0.json \
    --num_total_demos 10 \
    --num_in_cat_demos 5 \
    --num_in_grp_demos 3 \
    --demographic_list pipeline/kosbi/kosbi_demographic_groups.json \
    --output_dir data/generations \
    --output_fname output_contexts.json
```

### 2. Filtering: Remove Contexts NOT Including Target Group
We need to include target group words in the contexts we are generating, so we should remove contexts that do not have them. To remove, you need a finetuned classifier trained to classify the contexts that don't include target group words.
If there is no classifier (e.g., at the first iteration), skip the argument
`--filter_model_ckpt`.

```
python pipeline/kosbi/filter_context.py \
    --input_fname data/generations/output_contexts.json \
    --filter_model_ckpt {filter_model.ckpt} \
    --output_dir data/generations \
    --output_fname filtered_contexts.json
```

### 3. Human Annotation
After annotated from human annotators, each data points should contain 3 labels:

- `include? = {0,1}` : whether the context contains target group
- `context_label` : the label of the context is one of `{safe,unsafe,undecided}`
  + `undecided` is assigned when annotators don't agree
- `context_sub_label` : if labeled `unsafe`, the context is further labeled as 
  1) `stereotype`
  2) `prejudice_or_discrimination`
  3) `other`
  4) `undefined` : when three annotators could not decide the label through major voting, but 2 or more chose one of the unsafe sub-labels.

We remove `include?` label from our KoSBi because all contexts include target group.

### 4. Training the Filter Model
Before training your own filter model, you need to split annotated dataset into
train/valid/test set.

Make sure that `--task` should include `kosbi_incl_grp_classify`.

```
python src/train.py \
    --task kosbi_incl_grp_classify \
    --wandb_project {wandb_prj} \
    --wandb_entity {wandb_entity} \
    --do_train_only True \
    --checkpoint_dirpath data/models/kosbi_context \
    --trn_pth {train.json} \
    --val_pth {valid.json} \
    --tst_pth {test.json} \
    --epoch {epoch} \
    --lr {lr} \
    --batch_size {bs} \
    --gradient_clip_val {gc_val}
```

### 5. Augmenting Demonstrations Pool
With annotated dataset, we augment the demonstrations pool.

Note that, at the first iteration, we *replace* the initial demonstrations with the annotated dataset, not augment it. (i.e., skipping `--prev_demo_fname`)

```
python pipeline/kosbi/augment_demo_context.py \
    --input_fname data/generations/filtered_contexts.json \
    --prev_demo_fname {prev_demo.json} \
    --output_dir pipeline/kosbi/demo \
    --output_demo_fname kosbi_context_iter1.json
```

## Un-/Safe Sentence Generation

### 1. (Next) Sentence Generation
Before generating sentences, you need:

- **HyperCLOVA API url** and **API key** (as same as in context generation)
- Annotated contexts dataset
- Initial demonstration pool for prompting
  + Ours is in `pipeline/kosbi/demo/kosbi_sentence_iter0.json`

We used 10 demonstration samples of sentences when generating KoSBi. Among them, 5 demonstrations are from the pool with the same context label as what we are generating. Of course, you can change these numbers in your generation pipeline.

Here is a code for generating sentences: 

```
python pipeline/kosbi/generate_sentence.py \
    --api_url {API_url} \
    --api_key {API_key} \
    --demo_fpath pipeline/kosbi/demo/kosbi_sentence_iter0.json \
    --num_total_demos 10 \
    --num_in_context_label 5 \
    --context_fpath data/generations/filtered_contexts.json \
    --output_dir data/generations \
    --output_fname output_sentences.json
```

### 2. Filtering: Select Ambiguous Data
To make our benchmark more challenging, we select ambiguous datapoints among generated. In the paper, we used 5 checkpoints and calculated estimated max variability.

If there are no classifiers (e.g., at the first iteration), skip the argument `--filter_model_ckpt_dir`.

```
python pipeline/kosbi/filter_sentence.py \
    --input_fname data/generations/output_sentences.json \
    --filter_model_ckpt_dir {filter_ckpt_dir} \
    --filter_model_train_epochs 5 \
    --output_dir data/generations \
    --output_fname filtered_sentences.json
```

### 3. Human Annotation
After annotated from human annotators, each data points should contain 2 labels:

- `sentence_label` : the label of the sentence is one of `{safe,unsafe}`
- `sentence_sub_label` : if labeled `unsafe`, the sentence is further labeled as 
  1) `stereotype_{explicit,implicit}`
  2) `prejudice_{explicit,implicit}`
  3) `discrimination_{explicit,implicit}`
  4) `other`
  5) `undefined` : when three annotators could not decide the label through major voting, but 2 or more chose one of the unsafe sub-labels.

You can see examples in our KoSBi dataset.

### 4. Training the Filter Model
Before training your own filter model, you must split annotated dataset into train/valid/test sets.

It is recommended to set `checkpoint_save_top_k` == `epoch` and `earlystop_patience` == `epoch` to save all checkpoints. These checkpoints will be used for variability calculation.

Make sure that `--task` should include `kosbi_unsafe_sent_classify`.

```
python src/train.py \
    --task kosbi_unsafe_sent_classify \
    --wandb_project {wandb_prj} \
    --wandb_entity {wandb_entity} \
    --do_train_only True \
    --checkpoint_dirpath data/models/kosbi_sentence \
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
With annotated dataset, we augment the demonstrations pool. To be used as a demonstration, each data should contain an `sentence_label`, an `context_label`, and an `context_sub_label` label.

As mentioned in the paper, we updated the demonstrations pool with the annotated dataset, to which all annotators agree. Therefore, raw annotations are also needed at this step. Examples of raw annotations can be found in `data/SQuARe/with_raw_annotations`. While `Q2: Acceptable or Non-acceptable` annotations are used in SQuARe, in KoSBi, we need (raw) annotations for `Q2: Safe or Unsafe`.

Note that, at the first iteration, we *replace* the initial demonstrations with the annotated dataset, not augment it. (i.e., skipping `--prev_demo_fname`)

```
python pipeline/kosbi/augment_demo_sentence.py \
    --input_fname data/generations/filtered_sentences.json \
    --prev_demo_fname {prev_demo.json} \
    --filter_model_ckpt_dir data/models/kosbi_sentence \
    --filter_model_train_epochs 5 \
    --output_dir pipeline/kosbi/demo \
    --output_demo_fname kosbi_sentence_iter1.json \
    --most_ambiguous_ratio 0.25
```