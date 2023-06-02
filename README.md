# Korean Safety Benchmarks

## Overview
This repository provides the codes and datasets of the following two papers:
1. [**SQuARe: A Large-Scale Dataset of Sensitive Questions and Acceptable Responses Created through Human-Machine Collaboration**](https://arxiv.org/abs/2305.17696)
    * Hwaran Lee*, Seokhee Hong*, Joonsuk Park, Takyoung Kim, Meeyoung Cha, Yejin Choi, Byoungpil Kim, Gunhee Kim, Eun-Ju Lee, Yong Lim, Alice Oh, Sangchul Park and Jung-Woo Ha
    * _ACL 2023_
2. [**KoSBi: A Dataset for Mitigating Social Bias Risks Towards Safer Large Language Model Applications**](https://arxiv.org/abs/2305.17701)
    * Hwaran Lee*, Seokhee Hong*, Joonsuk Park, Takyoung Kim, Gunhee Kim, and Jung-Woo Ha
    * _ACL 2023_

## SQuARe
### Dataset
Our *SQuARe* dataset can be found in `data/SQuARe/`. Please refer to *SQuARe* paper for the detail of the dataset.

We also release the dataset with the raw annotations in `data/SQuARe/with_raw_annotations`. Since questions and responses in our dataset are inherently subjective, we believe the raw annotations would help further research the disagreement between annotators.

### Data Generation Pipeline
**Note**: Though we've made our dataset include English-translated, cautions are needed when directly using it since the sensitive topics we used reflect the idiosyncrasies of Korean society. We recommend that researchers build their own dataset.

The pipeline for dataset generation can be found in `pipeline/square`.

## KoSBi
### Dataset
Our *KoSBi* dataset can be found in `data/KosBi/`. Please refer to *KoSBi* paper for the detail of the dataset.

**Update**: We collected more data by running an additional iteration. You can find them in the files named `data/KoSBi/kosbi_v2_{train,valid,test}.json`, which include the original KoSBi datasets. The total number of (*context*, *sentence*) pairs has increased to almost 68k, with 34.2k safe sentences and 33.8k unsafe sentences.

### Data Generation Pipeline
Similar to *SQuARe*, the pipeline for dataset generation can be found in `pipeline/kosbi`.

---

## License
```

```
## How to cite

```

```
```

```

## Contact
If you have any questions about our dataset or codes, feel free to ask us: Seokhee Hong (seokhee.hong@vision.snu.ac.kr) or Hwaran Lee (hwaran.lee@navercorp.com)