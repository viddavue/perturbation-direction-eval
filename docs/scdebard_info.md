# scDEBART — Technical Summary for a Coding Agent

**Paper:** *scDEBART: Predicting in silico Single-Cell Perturbation Responses via Large-Scale Differential Expression Learning*  
**Authors:** Jieun Sung and Wankyu Kim (ICML 2026)

**Official resources:**
- [GitHub repository](https://github.com/Jieun-Sung/scDEBART)
- [Processed datasets on Hugging Face](https://huggingface.co/datasets/Jieun-S/scDEBART)

## 1. Executive summary

scDEBART is a transformer-based model designed to predict how gene expression changes when one or more genes are experimentally perturbed.

Its primary innovation is to **predict differential gene expression directly rather than predicting absolute expression levels**.

Traditional approaches typically predict the expression profile of a perturbed cell and then compare that prediction against an unperturbed reference. scDEBART instead outputs a vector of predicted log2 fold changes (logFC).

The model uses a BART encoder–decoder architecture pretrained on approximately 6.28 million differential-expression profiles derived from 66.6 million single cells.

During pretraining, it learns relationships between gene-expression changes across different cellular populations. During fine-tuning, it learns to associate specific genetic perturbations with transcriptome-wide responses.

### Key ideas

1. **Direct logFC prediction:** Predict gene-expression changes instead of absolute expression.
2. **scVI denoising:** Use probabilistic estimates of differential expression to reduce dropout-related noise.
3. **Biological gene embeddings:** Combine BioBERT embeddings, pathway membership, and baseline expression.
4. **Masked differential-expression pretraining:** Randomly hide 15% of logFC values and reconstruct them.
5. **Perturbation-conditioned fine-tuning:** Replace the logFC input with a perturbation indicator identifying which genes are inhibited or activated.

### Reported performance

Across five Perturb-seq datasets, scDEBART achieves a mean top-50 DEG enrichment factor of 11.96, compared with 1.74–2.99 for scGPT and GEARS.

The model also achieves 71.4% top-1 accuracy in a reverse perturbation identification experiment, although that experiment contains only seven test examples.

The paper's main finding is that **learning patterns of expression change during pretraining is particularly effective for downstream perturbation prediction**.

Importantly, these are model predictions of population-level gene-expression responses, not simulations of individual cells or mechanistic causal models. The pretraining comparisons are observational rather than experimental interventions.

Source: Sections 1–4 and Figure 1 of the paper. Sung and Kim - 2026 - scDEBART …

## 2. Model inputs and outputs

The central prediction task is:

**Given the baseline expression of a cell population and the identity of a perturbed gene, predict the log2 fold change of every modeled gene.**

### Input

For each gene \(g_i\), the model receives:

| Feature | Description | Dimensions |
|---|---|---|
| Gene ID | Identity of the gene | Integer / identifier |
| BioBERT embedding | Functional description of the gene | 768 |
| MSigDB pathways | Binary vector of pathway memberships | 7,411 |
| Baseline expression | scVI-denoised expression in unperturbed cells | 1 |
| Perturbation indicator | Encodes inhibition, activation, or no perturbation | 1 |

The perturbation indicator is:

\[
p_i =
\begin{cases}
-10 & \text{if gene }i\text{ is inhibited}\\
+10 & \text{if gene }i\text{ is activated}\\
0 & \text{otherwise}
\end{cases}
\]

For multi-gene perturbations, multiple genes receive nonzero indicators.

### Output

The model outputs:

\[
\hat{x}=[\hat{x}_1,\hat{x}_2,\ldots,\hat{x}_N]
\]

where each \(\hat{x}_i\) is the predicted log2 fold change of gene \(i\).

A positive value indicates upregulation; a negative value indicates downregulation.

The model uses up to **5,000 highly variable genes during fine-tuning**, whereas pretraining profiles contain at most 1,024 genes.

The output is not a reconstructed expression matrix. It is a vector of predicted differential-expression effects.

---

## 3. Architecture

scDEBART uses a BART-style encoder–decoder transformer.

### Gene embedding construction

Each gene's biological and expression features are projected into a shared 768-dimensional embedding space.

First, pathway information is projected:

\[
e_{\text{pathway}} = W_r r_i
\]

The pathway embedding is concatenated with the BioBERT embedding:

\[
e_{\text{knowledge}} =
W_k[e_{\text{pathway}};b_i]
\]

The numerical features are independently projected:

\[
e_{\text{perturbation}} = W_{\text{logFC}}p_i
\]

\[
e_{\text{expression}} =
W_{\text{expr}}\log(1+\mu_i)
\]

The final representation is:

\[
h_i=W_p[
e_{\text{knowledge}};
e_{\text{perturbation}};
e_{\text{expression}}
]
\]

All gene representations have dimension 768.

During pretraining, the perturbation indicator \(p_i\) is replaced with the observed logFC \(x_i\), subject to masking.

### Transformer configuration

| Component | Configuration |
|---|---|
| Architecture | BART encoder–decoder |
| Embedding dimension | 768 |
| Encoder layers | 8 |
| Decoder layers | 4 |
| Attention heads | 12 |
| Head dimension | 64 |
| Feedforward dimension | 3,072 |
| Activation | GELU |
| Dropout | 0.2 |
| Encoder attention | Bidirectional |
| Decoder self-attention | Causal |
| Decoder cross-attention | Encoder outputs |
| Normalization | Pre-layer normalization |

The model supports Flash Attention through PyTorch's scaled dot-product attention implementation.

### Prediction heads

Two MLP heads are applied sequentially to the decoder outputs.

**Embedding reconstruction head:**

```text
Decoder output [768]
        |
     Linear
   768 → 1536
        |
       GELU
        |
     Linear
   1536 → 768
        |
Reconstructed embedding [768]
```

**LogFC prediction head:**

```text
Reconstructed embedding [768]
        |
     Linear
    768 → 512
        |
       GELU
        |
     Linear
     512 → 1
        |
 Predicted logFC
```

The same heads are used for pretraining and fine-tuning.

A particularly important implementation detail is that **the decoder uses causal self-attention even though the encoder uses bidirectional attention**. Replacing this with an encoder-only transformer would change the proposed architecture.

Source: Section 3.2 and Appendix A.3, pages 3 and 19–20.

---

## 4. Pretraining pipeline

Pretraining is performed on differential-expression profiles generated from naturally occurring differences between cell populations.

It does not require genetic perturbation labels.

### Step 1: Collect and preprocess single-cell data

The authors use CELLxGENE datasets, retaining:

- Human datasets containing healthy cells.
- Datasets with 10,000–1,000,000 cells.
- Protein-coding genes.
- Cells passing dataset-specific quality-control thresholds.

The resulting corpus includes 617 datasets and approximately 66.6 million cells.

Expression is normalized, log-transformed, and reduced to highly variable genes for clustering.

### Step 2: Identify cell populations

Cells are clustered using SECUER.

The authors identify 98,753 clusters, with a median size of 278 cells.

For each dataset, they train scVI with 64 latent dimensions.

Cluster centroids are calculated in scVI latent space.

### Step 3: Select cluster comparisons

For each pair of clusters, compute cosine similarity between their latent centroids.

Keep pairs whose similarities lie between the 10th and 90th percentiles.

This excludes nearly identical clusters and extremely different clusters.

At most 100,000 eligible pairs are sampled per dataset.

### Step 4: Generate differential-expression profiles

For each selected cluster pair, use scVI posterior sampling to estimate differential expression.

The resulting values include:

```text
gene_id
baseline_expression
log2_fold_change
probability_of_differential_expression
detection_rate_baseline
detection_rate_comparison
```

The logFC estimate is the posterior median of sampled log2 expression ratios.

The authors use 4,000 posterior samples and a small numerical offset to avoid division by zero.

### Step 5: Filter genes

Keep genes satisfying:

| Criterion | Threshold |
|---|---|
| Detection rate in both populations | ≥ 0.3 |
| Probability of differential expression | ≥ 0.8 |
| Absolute fold change | ≥ 1.5 |
| Minimum genes per profile | 10 |
| Maximum genes per profile | 1,024 |

Genes are ranked by absolute logFC, and profiles are truncated to 1,024 genes.

The final pretraining corpus contains:

**6,282,796 differential-expression profiles covering 17,944 genes.**

### Step 6: Mask and reconstruct logFC

For each training profile:

1. Randomly select 15% of non-padding gene positions.
2. Set their input logFC to zero.
3. Keep their BioBERT, pathway, and baseline-expression features intact.
4. Run the masked profile through the BART model.
5. Predict the missing logFC values.
6. Reconstruct the original gene embeddings.
7. Calculate the loss only at masked positions.

The pretraining objective is:

\[
L_{\text{pretrain}}
=
0.8L_{\text{logFC}}+0.2L_{\text{embedding}}
\]

Both components use mean squared error:

\[
L_{\text{logFC}}
=
\frac{1}{|M|}
\sum_{i\in M}(x_i-\hat{x}_i)^2
\]

\[
L_{\text{embedding}}
=
\frac{1}{|M|}
\sum_{i\in M}\|h_i-\hat{h}_i\|_2^2
\]

Here, \(h_i\) is the original, unmasked input embedding.

This is the core self-supervised learning objective.

### Pretraining configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-3 |
| Batch size | 340 total |
| Maximum epochs | 50 |
| Early stopping patience | 5 |
| Stopping epoch | 22 |
| Warmup steps | 5,000 |
| Gradient clipping | 0.5 |
| Mixed precision | bfloat16 |
| Distributed training | DeepSpeed ZeRO-3 |

The authors report approximately 113 hours of training on four NVIDIA H100 80GB GPUs.

A reproduction should prioritize loading the released pretrained weights rather than repeating the full pretraining procedure unless specifically required.

---

## 5. Fine-tuning on perturbation data

After pretraining, the model is fine-tuned on experimental Perturb-seq data.

The paper evaluates five datasets:

| Dataset | Cell line | Perturbation |
|---|---|---|
| Norman | K562 | CRISPRa |
| Replogle | K562 | CRISPRi |
| Replogle | RPE1 | CRISPRi |
| Nadig | HepG2 | CRISPRi |
| Nadig | Jurkat | CRISPRi |

### Input transformation

During pretraining, each gene's numerical input represents observed logFC.

During fine-tuning, this feature is repurposed to encode the experimental perturbation.

For example, when inhibiting gene `TP53`:

```python
perturbation = {
    gene: 0.0 for gene in gene_ids
}

perturbation["TP53"] = -10.0
```

For activation:

```python
perturbation["TP53"] = 10.0
```

For simultaneous inhibition of multiple genes:

```python
for gene in ["TP53", "MYC"]:
    perturbation[gene] = -10.0
```

These are illustrative examples of the input representation, not specific experiments reported in the paper.

The baseline expression and biological gene embeddings are unchanged by this substitution.

### Target generation

Ground-truth logFC is computed by comparing experimentally perturbed cells against controls using scVI differential-expression analysis.

The authors additionally filter perturbation datasets to eliminate low-confidence perturbations.

Their filtering involves:

1. Standard single-cell quality control.
2. Target-expression filtering.
3. Perturbation Score thresholding.
4. Cluster-coherence filtering.

A separate scVI model is trained for each Perturb-seq dataset, with perturbation identity used as the batch key.

### Fine-tuning loss

Fine-tuning combines magnitude accuracy with correlation-based objectives.

\[
L_{\text{finetune}}
=
\alpha L_{\text{weighted MSE}}
+
\gamma L_{\text{correlation}}
\]

The weighted MSE emphasizes genes with larger true expression changes:

\[
L_{\text{weighted MSE}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(1+0.5|x_i^*|)
(\hat{x}_i-x_i^*)^2
\]

The correlation component is:

\[
L_{\text{correlation}}
=
1-
[
\beta\rho_{\text{global}}
+
(1-\beta)\rho_{\text{top}}
]
\]

Where:

- \(\rho_{\text{global}}\): Pearson correlation across all genes.
- \(\rho_{\text{top}}\): Pearson correlation across the top 20% of genes ranked by absolute ground-truth logFC.
- \(\alpha,\beta,\gamma\): Trainable scalar weights constrained to (0, 1) using sigmoid transformations.

The paper initializes these weights to 1.0, 0.5, and 0.5, respectively.

### Fine-tuning configuration

```yaml
optimizer: AdamW
learning_rate: 0.0001
max_epochs: 100
early_stopping_patience: 5

train_fraction: 0.8
validation_fraction: 0.1
test_fraction: 0.1

random_seeds: 3
mixed_precision: true
```

**Critical:** Splits are performed at the perturbation-gene level, not by randomly splitting individual cells.

This prevents the same perturbation identity from appearing in training and test sets.

For a faithful reproduction, use the published preprocessing, perturbation splits, and targets where available.

## 6. Evaluation methodology

The authors evaluate model predictions using four metrics.

| Metric | Purpose | Interpretation |
|---|---|---|
| Cosine similarity | Agreement between predicted and actual logFC directions | Higher is better |
| RMSE | Error in predicted logFC magnitude | Lower is better |
| Enrichment factor (EF) | Recovery of true differentially expressed genes | Higher is better |
| Directional F1 | Correct classification of upregulated and downregulated genes | Higher is better |

Evaluation uses \(K \in \{50,100,200,300,400,500\}\).

For cosine similarity and RMSE, the authors select the top-K genes by absolute *true* logFC. For enrichment factor, they compare the top-K genes independently ranked by absolute predicted and true logFC.

This distinction is important when reproducing the evaluation.

The F1 metric uses a 1.2-fold threshold, classifying genes as upregulated, downregulated, or unchanged. The reported score averages the F1 values for the upregulated and downregulated classes.

The paper also investigates two additional tasks.

**Reverse perturbation identification:** Given an observed logFC profile, predict the gene or gene combination responsible. Candidate perturbations are ranked by Euclidean distance between their predicted and observed logFC vectors.

**Drug-response transfer:** A model fine-tuned on Replogle K562 genetic perturbations is evaluated on SCIPLEX drug responses without further training. Drugs are approximated as simultaneous inhibition of their annotated protein targets.

The drug-transfer results demonstrate partial generalization but are substantially weaker than the genetic perturbation results.

---

## 7. Recommended implementation strategy

For a coding agent tasked with reproducing or extending scDEBART, I recommend the following implementation order.

### Phase 1 — Inspect the existing implementation

Start with the official repository:

https://github.com/Jieun-Sung/scDEBART

Determine:

- Whether pretrained model weights are available.
- Whether the model can be loaded independently of the training pipeline.
- Which formats are used for gene identifiers and biological embeddings.
- How preprocessing and scVI outputs are stored.
- How perturbation indicators are constructed.
- How the model handles sequences of different lengths.
- Whether evaluation scripts and published splits are provided.

**Prefer adapting the original implementation over reimplementing BART from scratch.**

### Phase 2 — Implement inference

The minimum useful inference interface should look conceptually like:

```python
def predict_perturbation(
    model,
    gene_ids,
    baseline_expression,
    perturbations,
    gene_embeddings,
):
    """
    Predict transcriptome-wide log2 fold changes.

    Args:
        model:
            Pretrained/fine-tuned scDEBART model.

        gene_ids:
            Ordered list of modeled genes.

        baseline_expression:
            scVI-denoised baseline expression,
            aligned to gene_ids.

        perturbations:
            Dictionary mapping target gene IDs
            to perturbation indicators:
                -10.0 = inhibition
                +10.0 = activation

        gene_embeddings:
            Precomputed BioBERT and MSigDB features.

    Returns:
        Predicted log2 fold changes aligned to gene_ids.
    """
```

This is a proposed interface, not a function signature taken from the repository.

The implementation should validate that all inputs share the same gene ordering and that every target gene exists in the model's gene vocabulary.

### Phase 3 — Reproduce fine-tuning

Once inference works, implement fine-tuning on an existing Perturb-seq dataset.

The pipeline is:

```text
Perturb-seq dataset
        |
        v
Cell quality control
        |
        v
Perturbation-response filtering
        |
        v
scVI differential expression
        |
        v
Baseline expression + true logFC
        |
        v
Gene-level train/validation/test split
        |
        v
Construct perturbation indicators
        |
        v
Load pretrained scDEBART
        |
        v
Fine-tune using weighted MSE
and correlation losses
        |
        v
Evaluate held-out perturbations
```

Initially, use a single dataset such as Replogle K562.

Avoid reconstructing the 66-million-cell pretraining corpus until the downstream pipeline has been validated.

### Phase 4 — Reproduce evaluation

Implement the following evaluation functions:

```python
cosine_similarity(pred, true, top_k)
rmse(pred, true, top_k)
enrichment_factor(pred, true, top_k)
directional_f1(pred, true, threshold=1.2)
```

Important implementation details:

- Use log2 fold changes consistently.
- Align predictions and targets by gene identifier.
- Preserve the exact evaluation gene universe.
- Apply the paper's detection-rate filtering.
- Rank genes by absolute logFC for DEG selection.
- Use the paper's enrichment-factor pseudocount.
- Evaluate only held-out perturbations.
- Aggregate metrics across three random seeds.

The EF formula in Appendix A.4.3 uses a pseudocount of 0.05, and the directional F1 calculation uses class-specific precision and recall with a numerical offset of \(10^{-6}\).

These should be preserved when reproducing published results.

---

## 8. Implementation risks and considerations

Several aspects deserve particular attention.

| Issue | Why it matters |
|---|---|
| Gene ordering | Predictions are meaningful only when all feature matrices and targets use the same gene order. |
| scVI consistency | Pretraining and fine-tuning use scVI-derived expression and logFC. |
| Model vocabulary | Genes absent from the learned vocabulary cannot be handled without additional design decisions. |
| Sequence length | Fine-tuning supports up to 5,000 genes, significantly more than the 1,024-gene pretraining profiles. |
| Padding masks | Attention and reconstruction losses must exclude padding positions. |
| Masked reconstruction | Pretraining loss is computed only at masked positions. |
| Decoder masking | The architecture uses causal decoder self-attention and encoder–decoder cross-attention. |
| Loss-weight optimization | The fine-tuning loss weights are trainable, not fixed constants. |
| Perturbation encoding | The ±10 indicators replace logFC inputs; they do not modify baseline expression directly. |
| Dataset splitting | Perturbations, rather than individual cells, define train/test separation. |
| Evaluation thresholds | DEG recovery and direction classification depend on precisely defined thresholds. |

Two additional issues are worth noting.

**First, the model learns population-level responses.** Its predictions should not be interpreted as generating biologically realistic individual-cell outcomes. The paper does not establish a cell-level generative model.

**Second, the model's strong DEG enrichment is partly associated with its preference for reliably detected genes.** This improves robustness but limits coverage of lowly expressed genes. Comparisons using different gene-filtering strategies may therefore produce substantially different results.

---

## 9. Suggested coding-agent task

The following is a self-contained implementation brief that can be pasted into Claude Code, Codex, or another coding agent.

### Task: Implement and validate scDEBART

**Objective**

Implement a reproducible pipeline for predicting transcriptome-wide gene-expression changes after genetic perturbations, following the scDEBART paper by Sung and Kim (2026).

Prioritize compatibility with the authors' published implementation, pretrained checkpoints, and processed datasets.

**Reference resources**

- Paper: Attached scDEBART PDF.
- GitHub: https://github.com/Jieun-Sung/scDEBART
- Data: https://huggingface.co/datasets/Jieun-S/scDEBART

**Requirements**

1. Examine the published source code, pretrained weights, data schema, and inference interface. Document any deviations between the implementation and paper.

2. Implement or adapt the scDEBART model, consisting of:
   - BioBERT gene embeddings (768 dimensions).
   - MSigDB pathway membership (7,411 dimensions).
   - Baseline-expression and perturbation-indicator projections.
   - An eight-layer transformer encoder.
   - A four-layer transformer decoder.
   - An embedding reconstruction head.
   - A scalar logFC prediction head.

3. Support perturbation-conditioned inference:
   - Input: baseline expression, gene identities, and perturbation targets.
   - Encoding: −10 for inhibition, +10 for activation, 0 otherwise.
   - Output: predicted log2 fold changes for all modeled genes.
   - Support simultaneous perturbation of multiple genes.

4. Implement fine-tuning using the paper's composite objective:
   - Weighted MSE with \(\lambda=0.5\).
   - Global Pearson correlation.
   - Top-20% Pearson correlation.
   - Trainable sigmoid-constrained loss weights.

5. Implement evaluation on held-out perturbation genes:
   - Cosine similarity.
   - RMSE.
   - DEG enrichment factor.
   - Directional F1.

6. Write tests verifying:
   - Input/output tensor dimensions.
   - Correct perturbation-indicator encoding.
   - Gene-order alignment.
   - Padding and attention masks.
   - Masked pretraining loss.
   - Differentiability of fine-tuning losses.
   - Correct top-K gene selection.
   - Reproducibility with fixed seeds.

7. Provide a minimal inference example and an evaluation script.

**Recommended project structure**

```text
scdebart/
├── configs/
│   ├── model.yaml
│   └── finetune.yaml
│
├── src/
│   ├── model.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── losses.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── perturbation.py
│   ├── inference.py
│   └── metrics.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   ├── test_embeddings.py
│   ├── test_losses.py
│   ├── test_perturbations.py
│   └── test_metrics.py
│
├── examples/
│   └── predict_knockdown.py
│
└── README.md
```

This directory layout is a suggested structure, not the published repository structure.

**Acceptance criteria**

The implementation is complete when:

- A released scDEBART checkpoint can be loaded, if publicly available.
- A user can specify one or more gene perturbations.
- The model produces a correctly aligned logFC vector.
- Fine-tuning can run on one supported Perturb-seq dataset.
- Evaluation reproduces the paper's metric definitions.
- Tests pass for perturbation encoding, model outputs, losses, and metrics.
- Configuration, random seeds, and preprocessing assumptions are documented.
- Reproduction results are compared against reported values without claiming equivalence when datasets, preprocessing, or checkpoints differ.

**Important instruction**

Do not silently replace the paper's architecture, loss functions, preprocessing choices, or evaluation definitions with conventional alternatives. Identify ambiguities, consult the released implementation, and document any necessary assumptions.

---

## 10. Final takeaway for implementation

The most important aspect of scDEBART is **not the transformer architecture itself, but the choice of learning objective and representation**.

The paper combines three elements:

1. **Expression-change pretraining:** Learning relationships between differential-expression patterns across millions of cell-population comparisons.
2. **Robust supervision:** Using scVI-denoised expression and excluding genes with unreliable detection.
3. **Perturbation conditioning:** Replacing observed logFC inputs with explicit perturbation indicators, then fine-tuning to predict full transcriptional responses.

An implementation that reproduces the transformer architecture but uses raw counts, absolute-expression targets, or an inappropriate evaluation split would miss important aspects of the method.

**Recommended starting point:** Load the official pretrained model, reproduce inference on Replogle K562, and validate DEG enrichment on held-out perturbations. Only then consider extending or modifying the architecture.
