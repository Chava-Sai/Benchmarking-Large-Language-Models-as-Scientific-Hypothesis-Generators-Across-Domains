# How to Run — Step by Step

## STEP 1: Set up API keys locally (do this first)

```bash
cd /Users/sai/Desktop/ICML
cp .env.example .env
```

Open `.env` and fill in:
```
ANTHROPIC_API_KEY=sk-ant-...       ← your Claude key
OPENAI_API_KEY=sk-...              ← your OpenAI key
HF_TOKEN=hf_...                    ← from huggingface.co/settings/tokens (needed for LLaMA)
WANDB_API_KEY=...                  ← from wandb.ai/settings (for training logs)
```

---

## STEP 2: Deploy code to SCC

```bash
cd /Users/sai/Desktop/ICML
bash deploy_to_scc.sh
```

You'll be prompted for your BU Kerberos password once.

---

## STEP 3: SSH into SCC and set up the environment

```bash
ssh saichava@scc1.bu.edu
cd ~/icml2026
```

**Copy your API keys to SCC:**
```bash
# On your Mac, in a separate terminal:
scp /Users/sai/Desktop/ICML/.env saichava@scc1.bu.edu:~/icml2026/.env
```

**Run the one-time setup (installs Python venv + all packages):**
```bash
bash scripts/00_setup_env.sh
```
This takes ~10 minutes. Watch for: `CUDA: True | NVIDIA H200`

**Accept HuggingFace LLaMA license** (one-time, required for LLaMA models):
```bash
source venv/bin/activate
huggingface-cli login   # paste your HF_TOKEN
# Then visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# and click "Agree to terms" (same for 3.3-70B)
```

---

## STEP 4: Submit jobs in order

Each job emails saichava@bu.edu when it finishes/fails.

```bash
# Job 1: Download + preprocess all datasets (~2 hrs)
sbatch scripts/01_download_data.slurm
# Check: squeue -u saichava

# Job 2: Hypothesis generation — all 7 models x 3 strategies (~9 hrs)
# Submit AFTER job 1 finishes (you'll get an email)
sbatch scripts/02_generate_hypotheses.slurm

# Job 3: Factual QA evaluation — all models (~5 hrs)
# Can run at SAME TIME as job 2:
sbatch scripts/03_factual_eval.slurm

# Job 4: LoRA fine-tuning LLaMA-3.1-8B (~5 hrs)
# Can run at SAME TIME as jobs 2 & 3:
sbatch scripts/04_finetune.slurm

# Job 5: Evaluate fine-tuned model (~2 hrs)
# Submit AFTER job 4 finishes
sbatch scripts/05_eval_finetuned.slurm

# Job 6: Compute all metrics + generate figures (~1 hr)
# Submit AFTER jobs 2, 3, 5 all finish
sbatch scripts/06_metrics_figures.slurm
```

**Monitoring:**
```bash
squeue -u saichava           # see running jobs
tail -f logs/generate_*.out  # live output of job 2
cat logs/generate_*.err      # check for errors
```

---

## STEP 5: Get results back to your Mac

```bash
# From your Mac:
rsync -avz saichava@scc1.bu.edu:~/icml2026/results/ /Users/sai/Desktop/ICML/results/
```

Figures will be in `results/figures/` (PDF + PNG).
Leaderboard in `results/metrics/leaderboard.csv`.

---

## Model Lineup (7 models)

| Model | Type | Size | Why |
|---|---|---|---|
| **Claude Sonnet 4.6** | API | — | Best reasoning, strongest upper bound |
| **GPT-4o-mini** | API | — | Widely-used commercial baseline |
| **LLaMA-3.3-70B** | Local (AWQ 4-bit) | 70B | Best open-source large model |
| **Qwen2.5-72B** | Local (AWQ 4-bit) | 72B | Excellent science reasoning |
| **LLaMA-3.1-8B** | Local | 8B | Efficient small baseline |
| **Mistral-7B** | Local | 7B | Efficient small baseline |
| **LLaMA-3.1-8B-FT** | Local (ours) | 8B | Our contribution |

---

## Timeline (Apr 21–24)

| When | What |
|---|---|
| **Tonight** | Deploy, setup env, submit jobs 1 + 2 + 3 + 4 simultaneously |
| **Apr 22 morning** | Jobs done; check outputs; submit jobs 5 & 6 |
| **Apr 22 afternoon** | All results ready; start writing paper |
| **Apr 23** | Full paper draft |
| **Apr 24 11:59PM UTC** | Submit on OpenReview |
