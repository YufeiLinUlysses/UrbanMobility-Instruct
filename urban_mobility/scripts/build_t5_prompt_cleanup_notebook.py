"""Build a Colab notebook for the corrected T5 training and six evaluations."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/urban_science_t5_prompt_cleanup_rerun.ipynb'
CORRECTED = ROOT / 'data/revisions/revision_v6_prompt_cleaned'


def md(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(True)}


def main():
    required = [
        *(CORRECTED / 'experiment' / task for task in ('generation', 'verification', 'paraphrase')),
        CORRECTED / 'reporting/split_manifest.json',
        CORRECTED / 'reporting/leakage_audit.json',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Corrected release is incomplete: {missing}')
    cells = [
        md('''# T5 corrected-corpus diagnostic: training and all evaluations

Run top to bottom in a fresh Colab GPU runtime. This notebook trains only T5-base on the corrected `revision_v6_prompt_cleaned` partitions, then evaluates base and corrected fine-tuned checkpoints on the in-domain test, France/Japan, and independent ten-dataset benchmark. It publishes uniquely named artifacts so revision-v5 evidence is preserved.
'''),
        md('## 0. Confirm GPU'),
        code('''import subprocess
subprocess.run(["nvidia-smi"], check=True)
'''),
        md('## 1. Configuration'),
        code(f'''from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    git_url: str = "https://github.com/YufeiLinUlysses/LLM_Urban_Planning.git"
    git_branch: str = "revision"
    project_dir: str = "/content/LLM_Urban_Planning/urban_mobility"
    dataset_repo: str = "UlyssesLynne/urban_planning_llm"
    dataset_revision: str = "775783b5b652a710182b3f1a871e231c5516d942"
    dataset_subdir: str = "data/revision_v6_prompt_cleaned"
    dataset_version: str = "revision_v6_prompt_cleaned"
    model_key: str = "t5_base"
    run_id: str = "revision-v6-t5-prompt-cleaned-v1"
    model_repo: str = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
    prediction_repo: str = "UlyssesLynne/urban-planning-llm-predictions-v3"
    artifact_root: str = "/content/urban_science_artifacts"
    publish_to_hf: bool = True
    smoke_test: bool = True
    email_notifications: bool = False
    train_batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 100

CFG = Config()
CFG
'''),
        md('''## 2. Credentials

Add `HF_TOKEN` to Colab Secrets. Optional completion emails use `GMAIL_ADDRESS`, `GMAIL_TOKEN`, and optionally `NOTIFY_EMAIL`.
'''),
        code('''import os
from getpass import getpass

def secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

token = secret("HF_TOKEN") or getpass("HF token: ")
os.environ["HF_TOKEN"] = token
print("HF credential available; value not printed.")
'''),
        md('## 3. Clone project and install environment'),
        code('''import shutil, subprocess
from pathlib import Path

project = Path(CFG.project_dir)
repo = project.parent
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", CFG.git_branch, "--single-branch", CFG.git_url, str(repo)], check=True)
else:
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
if not shutil.which("uv"):
    subprocess.run(["python", "-m", "pip", "install", "-q", "uv", "huggingface_hub"], check=True)
subprocess.run(["uv", "sync", "--frozen"], cwd=project, check=True)
'''),
        md('''## 4. Download and install the corrected experiment data

The corrected partitions are downloaded from the private Hugging Face dataset repository at an immutable commit. No ZIP upload is required.
'''),
        code('''import json, shutil
from huggingface_hub import snapshot_download

snapshot = Path(snapshot_download(
    repo_id=CFG.dataset_repo,
    repo_type="dataset",
    revision=CFG.dataset_revision,
    token=os.environ["HF_TOKEN"],
    allow_patterns=[f"{CFG.dataset_subdir}/**"],
))
staging = snapshot / CFG.dataset_subdir
if not staging.exists():
    raise FileNotFoundError(f"Corrected dataset directory is absent from pinned snapshot: {staging}")

for task in ("generation", "verification", "paraphrase"):
    source = staging / "experiment" / task
    destination = project / "data/05_model_input/experiment" / task
    if destination.exists(): shutil.rmtree(destination)
    shutil.copytree(source, destination)
reporting = project / "data/08_reporting/experiment"
reporting.mkdir(parents=True, exist_ok=True)
shutil.copy2(staging / "reporting/split_manifest.json", reporting / "split_manifest.json")
shutil.copy2(staging / "reporting/leakage_audit.json", reporting / "leakage_audit.json")

manifest = json.loads((reporting / "split_manifest.json").read_text())
assert manifest["dataset_version"] == CFG.dataset_version
assert manifest["generation_record_counts"] == {"cross_regional":4166,"train":8326,"test":1198,"validation":1036}
audit = json.loads((reporting / "leakage_audit.json").read_text())
assert audit["passed"]
print("Corrected partitions installed and deterministic audit verified.")
'''),
        md('## 5. Command helpers'),
        code('''import gc, smtplib
from collections import deque
from email.message import EmailMessage

def params(values):
    return ",".join(f"{k}={str(v).lower() if isinstance(v,bool) else v}" for k,v in values.items())

def run(*args):
    env=os.environ.copy(); env.update({"KEDRO_DISABLE_TELEMETRY":"1","PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false","MPLBACKEND":"Agg"})
    command=["uv","run",*args]
    print("Running:"," ".join(command),flush=True)
    tail=deque(maxlen=120)
    process=subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line,end="",flush=True)
        tail.append(line)
    return_code=process.wait()
    if return_code:
        diagnostic="".join(tail)
        print("\\nLAST 120 KEDRO LOG LINES\\n"+diagnostic,flush=True)
        error=subprocess.CalledProcessError(return_code,command,output=diagnostic)
        raise error

def release_memory():
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def notify(subject, body):
    if not CFG.email_notifications: return
    address=secret("GMAIL_ADDRESS"); password=(secret("GMAIL_TOKEN") or "").replace(" ","")
    target=secret("NOTIFY_EMAIL") or address
    if not address or not password: return
    message=EmailMessage(); message["From"]=address; message["To"]=target; message["Subject"]=subject; message.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:
        smtp.login(address,password); smtp.send_message(message)
'''),
        md('## 6. Run tests and optional smoke training'),
        code('''run("pytest", "tests/pipelines/data_augmentation", "tests/pipelines/test_experiment_data_nodes.py", "-q")
if CFG.smoke_test:
    run("kedro","run","--env=colab","--pipelines=train_model","--params="+params({
        "training.model_key":CFG.model_key,"training.run_id":CFG.run_id+"-smoke",
        "training.artifact_root":CFG.artifact_root,"training.publish_to_hf":False,
        "training.max_train_samples":200,"training.max_validation_samples":50,
        "training.per_device_train_batch_size":8,"training.per_device_eval_batch_size":8,
        "training.gradient_accumulation_steps":2,"training.use_chat_template":False,
    }))
    release_memory()
'''),
        md('## 7. Full corrected T5 training'),
        code('''train_params={
    "training.model_key":CFG.model_key,"training.run_id":CFG.run_id,
    "training.artifact_root":CFG.artifact_root,"training.model_repo_id":CFG.model_repo,
    "training.publish_to_hf":CFG.publish_to_hf,"training.use_chat_template":False,
    "training.per_device_train_batch_size":CFG.train_batch_size,
    "training.per_device_eval_batch_size":CFG.eval_batch_size,
    "training.gradient_accumulation_steps":CFG.gradient_accumulation_steps,
    "training.num_train_epochs":1,"training.learning_rate":CFG.learning_rate,
    "training.warmup_ratio":CFG.warmup_ratio,"training.weight_decay":CFG.weight_decay,
    "training.logging_steps":10,"training.eval_steps":CFG.eval_steps,"training.save_steps":CFG.save_steps,
    "training.save_total_limit":2,"training.early_stopping_patience":0,"training.random_seed":42,
}
run("kedro","run","--env=colab","--pipelines=train_model","--params="+params(train_params))
release_memory(); notify("T5 corrected training complete",CFG.run_id)
'''),
        md('## 8. Evaluate in-domain and France/Japan'),
        code('''def evaluate_primary(stage,scope):
    label="base" if stage=="base" else "finetuned"
    values={
        "evaluation.model_key":CFG.model_key,"evaluation.checkpoint_stage":stage,
        "evaluation.dataset_scope":scope,"evaluation.run_id":f"{CFG.run_id}-{label}-{scope.replace('_','-')}",
        "evaluation.artifact_root":CFG.artifact_root,"evaluation.prediction_repo_id":CFG.prediction_repo,
        "evaluation.publish_to_hf":CFG.publish_to_hf,"evaluation.batch_size":8,
        "evaluation.metric_batch_size":8,
        "evaluation.use_chat_template":False,"evaluation.metric_device":"cuda",
    }
    if stage=="fine_tuned": values.update({"evaluation.checkpoint_uri":CFG.model_repo,"evaluation.checkpoint_subfolder":f"{CFG.model_key}/{CFG.run_id}/checkpoint"})
    run("kedro","run","--env=colab","--pipelines=evaluate_model","--params="+params(values)); release_memory()

for scope in ("in_domain","cross_regional"):
    for stage in ("base","fine_tuned"):
        evaluate_primary(stage,scope)
'''),
        md('## 9. Prepare and evaluate the ten-dataset benchmark'),
        code('''run("kedro","run","--env=colab","--pipelines=prepare_multi_regional_data")
quality=json.loads((project/"data/08_reporting/multi_regional_external_v1/quality.json").read_text())
assert quality["passed"] and quality["accepted_record_count"]==350

def evaluate_external(stage):
    label="base" if stage=="base" else "finetuned"
    values={
        "evaluation.model_key":CFG.model_key,"evaluation.checkpoint_stage":stage,
        "evaluation.dataset_scope":"multi_regional","evaluation.run_id":f"{CFG.run_id}-{label}-multi-regional",
        "evaluation.artifact_root":CFG.artifact_root,"evaluation.prediction_repo_id":CFG.prediction_repo,
        "evaluation.publish_to_hf":CFG.publish_to_hf,"evaluation.batch_size":8,
        "evaluation.metric_batch_size":8,
        "evaluation.use_chat_template":False,"evaluation.metric_device":"cuda","evaluation.evaluate_paraphrase":False,
    }
    if stage=="fine_tuned": values.update({"evaluation.checkpoint_uri":CFG.model_repo,"evaluation.checkpoint_subfolder":f"{CFG.model_key}/{CFG.run_id}/checkpoint"})
    run("kedro","run","--env=colab","--pipelines=evaluate_multi_regional_model","--params="+params(values)); release_memory()

for stage in ("base","fine_tuned"): evaluate_external(stage)
'''),
        md('## 10. Verify all artifacts and print metrics'),
        code('''from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
model_files=set(api.list_repo_files(CFG.model_repo,repo_type="model"))
model_prefix=f"{CFG.model_key}/{CFG.run_id}/checkpoint"
for name in ("training_manifest.json","training_history.json","config.json","model.safetensors","tokenizer.json"):
    assert f"{model_prefix}/{name}" in model_files, f"Missing {name}"

prediction_files=set(api.list_repo_files(CFG.prediction_repo,repo_type="dataset"))
required=("evaluation_manifest.json","metrics.json","predictions.parquet","grouped_metrics.parquet","review_queue.parquet")
locations=[]
for scope,version in (("in_domain",CFG.dataset_version),("cross_regional",CFG.dataset_version),("multi_regional","multi_regional_external_v1")):
    for stage in ("base","fine_tuned"):
        label="base" if stage=="base" else "finetuned"
        run_id=f"{CFG.run_id}-{label}-{scope.replace('_','-')}"
        prefix=f"evaluations/{version}/{CFG.model_key}/{stage}/{scope}/{run_id}"
        for name in required: assert f"{prefix}/{name}" in prediction_files, f"Missing {prefix}/{name}"
        locations.append(prefix)
print("Verified checkpoint and all six evaluation bundles.")
print("\\n".join(locations))
notify("T5 corrected diagnostic verified","Checkpoint and six evaluations are complete.\\n"+"\\n".join(locations))
'''),
    ]
    notebook={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'name':'T5 corrected corpus diagnostic','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    OUTPUT.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print(OUTPUT)


if __name__=='__main__': main()
