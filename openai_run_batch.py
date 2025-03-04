from openai import OpenAI
import os
from dotenv import load_dotenv

from prompt_utils import gen_4_shot_examples, DATA_PATH

from tqdm import tqdm
import datetime
import json
import polars as pl

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
seed = 42
model_names = [
    "gpt-4o-2024-05-13",
    "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo-0125",
]
# model_names = ["gpt-4o-mini-2024-07-18"]

task_name = "guess_value"
request_dir = DATA_PATH / "runs" / "batch_requests"
df = pl.read_parquet("Neurips_HiST-LLM.parquet")

batch_num = 1
n_shot_dict = gen_4_shot_examples()

for model_name in model_names:
    tasks = list()
    for ix, row in tqdm(enumerate(df.iter_rows(named=True))):
        request_id = row["id"]

        message = json.loads(row["message"])
        task = {
            "custom_id": f"{request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "seed": 42,
                "temperature": 0,
                "max_tokens": 150,
                "messages": message,
            },
        }

        tasks.append(task)

    batch_save_path = (
        request_dir
        / f'{task_name}_Batch-{batch_num}_{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.jsonl'
    )
    with open(batch_save_path, "w") as handle:
        for entry in tasks:
            json.dump(entry, handle)
            handle.write("\n")
    batch_file = client.files.create(file=open(batch_save_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
