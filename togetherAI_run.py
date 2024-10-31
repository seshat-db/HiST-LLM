#!/usr/bin/env python3
import os, asyncio
from together import AsyncTogether
import aiofiles
from datetime import datetime
from typing import Any, AsyncIterator, Callable
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import polars as pl
from tenacity import (
    retry,
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
)


seed = 42
temperature = 0
model_names = [
    (
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama_3.1_8B",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
    ),
    (
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Llama_3.1_70B",
        "Meta-Llama-3.1-70B-Instruct-Turbo",
    ),
    ("meta-llama/Llama-3-70b-chat-hf", "Llama_3_70B", "Llama-3-70b-chat-hf"),
    (
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "Llama-3.1-405B",
        "Llama-3.1-405B-Instruct-Turbo",
    ),
]
model_name, model_dir, model_str = model_names[-3]

DATA_PATH = Path("/home/scrappy/csh/Hapi/data")
async_batch_size = 10


async def extract_q_request_id(df: pl.DataFrame):
    """
    Extracts tuples of ('Q', 'request_id') from rows where 'model_name' is None.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        list: A list of tuples, each containing the 'Q' value and 'request_id' from rows where 'model_name' is None.
    """
    # Filter the DataFrame for rows where 'model_name' is None

    # Select the 'Q' and 'request_id' columns and convert to list of tuples
    for slice_df in df.iter_slices(n_rows=async_batch_size):
        # await asyncio.sleep(0.2)
        yield (slice_df["message"].to_list(), slice_df["id"].to_list())


async def save_results_to_file(filename: str, data: list):
    """
    Asynchronously save data to a file in JSON format.

    Args:
        filename (str): The name of the file where data will be saved.
        data (list): A list of processed items to save.
    """
    async with aiofiles.open(filename, "w") as file:
        await file.write(json.dumps(data))


@retry(
    wait=wait_random_exponential(min=0.1, max=10, multiplier=0.5),
    stop=stop_after_attempt(8),
)
async def async_chat_completion(messages):
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    tasks = [
        async_client.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=0,
        )
        for message in messages
    ]
    responses = await asyncio.gather(*tasks)

    return responses


async def process_iterator(
    it: AsyncIterator[Any], batch_size: int = 100, results=list(), counter=0
):
    """
    Process items from an async iterator, saving results periodically to a file.

    Args:
        it (AsyncIterator[Any]): The async iterator providing data items.
        filename (str): The filename to save periodic results.
        batch_size (int): Number of results to collect before saving to a file.
    """
    # results = []

    async for item in it:
        messages = [json.loads(i) for i in item[0]]
        request_ids = item[1]
        responses = await async_chat_completion(messages)
        for response, request_id in zip(responses, request_ids):
            results.append(
                {
                    "request_id": request_id,
                    model_name: response.choices[0].message.content,
                }
            )

        counter += async_batch_size
        print(f"{model_dir}: {counter}")

        if counter % (async_batch_size * 10) == 0:
            filename = (
                DATA_PATH
                / "togetherai_runs"
                / model_dir
                / f"{model_str}_{counter}.json"
            )
            print(f"{model_dir} saving {len(results)} requests")
            await save_results_to_file(filename, results)

    # Save any remaining results
    if results:
        filename = (
            DATA_PATH / "togetherai_runs" / model_dir / f"{model_str}_finished.json"
        )
        await save_results_to_file(filename, results)


async def main():
    task_df = pl.read_parquet(DATA_PATH / "prompt_base.parquet")

    # async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_ai_path = DATA_PATH / "togetherai_runs" / model_dir
    checkpoints = os.listdir(together_ai_path)
    print(model_name)
    if checkpoints:
        checkpoint_dict = {
            i.replace(".json", "").replace(f"{model_str}_", ""): i for i in checkpoints
        }
        if "finished" in checkpoint_dict.keys():
            checkpoint = checkpoint_dict["finished"]
            print("you are finished dum dum")
            return None
        else:
            max_check = max([int(i) for i in checkpoint_dict.keys()])

            print(f"{model_dir}\nstarting from checkpoint:{max_check}")
            checkpoint = checkpoint_dict[str(max_check)]

        checkpoint_df = pl.read_json(together_ai_path / checkpoint)
        with open(together_ai_path / checkpoint) as f:
            results = json.load(f)
        print(f"original tasks shape {task_df.shape}")
        task_df = task_df.filter(~pl.col("id").is_in(checkpoint_df["request_id"]))
        print(f"updated tasks shape {task_df.shape}")
        task_it = extract_q_request_id(task_df)
        await process_iterator(task_it, results=results, counter=max_check)

    else:
        task_it = extract_q_request_id(task_df)
        await process_iterator(task_it)


if __name__ == "__main__":
    asyncio.run(main())
