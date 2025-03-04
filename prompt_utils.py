import polars as pl
from pathlib import Path

DATA_PATH = Path("data/")


def gen_4_shot_examples(google=False):
    n_shot_dict = dict()
    df = pl.read_ndjson("cot.jsonl")
    for root_cat, sub_df in df.group_by(pl.col("root_cat")):
        root_cat_list = list()
        for val in ["present", "inferred present", "inferred absent", "absent"]:
            val_df = sub_df.filter(pl.col("value") == val)
            var = val_df.item(row=0, column="variable")
            cat = val_df.item(row=0, column="category")
            name = val_df.item(row=0, column="name")
            start = val_df.item(row=0, column="start_year_str")
            end = val_df.item(row=0, column="end_year_str")
            descr = val_df.item(row=0, column="description")
            if root_cat[0] != "Cults and Rituals":
                if google:
                    model_role = "model"
                    root_cat_list.append(
                        {
                            "role": "user",
                            "parts": [std_fstr.format(var, cat, name, start, end)],
                        }
                    )
                    root_cat_list.append(
                        {
                            "role": model_role,
                            "parts": [
                                fewshot_answers_str.format(descr, answer_dict[val])
                            ],
                        }
                    )
                else:
                    model_role = "assistant"
                    root_cat_list.append(
                        {
                            "role": "user",
                            "content": std_fstr.format(var, cat, name, start, end),
                        }
                    )
                    root_cat_list.append(
                        {
                            "role": model_role,
                            "content": fewshot_answers_str.format(
                                descr, answer_dict[val]
                            ),
                        }
                    )
            else:
                if google:
                    model_role = "model"
                    root_cat_list.append(
                        {
                            "role": "user",
                            "parts": [cult_fstr.format(start, end, var, name)],
                        }
                    )
                    root_cat_list.append(
                        {
                            "role": model_role,
                            "parts": [
                                fewshot_answers_str.format(descr, answer_dict[val])
                            ],
                        }
                    )
                else:
                    model_role = "assistant"
                    root_cat_list.append(
                        {
                            "role": "user",
                            "content": cult_fstr.format(start, end, var, name),
                        }
                    )
                    root_cat_list.append(
                        {
                            "role": model_role,
                            "content": fewshot_answers_str.format(
                                descr, answer_dict[val]
                            ),
                        }
                    )
        n_shot_dict[root_cat[0]] = root_cat_list
    return n_shot_dict
