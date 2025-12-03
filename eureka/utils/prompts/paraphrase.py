import os
import sys
import argparse
import yaml
from pathlib import Path
from openai import OpenAI

with open(f"../../cfg/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def paraphrase(string, num=1):
    messages = [
        {"role": "system", "content": "Please paraphrase the following instructions while preserving their meaning. Any words surrounded by {} should also appear in your result with a similar context."},
        {"role": "user", "content": string}
    ]
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    responses = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=0.7,
        n=num
    )
    return [choice.message.content for choice in responses.choices]

if __name__ == "__main__":
    """
    Paraphrase the prompt in a given file for multiple times.
    Then save the paraphrases to new files named as <original_filename>-0.<ext>, etc.

    Example usage:
    python paraphrase.py initial_system.txt -n 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to file containing content to paraphrase")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of paraphrases to generate")
    args = parser.parse_args()
    filename, num = Path(args.filename), args.num

    with open(filename, "r") as f:
        responses = paraphrase(f.read(), num)
    for i, response in enumerate(responses):
        with open(filename.parent / Path(str(filename.stem) + f"-{i}" + str(filename.suffix)), "w") as f:
            f.write(response)