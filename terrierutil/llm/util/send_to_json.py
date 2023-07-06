import fire 
import json 
import re

def main(prompt_text : str,
         output_file : str,):
    assert output_file.endswith(".json"), "Output file must be a .json file"
    pattern = r'\{([^]]+)\}'
    params = re.findall(pattern, prompt_text)

    output = {
        "prompt": prompt_text,
        "params": params if params else [],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)