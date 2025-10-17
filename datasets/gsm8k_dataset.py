import re
import torch
import asyncio

from KVCOMM.llm import LLMChat
from KVCOMM.utils.log import logger


def gsm_data_process(dataset):

    list_data_dict = []
    for data in dataset:
        item = {"task":data["question"]}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)

    return list_data_dict


def _strip_string(s):
    return s.strip()

def _extract_boxed_content(text: str) -> str:
    """Return the content inside a \\boxed expression, handling nested braces."""
    remainder = text.split("boxed", 1)[-1].lstrip()
    if not remainder:
        return ""
    if remainder[0] != "{":
        return remainder.split("$", 1)[0].strip()

    depth = 0
    buffer = []
    for char in remainder[1:]:
        if char == "{":
            depth += 1
            buffer.append(char)
        elif char == "}":
            if depth == 0:
                break
            depth -= 1
            buffer.append(char)
        else:
            buffer.append(char)
    return "".join(buffer).strip()


def gsm_get_predict(pred_str):
    text = str(pred_str)
    if "The answer is " in text:
        candidate = text.split("The answer is ", 1)[-1].strip()
    elif "the answer is " in text:
        candidate = text.split("the answer is ", 1)[-1].strip()
    elif "boxed" in text:
        candidate = _extract_boxed_content(text)
    else:
        candidate = text

    candidate = _strip_string(candidate).rstrip("./").strip()
    if "boxed" in candidate:
        candidate = _extract_boxed_content(candidate)
        candidate = _strip_string(candidate).strip()

    float_pattern = r"-?\d+\.\d+|-?\d+"
    matches = re.findall(float_pattern, candidate)

    return matches[0] if matches else "0"


async def get_pred_value(pred_str):
    if isinstance(pred_str, list):
        answer = pred_str[0] if pred_str else ""
    else:
        answer = pred_str
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.split("A: ", 1)[-1]

    tokenizer = LLMChat._shared_tokenizer
    model = LLMChat._shared_model
    if tokenizer is None or model is None:
        raise RuntimeError("Shared LLM resources not initialised before calling get_pred_value.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are professional in extracting the exact value from the user response. "
                "Given the following user response, you should only output the value that is argued "
                "as the most correct answer by the user for evaluation, i.e., only an integer or "
                "float-type value in one line."
            ),
        },
        {"role": "user", "content": answer},
    ]

    token_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": token_inputs.to(model.device),
        "attention_mask": torch.ones_like(token_inputs).to(model.device),
    }

    prompt_length = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "return_dict_in_generate": True,
        "return_legacy_cache": False,
    }
    outputs = await asyncio.to_thread(model.generate, **inputs, **generate_kwargs)
    generated = outputs.sequences[:, prompt_length:]
    response_message = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    logger.opt(colors=True).debug(f"<blue>[GSM8K]</blue> {response_message}")
    return response_message


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def delete_extra_zero(n):
    try:
        n=float(n)
    except Exception:
        logger.warning("Unable to convert value to float: {}", n)
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):

    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string


    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)

    return string
