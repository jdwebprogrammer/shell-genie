from transformers import pipeline
import torch

CACHE_DIR = "./cache" # set whatever cache location you'd like

class CodeGen:
    def __init__(self, model="ise-uiuc/Magicoder-S-DS-6.7B", max_length=1024, num_return_sequences=1, temperature=0.0):
        self.generator = pipeline(model=model,task="text-generation",torch_dtype=torch.bfloat16,device_map="cuda:1")
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature

    def generate(self, instruction):
        prompt = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
            @@ Instruction
            {instruction}

            @@ Response
            """
        result = self.generator(prompt, max_length=self.max_length, num_return_sequences=self.num_return_sequences, temperature=self.temperature)
        return result[0]["generated_text"]


class BaseGenie:
    def __init__(self):
        self.code_gen = CodeGen()

    def ask(self, wish: str, explain: bool = False):
        raise NotImplementedError

    def post_execute(self, wish: str, explain: bool, command: str, description: str, feedback: bool):
        pass


class TrueOpenGenie(BaseGenie):
    def __init__(self, os_fullname: str, shell: str):
        self.os_fullname = os_fullname
        self.shell = shell

    def _build_prompt(self, wish: str, explain: bool = False):
        explain_text = ""
        format_text = "Command: <insert_command_here>"

        if explain:
            explain_text = "Also, provide a detailed description of how the command works."
            format_text += "\nDescription: <insert_description_here>\nThe description should be in the same language the user is using."
        format_text += "\nDon't enclose the command with extra quotes or backticks."
        
        prompt_list = [
            f"Instructions: Write a CLI command that does the following: {wish}. Make sure the command is correct and works on {self.os_fullname} using {self.shell}. {explain_text}",
            "Format:", format_text,
            "Make sure you use the format exactly as it is shown above."]
        prompt = "\n\n".join(prompt_list)
        return prompt

    def ask(self, wish: str, explain: bool = False):
        prompt = self._build_prompt(wish, explain)
        response = self.code_gen.generate(f"You're a command line tool that generates CLI commands for the user. {prompt}")
            ],
            max_tokens=300 if explain else 180,
            temperature=0,
        )
        responses_processed = response.strip().split("\n")
        responses_processed = [x.strip() for x in responses_processed if len(x.strip()) > 0]
        command = responses_processed[0].replace("Command:", "").strip()

        if command[0] == command[-1] and command[0] in ["'", '"', "`"]:
            command = command[1:-1]

        description = None
        if explain:
            description = responses_processed[1].split("Description: ")[1]

        return command, description

