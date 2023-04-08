from pydantic import BaseModel, Field

from .pydantic_parser import get_format_instructions, parse_pydantic_object


# TODO: Make it modular.
DEF_TASK_DESCRIPTION = """CONSTRAINTS:

1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"

COMMANDS:

1. Google Search: "google", args: "input": "<search>"
5. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
6. Start GPT Agent: "start_agent",  args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
7. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
8. List GPT Agents: "list_agents", args: ""
9. Delete GPT Agent: "delete_agent", args: "key": "<key>"
10. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
11. Read file: "read_file", args: "file": "<file>"
12. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
13. Delete file: "delete_file", args: "file": "<file>"
14. Search Files: "search_files", args: "directory": "<directory>"
15. Evaluate Code: "evaluate_code", args: "code": "<full _code_string>"
16. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
17. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
18. Execute Python File: "execute_python_file", args: "file": "<file>"
19. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

RESOURCES:

1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

PERFORMANCE EVALUATION:

1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities. 
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

{format_instructions}
"""


class Command(BaseModel):
    name: str = Field(description="command name")
    args: dict[str, str] = Field(
        description="A dictionary where keys and values are both strings, e.g., {'arg1': 'value1', 'arg2': 'value2'}"
    )


class Thought(BaseModel):
    text: str = Field(description="text")
    reasoning: str = Field(description="reasoning")
    plan: str = Field(
        description="- short bulleted\n- list that conveys\n- long-term plan"
    )
    criticism: str = Field(description="constructive self-criticism")
    speak: str = Field(description="thoughts summary to say to user")


def load_prompt():
    return DEF_TASK_DESCRIPTION.format(format_instructions=get_instructions())


def get_instructions():
    return get_format_instructions([Command, Thought])


def parse(text, pydantic_object):
    try:
        return parse_pydantic_object(text, pydantic_object)
    except Exception:
        raise Exception(
            f"I couldn't parse your format for object: {pydantic_object.__name__}, remember you should follow the instructions: {get_format_instructions([pydantic_object])}"
        )