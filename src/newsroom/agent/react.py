"""手撕 ReAct：不用 function calling，只靠提示词和文本解析实现「思考 → 行动 → 观察」。

国内面试常让你 30 分钟内现场写这个，所以单独实现了一版，和 loop.py 的区别：
- loop.py 用接口原生的 tool_calls，模型输出是结构化的，更稳，生产用它；
- 这里模型输出纯文本，我们用正则解析 Action 和 Action Input，能看清 ReAct 论文的原始思路，
  也能讲清它的缺点：格式容易错、模型可能自己编 Observation（所以要用 stop 序列截断）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..gateway.gateway import ModelGateway
from ..gateway.types import ChatRequest, Message, ToolCall
from ..tools.registry import ToolRegistry

REACT_SYSTEM = """你可以使用这些工具：
{tools}

严格按下面的格式一步一步来：
Thought: 你现在在想什么
Action: 工具名，必须是 [{names}] 之一
Action Input: 工具参数，一个 JSON 对象
Observation: 工具的返回结果（由系统填写，你不要自己写）
……（Thought / Action / Action Input / Observation 可以重复多次）
Thought: 我已经知道答案了
Final Answer: 最终答案"""

_ACTION = re.compile(r"Action:\s*([\w\-]+)")
_INPUT = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)
_FINAL = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


@dataclass
class ReactResult:
    answer: str | None
    steps: int
    stopped_reason: str  # final | max_steps
    transcript: str
    actions: list[tuple[str, dict]] = field(default_factory=list)


def react(gateway: ModelGateway, registry: ToolRegistry, question: str, max_steps: int = 6,
          task: str = "react") -> ReactResult:
    tool_desc = "\n".join(f"- {s.name}: {s.description} 参数：{json.dumps(s.parameters, ensure_ascii=False)}"
                          for s in registry.specs())
    system = REACT_SYSTEM.format(tools=tool_desc, names=", ".join(registry.tools))
    transcript = ""
    seen: set[str] = set()
    actions: list[tuple[str, dict]] = []

    for step in range(1, max_steps + 1):
        resp = gateway.chat(ChatRequest(
            messages=[Message(role="system", content=system),
                      Message(role="user", content=f"问题：{question}\n\n{transcript}")],
            task=task, stop=["Observation:"]))  # stop 序列：防止模型自己编造观察结果
        text = resp.message.content.split("Observation:")[0].strip()

        final = _FINAL.search(text)
        if final:
            return ReactResult(final.group(1).strip(), step, "final", transcript + text, actions)

        action, inp = _ACTION.search(text), _INPUT.search(text)
        if not action:
            observation = "格式错误：没有找到 Action。请按 Thought / Action / Action Input 的格式输出。"
        else:
            try:
                args = json.loads(inp.group(1)) if inp else {}
            except json.JSONDecodeError as e:
                args, observation = None, f"Action Input 不是合法 JSON：{e}"
            if args is not None:
                key = action.group(1) + json.dumps(args, sort_keys=True, ensure_ascii=False)
                if key in seen:  # 死循环检测：一模一样的动作做第二遍
                    observation = "你刚才已经做过完全一样的动作了，结果不会变。换个动作，或者给出 Final Answer。"
                else:
                    seen.add(key)
                    actions.append((action.group(1), args))
                    observation = registry.execute(
                        ToolCall(id=f"react-{step}", name=action.group(1), arguments=args)).content
        transcript += f"{text}\nObservation: {observation[:2000]}\n"

    return ReactResult(None, max_steps, "max_steps", transcript, actions)
