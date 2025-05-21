from openai import OpenAI
from transformers import set_seed
import json_repair
import time
import random


class GPTAgent:
    def __init__(self, model_name, system_prompt, task="", max_new_tokens=512,
                 temperature=0.7, seed=10086, time_sleep_min=10, time_sleep_max=30):
        """
        初始化 GPTAgent
        """

        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.model = OpenAI(
            base_url="",
            api_key="",
            timeout=300000
        )
        self.system_prompt = system_prompt
        self.seed = seed
        self.task = task
        self.time_sleep_min = time_sleep_min
        self.time_sleep_max = time_sleep_max
        set_seed(self.seed)

    def query(self, prompt, print_prompt=False):
        """
        调用 OpenAI API 并返回结果
        """
        # 限制 prompt 长度
        prompt = prompt[:32768]
        if print_prompt:
            print(f"Prompt: {prompt}")

        # 随机睡眠模拟请求间隔
        sleep_time = random.randint(self.time_sleep_min, self.time_sleep_max)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

        try:
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            # 获取 API 响应内容
            res = completion.choices[0].message.content
            if print_prompt:
                print(f"Response: {res}")

            # 修复 JSON 格式并返回
            return json_repair.loads(res.replace("```json", "").replace("```", ""))

        except Exception as e:
            print(f"Error during API call: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # 实例化 GPTAgent
    agent = GPTAgent(
        model_name="gpt-4o-mini",  # 替换为实际模型名称
        system_prompt="You are a helpful assistant.",
        task="Cloud test",
        max_new_tokens=512,
        temperature=0.7,
        seed=42,
        time_sleep_min=1,
        time_sleep_max=3,
    )

    # 定义测试输入
    prompt = "Describe the future of AI in education."
    
    # 调用 query 方法并输出结果
    result = agent.query(prompt, print_prompt=True)
    print("Test result:", result)
