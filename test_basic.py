"""
测试脚本：验证 AI 虚拟伴侣基本功能
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.agent.companion import CompanionAgent
from src.config.settings import settings

def test_basic_chat():
    """测试基本对话功能"""
    print("=" * 50)
    print(f"测试：{settings.companion_name} 基本对话")
    print("=" * 50)
    
    # 创建 Agent
    agent = CompanionAgent()
    
    # 测试对话
    test_inputs = [
        "你好，我叫小明。",
        "我喜欢打篮球，你呢？",
        "你还记得我的名字吗？",
    ]
    
    for user_input in test_inputs:
        print(f"\n[用户]：{user_input}")
        response = agent.chat(user_input)
        print(f"[{settings.companion_name}]：{response}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_basic_chat()
