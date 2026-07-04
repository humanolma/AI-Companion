"""
测试脚本：验证长期记忆（ChromaDB）功能
"""
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.companion import CompanionAgent
from src.config.settings import settings


def test_long_term_memory():
    """测试长期记忆功能"""
    print("=" * 60)
    print(f"测试：{settings.companion_name} 长期记忆功能")
    print("=" * 60)

    # 创建 Agent（启用长期记忆）
    agent = CompanionAgent(use_long_term_memory=True)

    # 第一轮对话
    print("\n[第一轮对话】")
    inputs = [
        "你好，我叫小明，今年 25 岁，住在上海。",
        "我喜欢打篮球和写代码。",
    ]
    for user_input in inputs:
        print(f"👤 用户：{user_input}")
        response = agent.chat(user_input)
        print(f"🤖 {settings.companion_name}：{response}")

    # 重置短期记忆（模拟第二天的对话）
    print("\n[重置短期记忆，模拟第二天对话】")
    agent.reset()

    # 第二轮对话（测试长期记忆检索）
    print("\n[第二轮对话 — 测试长期记忆】")
    test_input = "你还记得我叫什么名字吗？我住在哪？"
    print(f"👤 用户：{test_input}")
    response = agent.chat(test_input)
    print(f"🤖 {settings.companion_name}：{response}")

    # 验证：检查 ChromaDB 中是否有数据
    print("\n[验证】检查向量数据库...")
    if agent.long_term_memory:
        # 手动检索一次
        memories = agent.long_term_memory.retrieve_memories("用户名字住址")
        print(f"检索到的记忆条数：{len(memories)}")
        for i, mem in enumerate(memories):
            print(f"  记忆 {i+1}：{mem[:50]}...")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    test_long_term_memory()
