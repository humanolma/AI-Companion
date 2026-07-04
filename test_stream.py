"""
测试脚本：验证流式输出功能
"""
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.companion import CompanionAgent
from src.config.settings import settings


def test_stream():
    print("=" * 60)
    print(f"测试：{settings.companion_name} 流式输出功能")
    print("=" * 60)

    agent = CompanionAgent(use_long_term_memory=False, use_emotion=True)

    test_input = "今天面试通过了，太开心了！"

    print(f"\n[用户] {test_input}")
    print(f"[{settings.companion_name}] ", end="", flush=True)

    # 流式接收
    full_response = ""
    chunk_count = 0
    for piece in agent.chat_stream(test_input):
        print(piece, end="", flush=True)
        full_response += piece
        chunk_count += 1

    print(f"\n\n[流式统计]")
    print(f"  总字符数：{len(full_response)}")
    print(f"  分片次数：{chunk_count}")
    print(f"  检测情绪：{agent.current_emotion}")

    print(f"\n{'=' * 60}")
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_stream()
