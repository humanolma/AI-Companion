"""
测试脚本：验证情感感知功能
测试不同情绪输入下，Agent 的回复差异
"""
import sys
import os

# 设置 UTF-8 输出（Windows 终端兼容）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.companion import CompanionAgent
from src.config.settings import settings


def test_emotion():
    print("=" * 60)
    print(f"测试：{settings.companion_name} 情感感知功能")
    print("=" * 60)

    # 创建 Agent（启用情感感知，不启用长期记忆以加速测试）
    agent = CompanionAgent(use_long_term_memory=False, use_emotion=True)

    # 测试用例：不同情绪的输入
    test_cases = [
        ("今天面试通过了！太开心了！", "happy"),
        ("我好累，感觉什么都不想做...", "sad"),
        ("这个bug搞了一整天了，气死我了！", "angry"),
        ("明天要答辩了，好紧张怎么办...", "anxious"),
        ("今天天气还不错。", "neutral"),
    ]

    for user_input, expected_emotion in test_cases:
        print(f"\n{'─' * 50}")
        print(f"[用户] {user_input}")

        # 先检测情绪
        detected = agent.emotion_detector.detect(user_input)
        emotion_label = agent.emotion_detector.get_label(detected)
        print(f"[情绪检测] {detected} ({emotion_label})")

        # 获取回复
        response = agent.chat(user_input)
        print(f"[{settings.companion_name}] {response}")

        # 判断情绪检测是否正确
        if detected == expected_emotion:
            print(f"[结果] 情绪检测正确 ✓")
        else:
            print(f"[结果] 期望 {expected_emotion}，实际 {detected}（可能模型理解不同）")

        # 每轮重置短期记忆，避免上下文干扰
        agent.reset()

    print(f"\n{'=' * 60}")
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion()
