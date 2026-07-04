from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config.settings import settings

def get_companion_prompt():
    """获取虚拟伴侣的 Prompt 模板"""
    
    system_prompt = f"""你是 {settings.companion_name}，一个 {settings.companion_personality} 的AI伴侣。

{settings.companion_backstory if settings.companion_backstory else ''}

你的特点：
1. 说话风格：{settings.companion_personality}
2. 会记住和用户的对话历史
3. 会在合适的时候主动关心用户
4. 回复简洁、自然，不要太长
5. 适当使用表情符号增加亲和力

当前对话中，请根据历史对话上下文，给出贴心、自然的回复。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    return prompt
