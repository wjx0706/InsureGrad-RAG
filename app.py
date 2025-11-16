import gradio as gr
from utils import user_asks

# Gradio 界面设置
with gr.Blocks(title="五险一金规划助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧑‍🏫应届生的第一个五险一金社保规划师")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话框", type="messages")
            user_input = gr.Textbox(label="请输入你的问题", placeholder="如：我现在即将入职私企算法工程师岗位，想知道试用期公司会给我交社保和公积金吗？", lines=3)
            # 按钮可保留，但不需要 click 事件
            send_button = gr.Button("发送")
            # stop_button = gr.Button("⬛停止生成", variant="stop")

        # with gr.Column(scale=1):
        #     gr.Markdown("#### 你的意图")
        #     user_intent = gr.Radio(["简单科普", "急需帮助", "帮我避坑"], label="当前意图", value="简单科普")

        with gr.Column(scale=1):
            gr.Markdown("#### 你的基本信息")
            situation = gr.Dropdown(["正在求职", "即将入职", "还在上学", "实习中", "准备升学", "其他"], label="目前状态", value="即将入职")
            job_input = gr.Textbox(label="你的工作/岗位", placeholder="如：程序员、教师、自由职业者等")
            city = gr.Textbox(label="所在城市", placeholder="如：北京、上海")
            age = gr.Textbox(label="年龄", value="25")
            user_goal = gr.Textbox(label="你未来的计划是什么？", placeholder="如：我目前打算先工作，未来可能跳槽。", lines=3)
            other_info = gr.Textbox(label="其他补充信息（可选）")
            
            
    # 设置输入框和按钮
    click_event = user_input.submit(
        fn=user_asks,
        inputs=[user_input, chatbot, user_goal, job_input, situation, 
                city, age, other_info],
        outputs=[chatbot, user_input], # user_input 清空输入框
        queue=True # 启用队列，处理快速输入
    )

    # 设置发送按钮
    submit_event = send_button.click(
        fn=user_asks,
        inputs=[user_input, chatbot, user_goal, job_input, situation,
                city,  age, other_info],
        outputs=[chatbot, user_input],
        queue=True
    )

    # 设置取消按钮
    # stop_button.click(
    #     fn=None,
    #     inputs=None,
    #     outputs=None,
    #     cancels=[submit_event, click_event],
    #     queue=False
    # )

# 运行 Gradio 应用
demo.launch(share=True)

