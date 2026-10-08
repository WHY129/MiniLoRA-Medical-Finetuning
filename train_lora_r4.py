"""
LoRA 微调训练脚本：在 Qwen2.5 上用 LoRA 做医疗健康领域 SFT

输入: data/medical/train.jsonl + valid.jsonl（由 prepare_medical_sft.py 生成）
输出: outputs/qwen-medical-lora/（LoRA adapter 权重）

运行方式:
    python scripts/train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1
    python scripts/train_lora.py  # 全量训练

核心流程:
    1. 加载 tokenizer 和模型
    2. 配置 LoRA（冻结原始权重，只训练低秩矩阵）
    3. 加载数据，用 preprocess 转成 token 序列
    4. 用 HuggingFace Trainer 训练
    5. 保存 LoRA adapter
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


# ============================================================
# 常量定义
# ============================================================

# -100 是 PyTorch CrossEntropyLoss 的 ignore_index（忽略索引）
# 在 labels 中，值为 -100 的位置不参与 loss 计算
# SFT 标准做法：prompt 部分设为 -100（不学复述问题），只对 answer 部分计算 loss
# 这叫 "assistant-only loss mask"
IGNORE_INDEX = -100


# ============================================================
# 数据加载
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """读取 jsonl 文件，返回 list[dict]

    列表推导式写法，等价于:
        items = []
        for line in f:
            if line.strip():        # strip() 去掉空白后如果不为空
                items.append(json.loads(line))
        return items
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 数据格式转换：原始数据 → messages
# ============================================================

def build_messages(example: dict) -> list[dict]:
    """将一条原始数据转成 chat messages 格式

    输入 example 格式（来自 jsonl）:
        {
            "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
            "input": "高血压应该怎么办？",
            "output": "高血压患者应该..."
        }

    输出 messages 格式（3 条 message）:
        [
            {"role": "system",    "content": "你是一名谨慎的医疗健康问答助手。..."},
            {"role": "user",      "content": "请以谨慎、专业、易懂的方式...\n\n问题：高血压应该怎么办？"},
            {"role": "assistant", "content": "高血压患者应该..."},
        ]

    为什么要转成 messages 格式？
    - Qwen 是 chat 模型，训练和推理都用 messages 格式
    - tokenizer.apply_chat_template() 会根据 messages 自动拼接特殊 token
      比如 system 前会有 <|im_start|>system，后面会有 <|im_end|>
    - 不同模型的 chat template 不同，用 messages 格式可以统一处理
    """
    # 拼接 user 内容：instruction + 换行 + "问题：" + input
    user_content = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    if input_text:
        user_content = f"{user_content}\n\n问题：{input_text}"

    return [
        {"role": "system", "content": "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]


# ============================================================
# 数据预处理：messages → token 序列（模型能吃的格式）
# ============================================================

def preprocess(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    """将一条数据转成训练用的 token 序列

    返回:
        {
            "input_ids": [token1, token2, ...],          # prompt tokens + answer tokens
            "attention_mask": [1, 1, 1, ...],            # 全 1（后面 DataCollator 会动态 pad）
            "labels": [-100, -100, ..., ans_tok1, ...]   # prompt 部分 -100，answer 部分真实 id
        }

    这是 SFT 数据处理中最核心的函数，流程如下:
    """
    # --- 步骤 1: 构造 messages ---
    messages = build_messages(example)

    # --- 步骤 2: 分离 prompt 和 answer ---
    # prompt_messages = system + user（前 2 条），用来生成输入
    # answer = assistant 的回复（最后 1 条），是模型要学习生成的目标
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    # --- 步骤 3: tokenize prompt ---
    # apply_chat_template 会把 messages 转成模型特有的 token 序列
    # 比如 Qwen 的模板会生成:
    #   <|im_start|>system\n你是一名...<|im_end|>\n
    #   <|im_start|>user\n问题...<|im_end|>\n
    #   <|im_start|>assistant\n  ← add_generation_prompt=True 会加上这个
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,             # 返回 token ids，而不是字符串
        add_generation_prompt=True,  # 在末尾加上 assistant 的开头标记
        return_tensors=None,       # 返回 list，不返回 tensor（方便后续处理）
    )

    # ⚠️ 踩坑点：apply_chat_template 的返回值类型不固定
    # 有些版本返回 BatchEncoding 对象（有 .input_ids 属性）
    # 有些版本返回 list
    # 有些版本返回 tensor
    # 所以要做类型检查和转换
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids     # BatchEncoding → 提取 input_ids
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()      # tensor → list

    # --- 步骤 4: tokenize answer ---
    # answer 后面加 eos_token（结束标记），告诉模型"回答到这里结束"
    # add_special_tokens=False：不加 BOS 等特殊标记（prompt 部分已经加过了）
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # --- 步骤 5: 拼接 input_ids ---
    # 最终序列 = prompt tokens + answer tokens
    # 模型看到 prompt tokens 后，应该生成 answer tokens
    input_ids = prompt_ids + answer_ids

    # --- 步骤 6: 构造 labels（关键！）---
    # labels 是 SFT 的核心概念:
    #   - prompt 部分: 设为 -100 → CrossEntropyLoss 会忽略这些位置
    #   - answer 部分: 保留真实 token id → 模型在这些位置计算 loss
    #
    # 为什么 prompt 部分不计算 loss？
    #   因为 prompt 是"问题"，模型不需要学习"复述问题"
    #   模型只需要学习"看到问题后怎么回答"
    #
    # 直觉理解:
    #   input_ids: [你, 是, 谁, <|assistant|>, 我, 是, 助, 手]
    #   labels:    [-100, -100, -100, -100, 我, 是, 助, 手]
    #   → 模型只需要学习：当看到 "...<|assistant|>" 后，生成 "我是助手"
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    # --- 步骤 7: 截断到 max_length ---
    # 如果序列太长（超过模型最大上下文），截断
    # input_ids 和 labels 必须同步截断，保持一一对应
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    # --- 步骤 8: attention_mask ---
    # 当前所有 token 都是真实内容，所以全设为 1
    # 后面 DataCollatorForSeq2Seq 会动态 pad 短序列，pad 位置的 mask 设为 0
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# 模型加载 + LoRA 配置
# ============================================================

# def main() 开头的 tokenizer/model 加载部分也在讲同一件事
# 这里拆开讲是为了让你理解每一步

# --- 步骤 1: 加载 tokenizer ---
# tokenizer 负责"文本 ↔ token ids"的互相转换
# AutoTokenizer.from_pretrained 会自动下载/加载 tokenizer 配置
# trust_remote_code=True：Qwen 用了自定义 tokenizer 代码，必须开启

# --- 步骤 2: 设置 pad_token ---
# 有些模型（如 Qwen）没有定义 pad_token
# pad_token 用于 batch 训练时对齐不同长度的序列
# 一般设为 eos_token（结束标记）即可

# --- 步骤 3: 加载模型 ---
# AutoModelForCausalLM.from_pretrained 自动下载/加载模型权重
# torch_dtype=torch.bfloat16：用 bfloat16 精度（省一半显存，精度损失很小）
#   为什么用 bfloat16 不用 float16？
#   bfloat16 数值范围和 float32 一样，不容易溢出
#   float16 数值范围小，训练时容易出现 inf/nan
# device_map="auto"：自动把模型分配到可用的 GPU 上
#   RTX 4050 有 6GB 显存，Qwen2.5-0.5B 大约需要 1GB（bfloat16）

# --- 步骤 4: 关闭 use_cache ---
# use_cache = True 时，推理会缓存 KV（Key-Value），加速自回归生成
# 但训练时不需要这个缓存（训练是 teacher forcing，一次前向传播）
# 而且开启缓存会多占显存

# --- 步骤 5: 启用输入梯度 ---
# enable_input_require_grads() 让模型输入也参与梯度计算
# 为什么需要？LoRA 层的输入是原始模型层的输出
# 如果原始模型层的输出不需要梯度，LoRA 层就没法反向传播
# 这是 PEFT 库的特殊要求

# --- 步骤 6: 配置 LoRA ---
# LoraConfig 的每个参数含义:
#   task_type=TaskType.CAUSAL_LM: 因果语言模型（GPT 类自回归模型）
#   r=8: rank，低秩矩阵的维度。r 越大参数越多，表达能力越强
#   lora_alpha=16: 缩放因子。实际缩放 = alpha/r = 16/8 = 2
#   lora_dropout=0.05: LoRA 层的 dropout 比率，防止过拟合
#   target_modules: 对哪些层加 LoRA
#     - q_proj, k_proj, v_proj, o_proj: 注意力层的 Q/K/V/Output 投影
#     - gate_proj, up_proj, down_proj: FFN 层的三个投影
#   LoRA 公式: h = Wx + (alpha/r) * BAx
#     W: 原始权重（冻结，不训练）
#     A: d×r 矩阵（训练）
#     B: r×d 矩阵（训练）
#     alpha/r: 缩放因子

# --- 步骤 7: get_peft_model ---
# get_peft_model(model, lora_config) 做了什么？
# 1. 冻结原始模型所有参数（requires_grad = False）
# 2. 在 target_modules 指定的层后面插入 LoRA 层（A 和 B 矩阵）
# 3. 返回一个 PeftModel，只有 LoRA 参数可训练
# model.print_trainable_parameters() 打印可训练参数占比
# 预期: ~4.4M 可训练参数 / ~498M 总参数 ≈ 0.88%


# ============================================================
# 训练主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on Chinese medical SFT data.")
    # --- 命令行参数 ---
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/medical/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/medical/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--max-length", type=int, default=1024,    # 序列最大长度（token 数）
                        help="超过此长度的序列会被截断")
    parser.add_argument("--epochs", type=float, default=1.0,       # 训练轮数
                        help="1.0 = 看一遍全部数据")
    parser.add_argument("--batch-size", type=int, default=1,       # 每张卡的 batch size
                        help="RTX 4050 6GB 显存，只能设 1")
    parser.add_argument("--grad-accum", type=int, default=8,       # 梯度累积步数
                        help="batch_size=1 + grad_accum=8 等效于 batch_size=8")
    parser.add_argument("--lr", type=float, default=2e-4,          # 学习率
                        help="LoRA 用 2e-4，比全参微调大（因为只调少量参数）")
    parser.add_argument("--max-train-samples", type=int, default=0,# 限制训练数据量
                        help="0 = 用全部数据，>0 = 只用前 N 条（调试用）")
    parser.add_argument("--max-valid-samples", type=int, default=0,# 限制验证数据量
                        help="0 = 用全部数据")
    args = parser.parse_args()

    # ============================================================
    # 步骤 1: 加载 tokenizer 和模型
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad_token，用 eos 代替

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # bfloat16: 省显存，精度损失小。RTX 30/40 系列支持 bfloat16
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto": 自动分配到 GPU，如果显存不够会自动卸载到 CPU
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False       # 训练时关闭 KV 缓存（省显存，训练不需要）
    model.enable_input_require_grads()   # 让输入张量参与梯度计算（PEFT 需要）

    # ============================================================
    # 步骤 2: 配置 LoRA
    # ============================================================
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,    # 因果语言模型
        r=4,                             # rank：低秩矩阵维度
        lora_alpha=8,                   # 缩放因子：16/8 = 2
        lora_dropout=0.05,               # dropout 防过拟合
        target_modules=[                 # 对哪些线性层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层 4 个投影
            "gate_proj", "up_proj", "down_proj",      # FFN 层 3 个投影
        ],
    )
    model = get_peft_model(model, lora_config)    # 把 LoRA 挂到模型上
    model.print_trainable_parameters()            # 打印可训练参数占比（预期 ~0.88%）

    # ============================================================
    # 步骤 3: 加载数据
    # ============================================================
    train_items = load_jsonl(args.train_file)
    valid_items = load_jsonl(args.valid_file)

    # 如果指定了 max_samples，只取前 N 条（调试用，全量训练太慢）
    if args.max_train_samples > 0:
        train_items = train_items[: args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_items = valid_items[: args.max_valid_samples]

    # ============================================================
    # 步骤 4: 构建 Dataset
    # ============================================================
    # Dataset.from_list(): 把 list[dict] 转成 HuggingFace Dataset 对象
    # .map(): 对每条数据调用 preprocess 函数，把原始数据转成 token 序列
    #   num_proc: 可以设多进程加速，但 Windows 上有时有问题，默认 1 就行
    # remove_columns: 删除原始列（instruction/input/output），只保留 token 序列
    #   为什么要删？Trainer 默认会把所有列传给 model.forward()
    #   但 model.forward() 只认识 input_ids/labels/attention_mask，不认识 instruction
    train_dataset = Dataset.from_list(train_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(train_items[0].keys()),
    )
    valid_dataset = Dataset.from_list(valid_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(valid_items[0].keys()),
    )

    # ============================================================
    # 步骤 5: 配置 TrainingArguments
    # ============================================================
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),               # checkpoint 保存目录
        num_train_epochs=args.epochs,                   # 训练轮数
        per_device_train_batch_size=args.batch_size,    # 每张卡的训练 batch size
        per_device_eval_batch_size=args.batch_size,     # 每张卡的评估 batch size
        gradient_accumulation_steps=args.grad_accum,    # 梯度累积步数
        # 梯度累积原理:
        #   batch_size=1, grad_accum=8
        #   每次前向传播 1 条数据，累积 8 次梯度后才更新一次参数
        #   效果等同于 batch_size=8，但只需要 1/8 的显存
        learning_rate=args.lr,                          # 学习率
        # 为什么 LoRA 用 2e-4 而不是常见的 5e-5？
        # 因为 LoRA 只训练少量参数（~0.88%），可以用更大的学习率
        # 全参微调用大学习率会破坏原始权重
        logging_steps=10,            # 每 10 步打印一次 train loss
        save_steps=100,              # 每 100 步保存一次 checkpoint
        eval_strategy="steps",       # 按步数评估（还有 "epoch" 选项）
        eval_steps=100,              # 每 100 步在验证集上评估一次
        save_strategy="steps",       # 按步数保存
        save_total_limit=2,          # 最多保留 2 个 checkpoint（节省磁盘）
        bf16=torch.cuda.is_available(),  # CUDA 可用时开启 bfloat16 混合精度
        fp16=False,                  # 不用 float16（bfloat16 更稳定）
        report_to="none",            # 不上报到 wandb 等平台
        remove_unused_columns=False, # 保留所有列（因为我们自定义了 input_ids/labels）
        # 为什么不删？默认 Trainer 会删除模型 forward 不需要的列
        # 但我们已经手动构建了 input_ids/labels/attention_mask
        # 如果 Trainer 再删一次会出错
    )

    # ============================================================
    # 步骤 6: 创建 Trainer
    # ============================================================
    trainer = Trainer(
        model=model,                # 要训练的模型（已经挂了 LoRA）
        args=training_args,         # 训练参数
        train_dataset=train_dataset,    # 训练数据
        eval_dataset=valid_dataset,     # 验证数据
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        # DataCollatorForSeq2Seq 的作用:
        #   同一批次中不同样本长度不同（比如一个 200 token，一个 300 token）
        #   需要 pad 到本批次中最长的长度
        #   它会:
        #   1. 动态 pad input_ids（用 pad_token_id）
        #   2. 动态 pad labels（用 -100，这样 pad 位置不计算 loss）
        #   3. 动态生成 attention_mask（pad 位置为 0，真实 token 为 1）
        #   为什么用 DataCollatorForSeq2Seq 而不是 DataCollatorForLanguageModeling？
        #   因为 Seq2Seq 版本会正确处理 labels 的 padding（用 -100）
    )

    # ============================================================
    # 步骤 7: 开始训练
    # ============================================================
    # trainer.train() 做了什么？
    # 1. 把数据按 batch_size 分批
    # 2. 每批: forward → 计算 loss → backward → 累积梯度
    # 3. 每 grad_accum 步: 更新参数（optimizer.step）
    # 4. 每 eval_steps 步: 在验证集上评估 loss
    # 5. 每 save_steps 步: 保存 checkpoint
    # 6. 每 logging_steps 步: 打印 train loss
    trainer.train()

    # ============================================================
    # 步骤 8: 保存最终结果
    # ============================================================
    # save_pretrained 只保存 LoRA adapter 权重（A 和 B 矩阵）
    # 不保存原始模型权重（因为原始权重冻结了，没变过）
    # 所以 adapter 文件很小（几 MB），而原始模型是几百 MB
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 保存 tokenizer 是为了推理时能直接从 output_dir 加载一切
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
