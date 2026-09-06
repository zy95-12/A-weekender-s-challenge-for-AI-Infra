# GSM8K 功能与同输入 logits 验证

执行命令：`./poc validate-gsm8k --output results/gsm8k-logits`。整条自动流程退出码0，完成后保留 optimized 服务。测试日期：2026-09-06。

数据集：[openai/grade-school-math](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data)，固定 seed42 的8个 test indices：51、209、228、285、457、501、563、1309。train 前16题作为固定上下文，仅用于覆盖切块续接；不使用 test 参考答案。输入长度3268–3319 tokens。测试范围是这个预先固定的代表性子集，不是全1319题。

Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，FP16，vLLM0.10.2 V0，4张A10。split 为企业前4/云27/企业后5层，WAN每方向10Gbps、单向5ms。原生单体包含全部36层。

## 功能

baseline 和 optimized 各8条 GSM8K 问题通过公共 `/v1/completions` 请求，C8，max_tokens32；HTTP成功、输出非空、usage输入长度一致，结束后active/waiting/KV归零。流式chat的真实curl也收到文本和`[DONE]`，原始SSE保存供检查。

这里只要求功能跑通，不验收题目答案正确率、不要求单体与split自由回答一致。先前64题的答案观察不进入本报告的通过条件。

## 同输入、同绝对位置的数值结果

| split 配置 / 对照单体配置 | 阶段 | 全词表行数 | 最大逐位置 MAE | 最大绝对误差 | 最大 softmax TV |
|---|---|---:|---:|---:|---:|
| 全关 TP2+2 / 单体 TP2 full-prefill | prefill末位置 | 8 | 0 | 0 | 0 |
| 全关 TP2+2 / 单体 TP2 full-prefill | decode | 56 | 0 | 0 | 0 |
| 双TP1 P + TP1 D / 单体 TP1 chunk2048 | prefill末位置 | 8 | 0 | 0 | 0 |
| 双TP1 P + TP1 D / 单体 TP1 chunk2048 | decode | 56 | 0 | 0 | 0 |

每行词表大小151936；上述所有元素逐值一致，RMSE与最大概率差也为0。cosine的最小浮点计算结果为0.9999999999999998。两个P都参与实际验证：trace中8条teacher-forcing请求分别由P0/P1承担4条，整段prompt KV迁移后在D执行decode。

例如 test index51，prompt长度3291：

| 配置 | phase | token_id | absolute_position | num_computed_tokens | query_len | hidden row | logits row | top1 token | top1 logit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP2 full-prefill 原生与split | prefill | 198 | 3290 | 0 | 3291 | 3290 | 0 | 5338 | 27.0625 |
| TP1 chunk2048 原生与split | prefill | 198 | 3290 | 2048 | 1243 | 1242 | 0 | 5338 | 27.046875 |
| 两组原生与对应split | 首个decode | 5338 | 3291 | 3291 | 1 | 0 | 0 | 11 | 29.859375 |

原生读取实际sampler选中的hidden rows，split读取实际送往LM head的rows。先验证输入token和绝对位置，再计算误差；没有为验收将原生LM head改成全位置输出。prefill只验收prompt末位置，未对中间未输出logits的chunk虚构数据。decode使用对应原生生成的token做teacher forcing，避免输入分叉。

两个原生配置之间本身可能因TP、chunk形状和attention路径不同而有数值差异（表中prefill top1 logit即有差异）；本测试将各split与相同数值配置的单体配对，不宣称TP2/full与TP1/chunk也逐值相等。

预设门槛为逐位置MAE≤0.01、RMSE≤0.02、最大误差≤0.1、cosine≥0.9999；任一位置不通过即失败。top1与softmax作为观察项，回答是否相同不参与判定。本次结果比门槛更严格，为逐元素一致。

## 控制通道关闭回退

另将最优预设加上`--no-pd-control-channel`，选择上述固定样本中的前两题，不重新选题。公共接口生成成功；2个prefill行和14个decode行仍与对应原生逐元素一致，max error/softmax TV为0，排空通过。随后用一条`./poc up --preset optimized --wan`恢复推荐配置。该实验验证开关可用，不作为关闭通道后的吞吐收益结论。

## 复算

证据目录：[gsm8k-integration](evidence/issue-6/gsm8k-integration/)。包含summary、逐position记录、固定样本manifest、CPU测试日志、API输出，以及两份完整原始logits ZIP和复算源码/数据ZIP。SHA256清单覆盖所有附件；ZIP CRC及成员哈希逐项验证。压缩不会改变float32数组。

将`baseline_logits.zip`、`optimized_logits.zip`解压到同一个目录，再从`reproduce.zip`的evaluation/复制manifest.json和prompts.json到该目录，然后在源码目录执行：

```bash
python scripts/gsm8k_validate.py --phase compare --output /path/to/extracted-evaluation
```

离线compare只需Python/numpy，不需要GPU或模型权重；重新运行native/split才需要完整环境。使用`--limit 0`可扩展全测试集，但当前不声称已完成全量验证。
