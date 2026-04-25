# 真正的后台运行B2Z1训练 + 生成视频 - 完整指南

## 问题回顾
之前运行的 `python scripts/auto_train.py ...` 不是真正的后台运行。
当SSH session关闭时，训练也停止了。

## ✅ 解决方案：使用nohup后台运行

---

## 第一次：启动后台训练

### 步骤1: 连接到远程服务器
```bash
ssh root@ry3.9gpu.com -p 11092
# 输入密码: WftpaCCs
```

### 步骤2: 运行后台启动脚本
```bash
bash /root/RoboDuet/start_training_background.sh
```

**输出示例:**
```
==========================================================================
B2Z1 Training - Proper Background Execution
==========================================================================

[1/3] Starting training in background...
Command: python scripts/auto_train.py --robot b2z1 --num_envs 2048 ...

[2/3] Using nohup to ensure process survives SSH disconnect...
      Training started with PID: 12345
      Log file: /root/RoboDuet/logs/b2z1_training.log

[3/3] Verification successful!
      Process is running: YES

==========================================================================
TRAINING IS NOW RUNNING IN BACKGROUND!
==========================================================================

You can now:
  1. Close this SSH session (Ctrl+D or 'exit')
  2. The training will CONTINUE running on the server
  3. Open a NEW SSH session to check progress or generate video
```

### 步骤3: 安全地关闭SSH session
```bash
exit
# 或 Ctrl+D
```

**重要:** 现在训练会**继续运行**，即使SSH断开！

---

## 后续操作：监控和生成视频

### 打开新的SSH session（不影响训练）

```bash
ssh root@ry3.9gpu.com -p 11092
```

---

### 选项A：监控训练进度

```bash
# 实时查看训练日志
tail -f /root/RoboDuet/logs/b2z1_training.log
```

按 `Ctrl+C` 停止日志查看。

**输出示例:**
```
[INFO] Iteration 1000/50000 - Loss: 0.123 - Reward: 0.456
[INFO] Iteration 2000/50000 - Loss: 0.089 - Reward: 0.512
[INFO] Iteration 3000/50000 - Loss: 0.067 - Reward: 0.634
...
```

### 选项B：检查训练是否还在运行

```bash
ps aux | grep auto_train.py
```

应该能看到类似的输出（不是 `grep auto_train`）：
```
root    12345  95.2 45.3 987654 123456 ?  Rl  14:30  12:45 python scripts/auto_train.py ...
```

这表示训练正在运行。

### 选项C：查看GPU使用情况

```bash
nvidia-smi

# 或者每2秒更新一次
watch -n 2 nvidia-smi
```

---

### 生成Simulation视频（训练继续运行）

在**同一个新session**中（或再打开一个新session）：

```bash
cd /root/RoboDuet
python3 gen_video_from_checkpoint.py
```

这个脚本会：
1. 找到最新保存的模型checkpoint
2. 加载模型
3. 运行600步IsaacGym simulation
4. 生成MP4视频

**完全不会影响正在运行的训练！**

---

## 完整时间表示例

```
时间          操作                                状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
14:00    ssh连接 + 启动后台训练脚本           ✓ 训练启动
14:01    关闭SSH (exit)                      ✓ 训练继续进行
14:05    打开新SSH + 查看日志                 ✓ 训练在运行
14:10    同一session中运行gen_video脚本       ✓ 训练 + 生成视频
14:20    视频生成完成                        ✓ 训练仍在继续
14:21    下载MP4到本地                       ✓ 训练仍在继续
...
20:00    训练完成 (50000迭代)                 ✓ 完成
```

---

## 关键要点

| 项 | 说明 |
|---|---|
| **后台运行** | 使用 `nohup` 确保SSH断开后继续运行 |
| **日志** | 保存在 `/root/RoboDuet/logs/b2z1_training.log` |
| **监控** | 新SSH session 查看 `tail -f logs/...log` |
| **生成视频** | 同时在另一个session运行，不影响训练 |
| **关闭训练** | 用 `kill PID` 或直接重启服务器 |

---

## 常见问题

### Q1: 如何确认训练真的在后台运行?

```bash
# 新session中检查
ps aux | grep auto_train.py

# 或查看log文件
tail -20 /root/RoboDuet/logs/b2z1_training.log
```

### Q2: 训练会运行多长时间?

50000次迭代，2048个环境：
- 预计：**8-12小时**（取决于GPU性能）
- 会自动保存checkpoint到 `/root/RoboDuet/runs/`

### Q3: 可以在训练中途停止吗?

```bash
ps aux | grep auto_train.py
# 找到PID (第二列)

kill <PID>   # 优雅地停止
# 或
kill -9 <PID>   # 强制停止
```

### Q4: 如何查看已保存的checkpoint?

```bash
ls -lh /root/RoboDuet/runs/b2z1_training_v1_rtx4090/*/checkpoints_dog/
```

### Q5: 视频生成时训练会变慢吗?

**可能会略微变慢**，因为都用GPU。但:
- 训练占用90%+ GPU资源
- 视频生成占用10%左右
- 两者可以共存

---

## 故障排查

### 训练启动失败?

```bash
# 检查错误日志
tail -50 /root/RoboDuet/logs/b2z1_training.log

# 检查IsaacGym是否正确安装
python3 -c "import isaacgym; print('OK')"

# 检查CUDA
nvidia-smi
```

### 找不到checkpoint?

```bash
find /root/RoboDuet/runs -name "ac_weights_*.pt" -type f | head -5
```

### 视频生成失败?

确保训练已经保存了至少一个checkpoint:
```bash
# 等待几分钟让训练保存第一个checkpoint
sleep 300
python3 gen_video_from_checkpoint.py
```

---

## 推荐工作流

### 第一天

```
1. ssh连接
2. 运行 start_training_background.sh
3. exit 关闭SSH
4. 去忙其他事情
```

### 几小时后 (任何时间)

```
1. ssh连接
2. tail -f logs/b2z1_training.log (查看进度)
3. 在另一个window运行 gen_video_from_checkpoint.py
4. 等待视频生成
5. 下载MP4
```

### 添加到PPT

```
1. 打开 PhD_Briefing_B2Z1_Grasping_WITH_CHART.pptx
2. Slide 6 (或新增slide)
3. Insert → Video → 选择下载的MP4
4. 完成!
```

---

## 快速命令参考

**启动后台训练:**
```bash
bash /root/RoboDuet/start_training_background.sh
```

**监控进度:**
```bash
tail -f /root/RoboDuet/logs/b2z1_training.log
```

**生成视频:**
```bash
cd /root/RoboDuet && python3 gen_video_from_checkpoint.py
```

**下载视频:**
```bash
scp -P 11092 root@ry3.9gpu.com:~/.local/share/isaacgym/*.mp4 .
```

**检查进程:**
```bash
ps aux | grep auto_train.py
```

**停止训练:**
```bash
kill <PID>
```

---

## 总结

现在你有了一个**真正的后台训练系统**：

✅ 训练在后台持续运行  
✅ SSH断开后继续运行  
✅ 可以随时监控进度  
✅ 可以同时生成视频  
✅ 日志保存便于追踪  

祝训练顺利! 🚀
