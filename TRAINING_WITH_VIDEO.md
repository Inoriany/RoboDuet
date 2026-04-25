# 在运行训练时生成Simulation视频 - 完整指南

## 场景: 训练正在后台运行

你已经在某个MobaXterm session中运行:
```bash
python scripts/auto_train.py --robot b2z1 --num_envs 2048 --num_learning_iterations 50000 \
  --run_name b2z1_training_v1_rtx4090 --sim_device cuda:0 --headless
```

现在想生成一个simulation视频而**不停止训练**。

---

## ✅ 解决方案: 使用新的SSH Session

### 步骤1: 打开新的MobaXterm SSH Session

在MobaXterm中:
1. **Ctrl+Shift+S** (或菜单 Session → New Session)
2. 选择 SSH
3. 连接信息:
   - Host: `ry3.9gpu.com`
   - Port: `11092`
   - Username: `root`
   - Password: `WftpaCCs`

### 步骤2: 在新Session中运行视频生成脚本

```bash
cd /root/RoboDuet
python3 gen_video_from_checkpoint.py
```

这个脚本会:
1. ✅ 自动找到**最新保存的模型检查点**
2. ✅ 加载模型（不重新训练）
3. ✅ 启用IsaacGym rendering
4. ✅ 运行600步模拟（10秒@60fps）
5. ✅ 自动保存MP4视频

### 步骤3: 等待完成

输出示例:
```
======================================================================
B2Z1 VIDEO GENERATOR - From Existing Checkpoint
======================================================================

[1/4] Importing libraries...
      [OK]

[2/4] Finding checkpoint...
      Found: ac_weights_020800.pt
      Modified: Wed Mar 25 14:46:31 2026
      Size: 48.2 MB

[3/4] Setting up environment...
      Device: cuda:0
      Obs shape: (48,)
      Action shape: (18,)

[4/4] Running simulation...
      Policy loaded
      Rendering 600 steps (10 seconds @60fps)...
      0% complete
      20% complete
      40% complete
      60% complete
      80% complete
      100% complete

[OK] Simulation complete!

Video should be saved at:
  ~/.local/share/isaacgym/
  or ~/Videos/

To find it:
  find ~ -name '*.mp4' -mmin -5
```

### 步骤4: 找到并下载视频

```bash
# 在新Session中查找视频
find ~ -name '*.mp4' -type f -mmin -5

# 输出示例:
# /root/.local/share/isaacgym/b2z1_video_20260325_144605.mp4
```

用SFTP或scp下载:
```bash
scp -P 11092 root@ry3.9gpu.com:~/.local/share/isaacgym/b2z1_video_*.mp4 .
```

---

## 关键要点

| 项目 | 说明 |
|------|------|
| **训练Session** | 继续运行，不受影响 ✅ |
| **新Session** | 只用于生成视频 |
| **GPU使用** | 2个Session可共享GPU（训练占大部分） |
| **时间** | 5-10分钟生成10秒视频 |
| **模型** | 自动读取最新checkpoint |

---

## 多个Session示意图

```
MobaXterm中同时有2个Session:

Session 1 (训练 - 继续运行):
└─ python scripts/auto_train.py --robot b2z1 ... --headless
   ├─ 占用2048个环境
   ├─ 使用大部分GPU
   └─ 持续保存checkpoint到 runs/

Session 2 (视频生成 - 新打开):
└─ python3 gen_video_from_checkpoint.py
   ├─ 读取最新的checkpoint
   ├─ 使用少量GPU进行rendering
   └─ 保存MP4视频
```

---

## 常见问题

### Q1: Session 2会不会影响训练?
**A:** 基本不会。两个session可以共用GPU。训练已经占用大部分资源，视频生成占用很少。

### Q2: 模型一直在更新，哪个会被用?
**A:** 脚本自动读取**最新修改时间**的checkpoint。所以会用最新的。

### Q3: 可以生成多个视频吗?
**A:** 可以！在Session 2中:
```bash
# 第一次
python3 gen_video_from_checkpoint.py

# 等待完成后，再运行一次
python3 gen_video_from_checkpoint.py
```

会生成不同的视频（因为checkpoint更新了）。

### Q4: 视频保存在哪?
**A:** 通常在:
- `~/.local/share/isaacgym/`
- 或 `~/Videos/`

用以下命令查找:
```bash
find ~ -name '*.mp4' -mmin -5  # 最后5分钟修改的MP4
```

### Q5: 如何找到生成的视频路径?
脚本会在输出的最后显示:
```
Video should be saved at:
  ~/.local/share/isaacgym/
  or ~/Videos/

To find it:
  find ~ -name '*.mp4' -mmin -5
```

运行那个find命令就能看到确切路径。

---

## 完整工作流

```
1. 训练正在运行 (Session 1) ← 保持不动
   python scripts/auto_train.py ... --headless

2. 打开新SSH Session (Session 2)
   ssh root@ry3.9gpu.com -p 11092

3. 运行视频生成脚本
   cd /root/RoboDuet
   python3 gen_video_from_checkpoint.py

4. 等待完成（5-10分钟）
   └─ 脚本自动找最新模型
   └─ IsaacGym rendering视频
   └─ 保存MP4

5. 下载视频到本地
   scp -P 11092 root@ry3.9gpu.com:~/.local/share/isaacgym/*.mp4 .

6. 添加到PPT
   Open PhD_Briefing_B2Z1_Grasping_WITH_CHART.pptx
   Slide 6 或新增slide
   Insert → Video → 选择MP4

7. 演讲时播放
   ✓ 完成!
```

---

## 注意事项

⚠️ **重要:**
- 两个session都连接到**同一个远程服务器**
- Session 1做训练，Session 2做视频生成
- 不要在Session 2中运行 `auto_train.py`（会冲突）
- 确保两个session的working directory都是 `/root/RoboDuet`

---

## 脚本位置

远程已上传脚本:
```
/root/RoboDuet/gen_video_from_checkpoint.py  ← 推荐使用
/root/RoboDuet/gen_video_v2.py
/root/RoboDuet/gen_video.py
```

---

## 快速命令

完整的一行代码（在新SSH Session中）:
```bash
ssh root@ry3.9gpu.com -p 11092 -t "cd /root/RoboDuet && python3 gen_video_from_checkpoint.py"
```

完成后找视频:
```bash
ssh root@ry3.9gpu.com -p 11092 -t "find ~ -name '*.mp4' -mmin -5"
```

下载视频:
```bash
scp -P 11092 root@ry3.9gpu.com:~/.local/share/isaacgym/b2z1_video_*.mp4 .
```

---

祝演讲成功! 🎯
