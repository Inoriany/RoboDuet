# MobaXterm 生成B2Z1 Simulation视频 - 快速指南

## 前提条件
- MobaXterm SSH连接到 `ry3.9gpu.com:11092`
- 远程已有训练模型 (`/root/RoboDuet/runs/`)
- 已上传 `/root/RoboDuet/gen_video.py` 脚本

## 步骤1: 在MobaXterm中连接到远程服务器

```bash
ssh root@ry3.9gpu.com -p 11092
# 输入密码: WftpaCCs
```

## 步骤2: 进入RoboDuet目录

```bash
cd /root/RoboDuet
pwd  # 验证位置
```

## 步骤3: 检查已有的训练模型

```bash
# 列出所有模型检查点
find runs -name "ac_weights_*.pt" -type f | sort -V | tail -5

# 输出示例:
# /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/033037.985148_seed1102/checkpoints_dog/ac_weights_last_dog.pt
# /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/034214.230755_seed513/checkpoints_arm/ac_weights_020800.pt
```

## 步骤4: 运行视频生成脚本

### 方案A: 基本用法（推荐）
```bash
# 生成10秒视频 (600帧 @60fps)
python3 gen_video.py --steps 600

# 脚本会:
# 1. 自动找到最新训练的模型
# 2. 启动IsaacGym环境（带rendering）
# 3. 运行B2Z1机器人在模拟器中
# 4. 保存视频
```

### 方案B: 自定义步数
```bash
# 生成5秒视频
python3 gen_video.py --steps 300

# 生成20秒视频
python3 gen_video.py --steps 1200

# 生成30秒视频
python3 gen_video.py --steps 1800
```

### 方案C: 指定输出路径
```bash
python3 gen_video.py --steps 600 --output /tmp/b2z1_long_demo.mp4
```

### 方案D: 设置随机种子（保证可重现）
```bash
python3 gen_video.py --steps 600 --seed 42
```

## 步骤5: 等待完成

脚本运行期间会显示进度:
```
========================================================================
B2Z1 SIMULATION VIDEO GENERATOR
========================================================================

[1/4] Loading IsaacGym environment...
      [OK] Environment class loaded

[2/4] Finding trained model...
      [OK] Found latest model: /root/RoboDuet/runs/.../ac_weights_020800.pt
           Modified: Wed Mar 25 14:46:31 2026

[3/4] Initializing B2Z1 environment (headless=False for rendering)...
      [OK] Environment initialized
      Observation shape: (48,)
      Action shape: (18,)

[4/4] Running 600 steps of simulation...
      This will generate ~10.0 seconds of video
      Rendering to: /root/RoboDuet/b2z1_demo.mp4
      
      Frame 0/600 (0%) - Reward: 0.125
      Frame 60/600 (10%) - Reward: 0.347
      Frame 120/600 (20%) - Reward: 0.512
      ...
      Frame 540/600 (90%) - Reward: 0.823
      Frame 600/600 (100%) - Reward: 0.891

[OK] Simulation completed!
[OK] Video saved at: /root/RoboDuet/b2z1_demo.mp4
     Size: 45.3 MB
```

## 步骤6: 找到生成的视频

视频通常保存在以下位置之一:

```bash
# 检查主要输出路径
ls -lh /root/RoboDuet/b2z1_demo.mp4

# 如果找不到，检查其他可能的位置
find /root/RoboDuet -name "*.mp4" -type f -mmin -5  # 最后5分钟修改的MP4

find /tmp -name "*.mp4" -type f -mmin -5

ls -lh ~/.local/share/isaacgym/videos/
```

## 步骤7: 下载视频到本地Windows

```bash
# 在MobaXterm中按 Ctrl+S 或使用菜单 "Sftp" 打开SFTP窗口

# 或者在MobaXterm中用以下命令:
scp b2z1_demo.mp4 jacky@192.168.1.x:/path/to/local/folder/

# 或者用 MobaXterm 的拖放功能
# 在左侧文件树中找到 /root/RoboDuet/b2z1_demo.mp4
# 拖放到Windows资源管理器
```

## 步骤8: 插入到PPT

```
1. 打开 D:\CUHK\AIMS_5790\PhD_Briefing_B2Z1_Grasping_WITH_CHART.pptx
2. 添加新slide (或用Slide 6)
3. Insert → Video → 选择下载的 b2z1_demo.mp4
4. 调整大小和位置
5. 保存
```

## 常见问题排查

### Q1: 脚本找不到模型
**解决方案:**
```bash
# 手动指定模型路径
# 编辑 gen_video.py 中的 find_latest_model() 函数
# 或直接在命令行传入

# 检查模型是否存在
ls -lh /root/RoboDuet/runs/*/checkpoints*/*weights*.pt | head -5
```

### Q2: IsaacGym环境加载失败
**解决方案:**
```bash
# 检查IsaacGym是否安装
python3 -c "from isaacgym import gymapi; print(gymapi.__file__)"

# 如果失败,尝试:
cd /root/RoboDuet
pip install -e .

# 或查看训练日志
cd /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/*/
cat output.log | tail -50
```

### Q3: 没有GPU内存
**解决方案:**
```bash
# 运行更少的步数
python3 gen_video.py --steps 300

# 或减少环境数量 (编辑脚本中的 num_envs=1)
```

### Q4: 视频太大
**解决方案:**
```bash
# 生成更短的视频
python3 gen_video.py --steps 300  # 5秒而不是10秒

# 压缩视频 (如果安装了ffmpeg)
ffmpeg -i b2z1_demo.mp4 -crf 23 b2z1_demo_compressed.mp4
```

### Q5: 想要多个视频用于对比
**解决方案:**
```bash
# 使用不同的随机种子
python3 gen_video.py --steps 600 --seed 42 --output b2z1_demo_seed42.mp4
python3 gen_video.py --steps 600 --seed 123 --output b2z1_demo_seed123.mp4
python3 gen_video.py --steps 600 --seed 999 --output b2z1_demo_seed999.mp4
```

## 高级用法: 自定义脚本

如果想修改脚本（例如改变渲染设置、添加更多调试信息）:

```bash
# 编辑脚本
nano /root/RoboDuet/gen_video.py

# 保存并退出: Ctrl+X, Y, Enter

# 重新运行
python3 gen_video.py --steps 600
```

## 总结

完整的命令序列:
```bash
ssh root@ry3.9gpu.com -p 11092          # 连接到远程
cd /root/RoboDuet                        # 进入工作目录
python3 gen_video.py --steps 600         # 生成视频
ls -lh b2z1_demo.mp4                     # 验证文件
```

预计时间: **10-15分钟** (包括加载环境、运行模拟、保存视频)

---

**需要帮助?** 检查日志输出或在MobaXterm中运行:
```bash
python3 gen_video.py --help
```
