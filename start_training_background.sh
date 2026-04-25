#!/bin/bash
# 真正的后台运行B2Z1训练 - 即使关闭SSH也继续运行
# 使用 nohup + screen/tmux

set -e

echo "=========================================================================="
echo "B2Z1 Training - Proper Background Execution"
echo "=========================================================================="
echo ""

# Configuration
TRAINING_CMD="python scripts/auto_train.py --robot b2z1 --num_envs 2048 --num_learning_iterations 50000 --run_name b2z1_training_v1_rtx4090 --sim_device cuda:0 --headless"
LOG_DIR="/root/RoboDuet/logs"
LOG_FILE="${LOG_DIR}/b2z1_training.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "[1/3] Starting training in background..."
echo "Command: $TRAINING_CMD"
echo ""

# Method 1: Using nohup (simplest, works even if you exit SSH)
echo "[2/3] Using nohup to ensure process survives SSH disconnect..."
cd /root/RoboDuet

# Start training with nohup - redirects output to log file
nohup $TRAINING_CMD > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "      Training started with PID: $TRAIN_PID"
echo "      Log file: $LOG_FILE"
echo ""

# Give it a moment to start
sleep 2

# Verify process is running
if ps -p $TRAIN_PID > /dev/null; then
    echo "[3/3] Verification successful!"
    echo "      Process is running: YES"
    echo ""
    echo "=========================================================================="
    echo "TRAINING IS NOW RUNNING IN BACKGROUND!"
    echo "=========================================================================="
    echo ""
    echo "You can now:"
    echo "  1. Close this SSH session (Ctrl+D or 'exit')"
    echo "  2. The training will CONTINUE running on the server"
    echo "  3. Open a NEW SSH session to check progress or generate video"
    echo ""
    echo "=========================================================================="
    echo "HOW TO MONITOR TRAINING:"
    echo "=========================================================================="
    echo ""
    echo "In a new SSH session:"
    echo "  tail -f $LOG_FILE              # Watch training progress in real-time"
    echo "  ps aux | grep auto_train.py    # Check if process is still running"
    echo "  nvidia-smi                     # Check GPU usage"
    echo ""
    echo "=========================================================================="
    echo "HOW TO GENERATE VIDEO (while training continues):"
    echo "=========================================================================="
    echo ""
    echo "In a DIFFERENT new SSH session:"
    echo "  cd /root/RoboDuet"
    echo "  python3 gen_video_from_checkpoint.py"
    echo ""
    echo "=========================================================================="
    
else
    echo "[ERROR] Process failed to start!"
    echo "Check log file: $LOG_FILE"
    cat "$LOG_FILE" | tail -20
    exit 1
fi
