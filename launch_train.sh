#!/bin/bash
# launch_train.sh - StructDiff训练启动脚本

set -e

# 默认参数
CONFIG_FILE="configs/test_train.yaml"
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=1
RESUME=""
MASTER_PORT=29500

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "StructDiff训练启动脚本"
            echo ""
            echo "使用方法:"
            echo "  $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config CONFIG_FILE     配置文件路径 (默认: configs/test_train.yaml)"
            echo "  --output_dir OUTPUT_DIR  输出目录 (默认: outputs/TIMESTAMP)"
            echo "  --num_gpus NUM_GPUS      GPU数量 (默认: 1)"
            echo "  --resume CHECKPOINT      恢复训练的检查点路径"
            echo "  --master_port PORT       分布式训练的主端口 (默认: 29500)"
            echo "  -h, --help               显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 单GPU训练"
            echo "  $0 --config configs/test_train.yaml"
            echo ""
            echo "  # 多GPU训练 (2个GPU)"
            echo "  $0 --config configs/test_train.yaml --num_gpus 2"
            echo ""
            echo "  # 恢复训练"
            echo "  $0 --config configs/test_train.yaml --resume outputs/checkpoints/checkpoint_latest.pth"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🚀 启动 StructDiff 训练"
echo "📝 配置文件: $CONFIG_FILE"
echo "📁 输出目录: $OUTPUT_DIR"
echo "🔧 GPU数量: $NUM_GPUS"

if [[ -n "$RESUME" ]]; then
    echo "🔄 恢复训练: $RESUME"
fi

# 构建训练命令
TRAIN_CMD="python train_full.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR"

if [[ -n "$RESUME" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

# 根据GPU数量选择训练方式
if [[ $NUM_GPUS -eq 1 ]]; then
    echo "🎯 启动单GPU训练..."
    echo "命令: $TRAIN_CMD"
    echo ""
    
    # 设置CUDA可见设备
    export CUDA_VISIBLE_DEVICES=0
    
    # 执行训练
    $TRAIN_CMD
    
elif [[ $NUM_GPUS -gt 1 ]]; then
    echo "🚀 启动多GPU分布式训练..."
    
    # 检查可用GPU数量
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    if [[ $NUM_GPUS -gt $AVAILABLE_GPUS ]]; then
        echo "警告: 请求的GPU数量($NUM_GPUS)超过可用GPU数量($AVAILABLE_GPUS)"
        echo "使用可用的GPU数量: $AVAILABLE_GPUS"
        NUM_GPUS=$AVAILABLE_GPUS
    fi
    
    # 构建分布式训练命令
    DIST_CMD="python -m torch.distributed.launch"
    DIST_CMD="$DIST_CMD --nproc_per_node=$NUM_GPUS"
    DIST_CMD="$DIST_CMD --master_port=$MASTER_PORT"
    DIST_CMD="$DIST_CMD train_full.py"
    DIST_CMD="$DIST_CMD --config $CONFIG_FILE"
    DIST_CMD="$DIST_CMD --output_dir $OUTPUT_DIR"
    
    if [[ -n "$RESUME" ]]; then
        DIST_CMD="$DIST_CMD --resume $RESUME"
    fi
    
    echo "命令: $DIST_CMD"
    echo ""
    
    # 执行分布式训练
    $DIST_CMD
    
else
    echo "错误: GPU数量必须大于0"
    exit 1
fi

echo ""
echo "🎉 训练完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📊 查看TensorBoard日志:"
echo "   tensorboard --logdir $OUTPUT_DIR/tensorboard"
echo "📋 查看训练日志:"
echo "   tail -f $OUTPUT_DIR/logs/training_*.log" 