# monitor_training.py - 训练监控脚本
import argparse
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
    def get_latest_log_file(self):
        """获取最新的日志文件"""
        if not self.log_dir.exists():
            return None
        
        log_files = list(self.log_dir.glob("training_*.log"))
        if not log_files:
            return None
        
        return max(log_files, key=lambda f: f.stat().st_mtime)
    
    def parse_log_metrics(self, log_file, num_lines=50):
        """解析日志文件中的训练指标"""
        if not log_file or not log_file.exists():
            return {}
        
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'gpu_memory': [],
            'learning_rates': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 只读取最后num_lines行
            recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
            
            for line in recent_lines:
                line = line.strip()
                
                # 解析训练损失
                if "训练损失 - 总计:" in line:
                    try:
                        parts = line.split("训练损失 - 总计:")
                        if len(parts) > 1:
                            loss_str = parts[1].split(",")[0].strip()
                            loss = float(loss_str)
                            metrics['train_losses'].append(loss)
                    except:
                        pass
                
                # 解析验证损失
                elif "验证损失 - 总计:" in line:
                    try:
                        parts = line.split("验证损失 - 总计:")
                        if len(parts) > 1:
                            loss_str = parts[1].split(",")[0].strip()
                            loss = float(loss_str)
                            metrics['val_losses'].append(loss)
                    except:
                        pass
                
                # 解析Epoch信息
                elif "🚀 Epoch" in line:
                    try:
                        parts = line.split("Epoch")[1].split("/")
                        if len(parts) >= 2:
                            current_epoch = int(parts[0].strip())
                            metrics['epochs'].append(current_epoch)
                    except:
                        pass
        
        except Exception as e:
            print(f"解析日志文件错误: {e}")
        
        return metrics
    
    def get_tensorboard_metrics(self):
        """从TensorBoard日志中获取详细指标"""
        if not self.tensorboard_dir.exists():
            return {}
        
        metrics = {}
        
        try:
            # 查找TensorBoard事件文件
            event_files = list(self.tensorboard_dir.rglob("events.out.tfevents.*"))
            
            if not event_files:
                return metrics
            
            # 使用最新的事件文件
            latest_event_file = max(event_files, key=lambda f: f.stat().st_mtime)
            
            # 创建事件累积器
            ea = EventAccumulator(str(latest_event_file))
            ea.Reload()
            
            # 获取可用的标量标签
            scalar_tags = ea.Tags()['scalars']
            
            for tag in scalar_tags:
                try:
                    scalar_events = ea.Scalars(tag)
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    metrics[tag] = {'steps': steps, 'values': values}
                except:
                    pass
                    
        except Exception as e:
            print(f"读取TensorBoard数据错误: {e}")
        
        return metrics
    
    def get_checkpoint_info(self):
        """获取检查点信息"""
        info = {
            'latest_checkpoint': None,
            'best_checkpoint': None,
            'total_checkpoints': 0,
            'latest_epoch': 0,
            'best_loss': float('inf')
        }
        
        if not self.checkpoint_dir.exists():
            return info
        
        # 检查最新检查点
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        if latest_path.exists():
            info['latest_checkpoint'] = str(latest_path)
            try:
                checkpoint = torch.load(latest_path, map_location='cpu')
                info['latest_epoch'] = checkpoint.get('epoch', 0)
            except:
                pass
        
        # 检查最佳检查点
        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            info['best_checkpoint'] = str(best_path)
            try:
                checkpoint = torch.load(best_path, map_location='cpu')
                info['best_loss'] = checkpoint.get('loss', float('inf'))
            except:
                pass
        
        # 统计总检查点数
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        info['total_checkpoints'] = len(checkpoint_files)
        
        return info
    
    def get_gpu_status(self):
        """获取GPU状态信息"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 6:
                            gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_used': int(parts[2]),
                                'memory_total': int(parts[3]),
                                'utilization': int(parts[4]),
                                'temperature': int(parts[5])
                            })
                return gpu_info
        except:
            pass
        
        return []
    
    def print_status(self):
        """打印当前训练状态"""
        print("=" * 60)
        print("🚀 StructDiff 训练监控")
        print("=" * 60)
        
        # 检查点信息
        checkpoint_info = self.get_checkpoint_info()
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 当前Epoch: {checkpoint_info['latest_epoch']}")
        print(f"💾 检查点数量: {checkpoint_info['total_checkpoints']}")
        print(f"🏆 最佳损失: {checkpoint_info['best_loss']:.4f}")
        
        # 最近的训练指标
        log_file = self.get_latest_log_file()
        if log_file:
            print(f"📋 日志文件: {log_file.name}")
            metrics = self.parse_log_metrics(log_file)
            
            if metrics['train_losses']:
                recent_train_loss = metrics['train_losses'][-1]
                print(f"🔥 最近训练损失: {recent_train_loss:.4f}")
            
            if metrics['val_losses']:
                recent_val_loss = metrics['val_losses'][-1]
                print(f"✅ 最近验证损失: {recent_val_loss:.4f}")
        else:
            print("⚠️  未找到日志文件")
        
        # GPU状态
        gpu_status = self.get_gpu_status()
        if gpu_status:
            print("\n🖥️  GPU 状态:")
            for gpu in gpu_status:
                memory_usage = gpu['memory_used'] / gpu['memory_total'] * 100
                print(f"   GPU {gpu['index']}: {gpu['name']}")
                print(f"     内存: {gpu['memory_used']}/{gpu['memory_total']} MB ({memory_usage:.1f}%)")
                print(f"     利用率: {gpu['utilization']}%")
                print(f"     温度: {gpu['temperature']}°C")
        else:
            print("⚠️  无法获取GPU状态")
        
        print("=" * 60)
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        tensorboard_metrics = self.get_tensorboard_metrics()
        
        if not tensorboard_metrics:
            print("⚠️  未找到TensorBoard数据，无法绘制曲线")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('StructDiff 训练监控', fontsize=16)
        
        # 训练损失曲线
        if 'Train/Loss_Total' in tensorboard_metrics:
            data = tensorboard_metrics['Train/Loss_Total']
            axes[0, 0].plot(data['steps'], data['values'], 'b-', label='训练损失')
            axes[0, 0].set_title('训练损失')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
        
        # 验证损失曲线
        if 'Val/Loss_Total' in tensorboard_metrics:
            data = tensorboard_metrics['Val/Loss_Total']
            axes[0, 1].plot(data['steps'], data['values'], 'r-', label='验证损失')
            axes[0, 1].set_title('验证损失')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('损失')
            axes[0, 1].grid(True)
            axes[0, 1].legend()
        
        # 学习率曲线
        if 'Train/Learning_Rate' in tensorboard_metrics:
            data = tensorboard_metrics['Train/Learning_Rate']
            axes[1, 0].plot(data['steps'], data['values'], 'g-', label='学习率')
            axes[1, 0].set_title('学习率')
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        
        # GPU内存使用曲线
        if 'Train/GPU_Memory_GB' in tensorboard_metrics:
            data = tensorboard_metrics['Train/GPU_Memory_GB']
            axes[1, 1].plot(data['steps'], data['values'], 'm-', label='GPU内存')
            axes[1, 1].set_title('GPU内存使用')
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('内存 (GB)')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def monitor_loop(self, interval=30):
        """持续监控循环"""
        print("🔄 开始持续监控（按 Ctrl+C 停止）...")
        
        try:
            while True:
                os.system('clear')  # 清屏
                self.print_status()
                print(f"\n⏰ 下次更新将在 {interval} 秒后...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description='StructDiff 训练监控脚本')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='训练输出目录')
    parser.add_argument('--mode', type=str, choices=['status', 'plot', 'monitor'], 
                       default='status', help='监控模式')
    parser.add_argument('--interval', type=int, default=30,
                       help='监控间隔（秒）')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='保存训练曲线图的路径')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TrainingMonitor(args.output_dir)
    
    if args.mode == 'status':
        # 显示当前状态
        monitor.print_status()
    
    elif args.mode == 'plot':
        # 绘制训练曲线
        monitor.plot_training_curves(args.save_plot)
    
    elif args.mode == 'monitor':
        # 持续监控
        monitor.monitor_loop(args.interval)

if __name__ == "__main__":
    main() 