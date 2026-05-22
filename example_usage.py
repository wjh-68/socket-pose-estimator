#!/usr/bin/env python3
"""
Example script showing how to use ChargeportPoseEstimator 
in both offline and online modes.
"""

import sys
import argparse
from chargeport_pose_estimator_ba import ChargeportPoseEstimator


def run_offline_example():
    """Example: Run pose estimation on offline dataset."""
    print("=" * 60)
    print("离线模式示例 - 从文件读取图像和位姿")
    print("=" * 60)
    
    estimator = ChargeportPoseEstimator(
        data_source_mode='offline',
        data_dir='dataset/0515',
        result_dir='result/0515_offline',
        max_frames=50,  # 只处理前50帧用于演示
    )
    
    print("\n配置信息:")
    print(f"  数据源模式: {estimator.data_source_mode}")
    print(f"  数据目录: {estimator.data_dir}")
    print(f"  结果目录: {estimator.result_dir}")
    print(f"  最大帧数: {estimator.max_frames}")
    print()
    
    try:
        estimator.run()
        print("\n✓ 离线处理完成")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        return False
    
    return True


def run_online_example():
    """Example: Run pose estimation with real-time hardware."""
    print("=" * 60)
    print("在线模式示例 - 实时从相机和机械臂获取数据")
    print("=" * 60)
    
    estimator = ChargeportPoseEstimator(
        data_source_mode='online',
        result_dir='result/online_session',
        robot_ip='192.168.1.20',
        robot_port=30004,
        camera_id=0,
        camera_width=2560,
        camera_height=1440,
        camera_brightness=128,
        sync_tolerance_ns=20_000_000,  # 20ms
        max_frames=-1,  # 无限处理
    )
    
    print("\n配置信息:")
    print(f"  数据源模式: {estimator.data_source_mode}")
    print(f"  机械臂地址: {estimator.data_source.robot_ip}:{estimator.data_source.robot_port}")
    print(f"  相机 ID: {estimator.data_source.camera_id}")
    print(f"  相机分辨率: {estimator.data_source.camera_width}x{estimator.data_source.camera_height}")
    print(f"  同步容差: {estimator.data_source.sync_tolerance_ns / 1e6:.1f}ms")
    print(f"  结果目录: {estimator.result_dir}")
    print()
    print("按 'q' 退出或 Ctrl+C 中断")
    print()
    
    try:
        estimator.run()
        print("\n✓ 在线处理完成")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        return False
    
    return True


def run_batch_offline():
    """Example: Batch process multiple offline datasets."""
    print("=" * 60)
    print("批量处理示例 - 连续处理多个数据集")
    print("=" * 60)
    
    datasets = [
        ('dataset/0515', 'result/0515_batch'),
        # 添加更多数据集...
        # ('dataset/0516', 'result/0516_batch'),
    ]
    
    for data_dir, result_dir in datasets:
        print(f"\n处理数据集: {data_dir}")
        
        estimator = ChargeportPoseEstimator(
            data_source_mode='offline',
            data_dir=data_dir,
            result_dir=result_dir,
            max_frames=100,
        )
        
        try:
            estimator.run()
            print(f"✓ {data_dir} 处理完成")
        except Exception as e:
            print(f"✗ {data_dir} 处理失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ 所有数据集处理完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Chargeport Pose Estimator - Example Usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 运行离线模式
  python example_usage.py --mode offline
  
  # 运行在线模式
  python example_usage.py --mode online
  
  # 批量处理
  python example_usage.py --mode batch
        '''
    )
    
    parser.add_argument(
        '--mode',
        choices=['offline', 'online', 'batch'],
        default='offline',
        help='运行模式 (default: offline)'
    )
    
    args = parser.parse_args()
    
    print("\n")
    
    if args.mode == 'offline':
        success = run_offline_example()
    elif args.mode == 'online':
        success = run_online_example()
    elif args.mode == 'batch':
        run_batch_offline()
        success = True
    else:
        print(f"未知模式: {args.mode}")
        success = False
    
    print("\n")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
