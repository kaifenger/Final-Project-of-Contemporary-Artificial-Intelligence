# v0.3 开发过程中遇到的Bug及解决方案

## Bug 1: DataLoader多线程与VPN软件冲突

**错误现象**:
```
Epoch 0:   0%|                                                    | 0/100 [00:00<?, ?it/s]
❌ Early Fusion训练失败！
```
- 训练刚启动就自动停止
- VPN程序崩溃退出
- 无明显错误traceback

**原因分析**:
- `num_workers=4`启用了多进程数据加载
- Windows系统下PyTorch多进程与VPN软件的网络钩子冲突
- 子进程创建时触发VPN程序异常

**解决方案**:
```yaml
# 修改所有配置文件
num_workers: 0  # 改为单进程模式，避免VPN冲突
```

**影响**: 
- 数据加载速度略微下降（~5%）
- 但CPU训练中计算是瓶颈，实际影响可忽略
- 稳定性大幅提升

---

## Bug 2: PyTorch缓存导致导入卡死

**错误现象**:
```bash
PS> python src\train_fusion.py --config configs\early_fusion.yaml
============================================================
启动训练脚本...
============================================================
正在导入PyTorch...
# 卡住无响应，无任何输出
```

**触发条件**:
- 训练完成1个epoch后程序突然停止
- 重启脚本时`import torch`就卡住
- 即使简单的`python -c "import torch"`也无响应

**原因分析**:
1. PyTorch在训练过程中使用Intel MKL库进行线程管理
2. 程序异常退出时MKL线程未正确释放
3. 遗留的锁定文件/共享内存导致下次导入时死锁
4. Windows临时文件缓存（`%TEMP%`）中的PyTorch缓存损坏

**解决方案**:
1. **立即方案**: 重启系统清除所有进程和缓存
2. **预防方案**: 在代码中添加环境变量设置（已实现）:
   ```python
   # 在导入torch前设置
   import os
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   ```
3. **清理脚本**: 定期清理Python进程和临时文件:
   ```powershell
   Get-Process python* | Stop-Process -Force
   Remove-Item -Path $env:TEMP\torch_* -Recurse -Force -ErrorAction SilentlyContinue
   ```

**影响**: 
- 开发过程中浪费约30分钟排查
- 训练中断需重新开始当前epoch
- 通过添加checkpoint机制和内存清理代码降低风险

---

## 其他修复的技术问题

### 3. 数据解包格式不匹配
**错误**: `ValueError: too many values to unpack (expected 3)`
- **原因**: `MultimodalDataset.__getitem__`返回字典，但训练代码期望元组
- **修复**: 修改训练循环从字典提取数据
```python
# 修改前: for text, images, labels in loader:
# 修改后: 
for batch in loader:
    text = batch['text']
    images = batch['image']
    labels = batch['label']
```

### 4. Tokenizer缺失
**错误**: `AttributeError: 'list' object has no attribute 'items'`
- **原因**: `TextPreprocessor`只清洗文本，未tokenize
- **修复**: 在Dataset中添加tokenizer参数，返回tokenized字典

### 5. 相对路径错误
**错误**: `FileNotFoundError: No such file or directory: '../data_split/train_split.csv'`
- **原因**: 批处理脚本切换到项目根目录，但配置文件路径错误
- **修复**: 所有路径改为相对于项目根目录（`data_split/` 而非 `../data_split/`）

### 6. CPU训练pin_memory警告
**警告**: `'pin_memory' argument is set as true but no accelerator is found`
- **原因**: CPU训练时pin_memory无效且产生警告
- **修复**: 根据设备类型动态设置`pin_memory = device.type == 'cuda'`

---

## 总结与经验

1. **多进程问题**: Windows环境下PyTorch多进程容易与系统软件冲突，CPU训练时使用单进程更稳定
2. **内存管理**: 添加显式的`gc.collect()`和`torch.cuda.empty_cache()`，每10个batch清理一次
3. **缓存清理**: 长时间训练后建议重启系统，避免累积的临时文件和锁定资源
4. **调试输出**: 在关键位置添加进度提示，快速定位卡住的环节
5. **容错设计**: 实现checkpoint保存和恢复机制，避免意外中断导致重头开始

**代码质量提升**:
- 从初始版本的频繁崩溃到最终稳定运行15 epochs
- 通过10次代码修复和6次GitHub提交逐步完善
- 最终实现了完整的内存管理和异常处理机制
