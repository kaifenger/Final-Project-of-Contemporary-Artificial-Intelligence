@echo off
chcp 65001 >nul

REM 切换到批处理文件所在目录
cd /d %~dp0

echo ========================================
echo v0.3 多模态融合方法对比实验
echo ========================================
echo.

set PYTHON=C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe

echo [1/4] 训练 Early Fusion (预计2.5小时)...
echo 开始时间: %time%
%PYTHON% src\train_fusion.py --config configs\early_fusion.yaml
if %errorlevel% neq 0 (
    echo ❌ Early Fusion训练失败！
    pause
    exit /b 1
)
echo ✅ Early Fusion完成
echo.

echo [2/4] 训练 Late Fusion (预计10分钟)...
echo 开始时间: %time%
%PYTHON% src\train_fusion.py --config configs\late_fusion.yaml
if %errorlevel% neq 0 (
    echo ❌ Late Fusion训练失败！
    pause
    exit /b 1
)
echo ✅ Late Fusion完成
echo.

echo [3/4] 训练 CLIP Fusion (预计1小时)...
echo 开始时间: %time%
%PYTHON% src\train_fusion.py --config configs\clip_fusion.yaml
if %errorlevel% neq 0 (
    echo ❌ CLIP Fusion训练失败！
    pause
    exit /b 1
)
echo ✅ CLIP Fusion完成
echo.

echo [4/4] 训练 BLIP-2 Fusion (预计1.5小时)...
echo 开始时间: %time%
%PYTHON% src\train_fusion.py --config configs\blip2_fusion.yaml
if %errorlevel% neq 0 (
    echo ❌ BLIP-2 Fusion训练失败！
    pause
    exit /b 1
)
echo ✅ BLIP-2 Fusion完成
echo.

echo ========================================
echo 所有模型训练完成！
echo 完成时间: %time%
echo ========================================
echo.

echo 生成对比报告...
%PYTHON% src\compare_fusion.py --output_dir experiments\fusion_comparison

echo.
echo ✅ 全部完成！
echo 对比结果已保存到: experiments\fusion_comparison
echo.
pause
