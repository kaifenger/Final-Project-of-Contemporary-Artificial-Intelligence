# 一键训练所有多模态融合模型 (PowerShell脚本)
# 顺序训练：Early Fusion → Late Fusion → Cross-Attention

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  🎯 多模态融合模型 - 批量训练脚本" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# 训练配置
$configs = @(
    @{
        Name = "Early Fusion"
        Config = "configs/early_fusion.yaml"
        Description = "早期融合（特征拼接）"
    },
    @{
        Name = "Late Fusion"
        Config = "configs/late_fusion.yaml"
        Description = "晚期融合（决策加权）"
    },
    @{
        Name = "Cross-Attention Fusion"
        Config = "configs/cross_attention.yaml"
        Description = "跨模态注意力融合"
    }
)

# 打印训练计划
Write-Host "📋 训练计划:" -ForegroundColor Green
for ($i = 0; $i -lt $configs.Count; $i++) {
    $num = $i + 1
    Write-Host "  $num. $($configs[$i].Name) - $($configs[$i].Description)"
}

Write-Host ""
Write-Host "📊 总计: $($configs.Count) 个模型" -ForegroundColor Cyan
Write-Host "⚙️  训练策略: 分层学习率 (backbone: 1e-5, projection/classifier: 1e-3)"
Write-Host "📉 Early Stopping: patience=3, min_delta=0.001"
Write-Host ""

# 询问确认
Write-Host ("-" * 70) -ForegroundColor Gray
$confirmation = Read-Host "❓ 确认开始训练? [Y/n]"
if ($confirmation -and $confirmation -ne 'Y' -and $confirmation -ne 'y' -and $confirmation -ne 'yes') {
    Write-Host "❌ 训练已取消" -ForegroundColor Red
    exit
}

# 记录总开始时间
$totalStartTime = Get-Date

# 训练结果统计
$results = @()

# 依次训练每个模型
for ($i = 0; $i -lt $configs.Count; $i++) {
    $index = $i + 1
    $config = $configs[$i]
    
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  [$index/$($configs.Count)] 开始训练: $($config.Name)" -ForegroundColor Yellow
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "📝 描述: $($config.Description)"
    Write-Host "⚙️  配置文件: $($config.Config)"
    Write-Host "🕐 开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    
    # 检查配置文件是否存在
    if (-not (Test-Path $config.Config)) {
        Write-Host "❌ 错误: 配置文件不存在 - $($config.Config)" -ForegroundColor Red
        $results += @{
            Name = $config.Name
            Success = $false
        }
        continue
    }
    
    # 记录开始时间
    $startTime = Get-Date
    
    # 执行训练
    Write-Host "🚀 执行命令: python src/train_fusion.py --config $($config.Config)" -ForegroundColor Green
    Write-Host ""
    
    try {
        python src/train_fusion.py --config $config.Config
        $exitCode = $LASTEXITCODE
        
        # 计算训练时长
        $elapsed = (Get-Date) - $startTime
        $hours = [math]::Floor($elapsed.TotalHours)
        $minutes = $elapsed.Minutes
        $seconds = $elapsed.Seconds
        
        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "✅ $($config.Name) 训练完成!" -ForegroundColor Green
            Write-Host "⏱️  用时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
            $results += @{
                Name = $config.Name
                Success = $true
            }
        } else {
            Write-Host ""
            Write-Host "❌ $($config.Name) 训练失败! 退出代码: $exitCode" -ForegroundColor Red
            Write-Host "⏱️  已用时: ${hours}小时 ${minutes}分钟"
            $results += @{
                Name = $config.Name
                Success = $false
            }
            
            # 询问是否继续
            if ($index -lt $configs.Count) {
                Write-Host ""
                Write-Host ("-" * 70) -ForegroundColor Gray
                $continueTraining = Read-Host "❓ 当前模型训练失败，是否继续训练下一个模型? [Y/n]"
                if ($continueTraining -and $continueTraining -ne 'Y' -and $continueTraining -ne 'y' -and $continueTraining -ne 'yes') {
                    Write-Host "⏸️  批量训练已终止" -ForegroundColor Yellow
                    break
                }
            }
        }
    } catch {
        Write-Host ""
        Write-Host "❌ $($config.Name) 训练过程中发生错误!" -ForegroundColor Red
        Write-Host "💥 错误信息: $_" -ForegroundColor Red
        $results += @{
            Name = $config.Name
            Success = $false
        }
    }
}

# 计算总用时
$totalElapsed = (Get-Date) - $totalStartTime
$totalHours = [math]::Floor($totalElapsed.TotalHours)
$totalMinutes = $totalElapsed.Minutes
$totalSeconds = $totalElapsed.Seconds

# 打印训练总结
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  📊 训练总结" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

Write-Host "训练结果:"
foreach ($result in $results) {
    if ($result.Success) {
        Write-Host "  • $($result.Name.PadRight(25)) ✅ 成功" -ForegroundColor Green
    } else {
        Write-Host "  • $($result.Name.PadRight(25)) ❌ 失败" -ForegroundColor Red
    }
}

$successCount = ($results | Where-Object { $_.Success }).Count
$failCount = $results.Count - $successCount

Write-Host ""
Write-Host "统计:"
Write-Host "  • 成功: $successCount/$($results.Count)"
Write-Host "  • 失败: $failCount/$($results.Count)"
Write-Host "  • 总用时: ${totalHours}小时 ${totalMinutes}分钟 ${totalSeconds}秒"

if ($successCount -eq $configs.Count) {
    Write-Host ""
    Write-Host "🎉 所有模型训练完成!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 检查点保存位置:" -ForegroundColor Cyan
    Write-Host "  • checkpoints/early_fusion/"
    Write-Host "  • checkpoints/late_fusion/"
    Write-Host "  • checkpoints/cross_attention/"
    Write-Host ""
    Write-Host "📈 TensorBoard日志:" -ForegroundColor Cyan
    Write-Host "  • experiments/logs/early_fusion/"
    Write-Host "  • experiments/logs/late_fusion/"
    Write-Host "  • experiments/logs/cross_attention/"
    Write-Host ""
    Write-Host "💡 查看训练结果:" -ForegroundColor Yellow
    Write-Host "  tensorboard --logdir experiments/logs"
} else {
    Write-Host ""
    Write-Host "⚠️  部分模型训练失败，请检查错误信息" -ForegroundColor Yellow
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
