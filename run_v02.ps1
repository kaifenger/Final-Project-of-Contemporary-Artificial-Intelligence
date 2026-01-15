# 运行脚本 - Windows批处理文件
# 用于训练和评估单模态基线模型

# Python路径
$PYTHON = "C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe"

Write-Host "========================================" -ForegroundColor Green
Write-Host "Multimodal Sentiment Classification" -ForegroundColor Green
Write-Host "v0.2: Single-modal Baseline Models" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 菜单
Write-Host "`n请选择操作:"
Write-Host "1. 训练纯文本模型 (BERT)"
Write-Host "2. 训练纯图像模型 (ResNet50)"
Write-Host "3. 评估文本模型"
Write-Host "4. 评估图像模型"
Write-Host "5. 训练所有基线模型"
Write-Host "6. 退出"

$choice = Read-Host "`n请输入选项 (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`n开始训练文本模型..." -ForegroundColor Yellow
        & $PYTHON src/train.py --config configs/text_only.yaml
    }
    "2" {
        Write-Host "`n开始训练图像模型..." -ForegroundColor Yellow
        & $PYTHON src/train.py --config configs/image_only.yaml
    }
    "3" {
        Write-Host "`n评估文本模型..." -ForegroundColor Yellow
        & $PYTHON src/evaluate.py --config configs/text_only.yaml --checkpoint checkpoints/text_only/best_model.pth
    }
    "4" {
        Write-Host "`n评估图像模型..." -ForegroundColor Yellow
        & $PYTHON src/evaluate.py --config configs/image_only.yaml --checkpoint checkpoints/image_only/best_model.pth
    }
    "5" {
        Write-Host "`n训练所有基线模型..." -ForegroundColor Yellow
        Write-Host "`n[1/2] 训练文本模型..."
        & $PYTHON src/train.py --config configs/text_only.yaml
        Write-Host "`n[2/2] 训练图像模型..."
        & $PYTHON src/train.py --config configs/image_only.yaml
        Write-Host "`n所有模型训练完成!" -ForegroundColor Green
    }
    "6" {
        Write-Host "`n退出。" -ForegroundColor Cyan
        exit
    }
    default {
        Write-Host "`n无效选项!" -ForegroundColor Red
    }
}

Write-Host "`n按任意键继续..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
