from tensorboard.backend.event_processing import event_accumulator
import os

# Early Fusion
print("Early Fusion 验证集准确率:")
print("=" * 50)
ea = event_accumulator.EventAccumulator('experiments/logs/early_fusion/events.out.tfevents.1768627348.LAPTOP-HS6D181B.42124.0')
ea.Reload()
val_acc = ea.Scalars('Epoch/Val_Acc')
for i, event in enumerate(val_acc):
    print(f'Epoch {i+1}: {event.value:.6f}')

print("\n" + "=" * 50)
print(f"最佳准确率: {max([e.value for e in val_acc]):.6f}")
print(f"最终准确率: {val_acc[-1].value:.6f}")
