"""
================================================================================
SILK SAREE CLASSIFICATION — RESEARCH GRADE PIPELINE
================================================================================
NIT Silchar | Summer Research Internship | Under Prof. Binoy Roy
Classes  : Chanderi, Kanjeevaram, Kosa, Paithani, Patola, Uppada
Models   : MobileNetV2, EfficientNet-B0, EfficientNet-B3, ResNet-50,
           DenseNet-201, InceptionV3, ViT-B/16
Platform : Local machine (Windows / Mac)
================================================================================
SETUP — run this once in terminal before executing:
    pip install torch torchvision pillow scikit-learn seaborn matplotlib
                grad-cam timm tqdm onnx
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — CONFIGURATION  (only edit this section)
# ─────────────────────────────────────────────────────────────────────────────
import os


DATASET_PATH = r"C:\Users\HP\Desktop\silk_project\dataset"

# All outputs (models, plots, results) go here — created automatically
OUTPUT_DIR   = r"C:\Users\HP\Desktop\silk_project\outputs"

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['chanderi', 'kanjeevaram', 'kosa', 'paithani', 'patola', 'uppada']
NUM_CLASSES  = len(CLASS_NAMES)

# ── Training hyperparameters ──────────────────────────────────────────────────
IMG_SIZE         = 224      # all models use 224×224 except InceptionV3 (299)
INCEPTION_SIZE   = 299      # InceptionV3 specific
BATCH_SIZE       = 16       # reduce to 8 if you get out-of-memory errors
NUM_WORKERS      = 0        # keep 0 on Windows to avoid multiprocessing errors
SEED             = 42

STAGE1_EPOCHS    = 5        # frozen backbone — trains head only
STAGE2_EPOCHS    = 30       # full fine-tuning — all layers
LR_HEAD          = 1e-3     # learning rate for stage 1
LR_FINETUNE      = 1e-4     # learning rate for stage 2
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTHING  = 0.1      # reduces overconfidence
PATIENCE         = 8        # early stopping patience (epochs)

# ImageNet normalisation — standard for all pretrained models
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys, time, copy, json, warnings, random
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models

import timm                            # for ViT-B/16

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ── Output directories ────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

for d in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  SILK SAREE CLASSIFICATION — RESEARCH PIPELINE")
print("=" * 65)
print(f"  Device    : {device}")
if device.type == 'cuda':
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Classes   : {CLASS_NAMES}")
print(f"  Output    : {OUTPUT_DIR}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 DATASET AUDIT")
print("─" * 45)

counts = {}
for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)
    if not os.path.isdir(cls_path):
        continue
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    counts[cls] = len(imgs)

total_imgs      = sum(counts.values())
min_count       = min(counts.values())
max_count       = max(counts.values())
imbalance_ratio = max_count / min_count

for cls, cnt in sorted(counts.items()):
    bar  = '█' * (cnt // 5)
    flag = "✅" if cnt >= 150 else "⚠️  low"
    print(f"  {cls:<20} {cnt:>4} images  {flag}  {bar}")

print("─" * 45)
print(f"  Total images      : {total_imgs}")
print(f"  Smallest class    : {min_count}")
print(f"  Largest class     : {max_count}")
print(f"  Imbalance ratio   : {imbalance_ratio:.2f}x  "
      f"{'✅ acceptable' if imbalance_ratio < 1.5 else '⚠️  consider balancing'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.02),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(img_size=224):
    """Build train/val/test dataloaders for a given image size."""

    full_dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=get_transforms(img_size, is_train=True)
    )

    # Stratified 70 / 15 / 15 split
    total   = len(full_dataset)
    train_n = int(0.70 * total)
    val_n   = int(0.15 * total)
    test_n  = total - train_n - val_n

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_n, val_n, test_n],
        generator=generator
    )

    # Val and test get eval transforms (no augmentation)
    val_dataset  = datasets.ImageFolder(
        DATASET_PATH,
        transform=get_transforms(img_size, is_train=False)
    )
    test_dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=get_transforms(img_size, is_train=False)
    )

    val_set  = torch.utils.data.Subset(val_dataset,  val_set.indices)
    test_set = torch.utils.data.Subset(test_dataset, test_set.indices)

    # Weighted sampler for class balance
    train_labels   = [full_dataset.targets[i] for i in train_set.indices]
    class_counts_  = np.bincount(train_labels)
    class_weights_ = 1.0 / class_counts_
    sample_weights = [class_weights_[l] for l in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=sampler,   num_workers=NUM_WORKERS,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False,     num_workers=NUM_WORKERS,
                              pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False,     num_workers=NUM_WORKERS,
                              pin_memory=(device.type == 'cuda'))

    print(f"\n  DataLoaders built (img_size={img_size})")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_model(arch):
    """
    Builds a pretrained model with a new classification head
    for NUM_CLASSES outputs.
    """
    print(f"\n  Building {arch} ...")

    if arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif arch == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT')
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif arch == 'densenet201':
        model = models.densenet201(weights='DEFAULT')
        model.classifier = nn.Linear(
            model.classifier.in_features, NUM_CLASSES
        )

    elif arch == 'inception_v3':
        model = models.inception_v3(weights='DEFAULT', aux_logits=True)
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features, NUM_CLASSES
        )
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif arch == 'vit_b16':
        # ViT from timm — most accurate but heaviest
        model = timm.create_model(
            'vit_base_patch16_224', pretrained=True,
            num_classes=NUM_CLASSES
        )

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Total params: {total/1e6:.2f}M | "
          f"Trainable: {trainable/1e6:.2f}M")

    return model.to(device), total


def freeze_backbone(model, arch):
    """Freeze all layers except the classification head."""
    for param in model.parameters():
        param.requires_grad = False

    # Use if/elif — avoids Python evaluating all attributes at once
    if arch in ('mobilenet_v2', 'efficientnet_b0', 'efficientnet_b3', 'densenet201'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == 'resnet50':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == 'inception_v3':
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.AuxLogits.fc.parameters():
            param.requires_grad = True
    elif arch == 'vit_b16':
        for param in model.head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Frozen. Trainable params: {trainable:,}")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Unfrozen. Trainable params: {trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, arch=''):
    """Single epoch — train if optimizer given, else validate."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in tqdm(loader, leave=False, ncols=70):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train and optimizer:
                optimizer.zero_grad()

            # InceptionV3 returns (output, aux_output) during training
            if is_train and arch == 'inception_v3':
                outputs, aux_outputs = model(images)
                loss = criterion(outputs, labels) + \
                       0.4 * criterion(aux_outputs, labels)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if is_train and optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds      = torch.max(outputs, 1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def train_model(arch, train_loader, val_loader):
    """
    Full 2-stage training with early stopping.
    Returns trained model + history dict.
    """
    # InceptionV3 needs 299×299
    img_size = INCEPTION_SIZE if arch == 'inception_v3' else IMG_SIZE

    model, total_params = build_model(arch)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'stage2_start': STAGE1_EPOCHS + 1,
        'total_params': total_params,
        'arch': arch
    }

    best_val_acc   = 0.0
    best_wts       = copy.deepcopy(model.state_dict())
    save_path      = os.path.join(MODELS_DIR, f"{arch}_best.pth")

    # ── STAGE 1: HEAD ONLY ─────────────────────────────────────────────────
    print(f"\n  ── Stage 1: Frozen backbone ({STAGE1_EPOCHS} epochs) ──")
    freeze_backbone(model, arch)

    opt1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer=opt1, arch=arch)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion,
                                    arch=arch)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        saved = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            saved = " 💾"

        print(f"  S1 Ep {epoch:02d}/{STAGE1_EPOCHS} | "
              f"Train {tr_acc:.1f}% ({tr_loss:.4f}) | "
              f"Val {vl_acc:.1f}% ({vl_loss:.4f}) | "
              f"{time.time()-t0:.0f}s{saved}")

    # ── STAGE 2: FULL FINE-TUNING ──────────────────────────────────────────
    print(f"\n  ── Stage 2: Full fine-tuning ({STAGE2_EPOCHS} epochs) ──")
    unfreeze_all(model)

    opt2 = optim.AdamW(
        model.parameters(),
        lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        opt2, T_max=STAGE2_EPOCHS, eta_min=1e-6
    )

    no_improve = 0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer=opt2, arch=arch)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion,
                                    arch=arch)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        saved = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            saved = " 💾"
            no_improve = 0
        else:
            no_improve += 1

        lr_now = opt2.param_groups[0]['lr']
        print(f"  S2 Ep {epoch:02d}/{STAGE2_EPOCHS} | "
              f"Train {tr_acc:.1f}% ({tr_loss:.4f}) | "
              f"Val {vl_acc:.1f}% ({vl_loss:.4f}) | "
              f"LR {lr_now:.2e} | "
              f"{time.time()-t0:.0f}s{saved}")

        if no_improve >= PATIENCE:
            print(f"\n  ⚡ Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_wts)
    history['best_val_acc'] = best_val_acc
    print(f"\n  ✅ Best val accuracy: {best_val_acc:.2f}%")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, arch, model_label):
    """Full evaluation on test set. Returns metrics dict."""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc        = accuracy_score(all_labels, all_preds) * 100
    macro_f1   = f1_score(all_labels, all_preds,
                          average='macro', zero_division=0) * 100
    macro_prec = precision_score(all_labels, all_preds,
                                 average='macro', zero_division=0) * 100
    macro_rec  = recall_score(all_labels, all_preds,
                              average='macro', zero_division=0) * 100

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {model_label}")
    print(f"{'='*60}")
    print(f"  Accuracy        : {acc:.2f}%")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  Macro Precision : {macro_prec:.2f}%")
    print(f"  Macro Recall    : {macro_rec:.2f}%")
    print(f"  Test set size   : {len(all_labels)} images")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=3))

    return {
        'accuracy': acc, 'macro_f1': macro_f1,
        'precision': macro_prec, 'recall': macro_rec,
        'preds': all_preds, 'labels': all_labels,
        'probs': all_probs
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PLOT: TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, model_label):
    epochs = range(1, len(history['train_acc']) + 1)
    s2     = history.get('stage2_start', STAGE1_EPOCHS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training History — {model_label}',
                 fontsize=15, fontweight='bold', y=1.02)

    # ── Accuracy ─────────────────────────────────────────────
    ax1.plot(epochs, history['train_acc'], 'b-o',
             markersize=3, linewidth=2, label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'],   'r-o',
             markersize=3, linewidth=2, label='Val Accuracy')
    ax1.axvline(x=s2, color='gray',  linestyle='--',
                alpha=0.7, linewidth=1.5, label='Stage 2 starts')
    ax1.axhline(y=90, color='green', linestyle='--',
                alpha=0.6, linewidth=1.5, label='90% target')
    ax1.fill_between(epochs, history['train_acc'],
                     history['val_acc'], alpha=0.08, color='purple',
                     label='Train–Val gap')
    ax1.set_title('Accuracy over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # ── Loss ─────────────────────────────────────────────────
    ax2.plot(epochs, history['train_loss'], 'b-o',
             markersize=3, linewidth=2, label='Train Loss')
    ax2.plot(epochs, history['val_loss'],   'r-o',
             markersize=3, linewidth=2, label='Val Loss')
    ax2.axvline(x=s2, color='gray', linestyle='--',
                alpha=0.7, linewidth=1.5, label='Stage 2 starts')
    ax2.set_title('Loss over Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save = os.path.join(PLOTS_DIR,
           f"curves_{model_label.lower().replace(' ','_').replace('/','')}.png")
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Training curves saved: {save}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(results, model_label):
    cm       = confusion_matrix(results['labels'], results['preds'])
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'Confusion Matrix — {model_label}\n'
        f"Test Accuracy: {results['accuracy']:.2f}%  |  "
        f"Macro F1: {results['macro_f1']:.2f}%",
        fontsize=14, fontweight='bold'
    )

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Raw Counts', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # Normalised percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=axes[1], cbar_kws={'shrink': 0.8},
                vmin=0, vmax=100)
    axes[1].set_title('Normalised (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save = os.path.join(PLOTS_DIR,
           f"cm_{model_label.lower().replace(' ','_').replace('/','')}.png")
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion matrix saved: {save}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — PLOT: MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison_table(all_results, all_histories):
    models_   = list(all_results.keys())
    accs      = [all_results[m]['accuracy']   for m in models_]
    f1s       = [all_results[m]['macro_f1']   for m in models_]
    precs     = [all_results[m]['precision']  for m in models_]
    recs      = [all_results[m]['recall']     for m in models_]
    params    = [all_histories[m]['total_params'] / 1e6
                 for m in models_]

    # ── Bar chart comparison ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Comparison — Silk Saree Classification',
                 fontsize=15, fontweight='bold')

    x      = np.arange(len(models_))
    width  = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

    for i, (metric, vals, color) in enumerate(zip(
        ['Accuracy', 'Macro F1', 'Precision', 'Recall'],
        [accs, f1s, precs, recs],
        colors
    )):
        axes[0].bar(x + i * width, vals, width,
                    label=metric, color=color,
                    alpha=0.85, edgecolor='white')

    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_ylabel('Score (%)', fontsize=11)
    axes[0].set_title('Performance Metrics', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(models_, rotation=30, ha='right', fontsize=9)
    axes[0].axhline(y=90, color='red', linestyle='--',
                    alpha=0.5, linewidth=1.5, label='90% target')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim([0, 110])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Parameter count vs accuracy scatter
    axes[1].scatter(params, accs, s=200, zorder=5,
                    c=colors[:len(models_)], edgecolors='black', linewidths=1.5)
    for i, m in enumerate(models_):
        axes[1].annotate(m, (params[i], accs[i]),
                         textcoords='offset points',
                         xytext=(8, 4), fontsize=9)
    axes[1].set_xlabel('Parameters (Millions)', fontsize=11)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=11)
    axes[1].set_title('Accuracy vs Model Size',
                      fontsize=12, fontweight='bold')
    axes[1].axhline(y=90, color='red', linestyle='--',
                    alpha=0.5, linewidth=1.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save = os.path.join(PLOTS_DIR, 'model_comparison.png')
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Printed table ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Model':<20} {'Acc%':>7} {'F1%':>7} "
          f"{'Prec%':>7} {'Rec%':>7} {'Params':>10}  {'Rank'}")
    print("-" * 80)

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        p_m   = all_histories[name]['total_params'] / 1e6
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'  {rank}.')
        print(f"  {name:<20} {res['accuracy']:>7.2f} "
              f"{res['macro_f1']:>7.2f} "
              f"{res['precision']:>7.2f} "
              f"{res['recall']:>7.2f} "
              f"{p_m:>9.1f}M  {medal}")

    print("=" * 80)

    # Save as JSON
    summary = {
        name: {
            'accuracy':  res['accuracy'],
            'macro_f1':  res['macro_f1'],
            'precision': res['precision'],
            'recall':    res['recall'],
            'params_M':  all_histories[name]['total_params'] / 1e6,
        }
        for name, res in all_results.items()
    }
    with open(os.path.join(RESULTS_DIR, 'comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  📄 Comparison table saved: {PLOTS_DIR}/model_comparison.png")
    print(f"  📄 JSON results saved    : {RESULTS_DIR}/comparison.json")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — GRAD-CAM VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def get_target_layer(model, arch):
    """Returns last conv layer for GradCAM for each architecture."""
    layers = {
        'mobilenet_v2'   : [model.features[-1][0]],
        'efficientnet_b0': [model.features[-1][0]],
        'efficientnet_b3': [model.features[-1][0]],
        'resnet50'       : [model.layer4[-1].conv3],
        'densenet201'    : [model.features.denseblock4
                                .denselayer32.conv2],
        'inception_v3'   : [model.Mixed_7c],
        'vit_b16'        : [model.blocks[-1].norm1],
    }
    return layers.get(arch, None)


def plot_gradcam(model, arch, test_loader, model_label):
    """Generates GradCAM heatmaps — one image per class."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  ⚠️  Install grad-cam: pip install grad-cam")
        return

    target_layers = get_target_layer(model, arch)
    if not target_layers:
        print(f"  ⚠️  GradCAM layer not configured for {arch}")
        return

    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers)

    # One image per class from test set
    class_imgs  = {i: None for i in range(NUM_CLASSES)}
    class_origs = {i: None for i in range(NUM_CLASSES)}
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t  = torch.tensor(STD).view(3, 1, 1)

    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            lbl = lbl.item()
            if class_imgs[lbl] is None:
                class_imgs[lbl]  = img.unsqueeze(0)
                orig = torch.clamp(img * std_t + mean_t, 0, 1)
                class_origs[lbl] = orig.permute(1, 2, 0).numpy()
        if all(v is not None for v in class_imgs.values()):
            break

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(22, 9))
    fig.suptitle(
        f'Grad-CAM — {model_label}\n'
        f'Top: original  |  Bottom: model attention (red = high attention)',
        fontsize=13, fontweight='bold'
    )

    for cls_idx in range(NUM_CLASSES):
        if class_imgs[cls_idx] is None:
            continue

        inp      = class_imgs[cls_idx].to(device)
        rgb_img  = class_origs[cls_idx].astype(np.float32)
        targets  = [ClassifierOutputTarget(cls_idx)]
        grayscale = cam(input_tensor=inp, targets=targets)
        cam_img   = show_cam_on_image(rgb_img, grayscale[0], use_rgb=True)

        axes[0, cls_idx].imshow(rgb_img)
        axes[0, cls_idx].set_title(CLASS_NAMES[cls_idx],
                                   fontsize=11, fontweight='bold')
        axes[0, cls_idx].axis('off')

        axes[1, cls_idx].imshow(cam_img)
        axes[1, cls_idx].set_title('Focus region',
                                   fontsize=9, color='darkred')
        axes[1, cls_idx].axis('off')

    plt.tight_layout()
    save = os.path.join(PLOTS_DIR,
           f"gradcam_{model_label.lower().replace(' ','_').replace('/','')}.png")
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Grad-CAM saved: {save}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — SINGLE IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_image(image_path, model, arch='mobilenet_v2',
                  threshold=0.70):
    """
    Predict silk type from any single image.
    Shows image + confidence bar chart side by side.
    """
    img_size  = INCEPTION_SIZE if arch == 'inception_v3' else IMG_SIZE
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output[0], dim=0).cpu().numpy()

    pred_idx   = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.imshow(img)
    ax1.axis('off')

    if confidence >= threshold * 100:
        color = '#2E7D32'
        icon  = '✅'
    else:
        color = '#E65100'
        icon  = '🤔'

    ax1.set_title(
        f"{icon}  {pred_class.upper()}\n"
        f"Confidence: {confidence:.1f}%",
        fontsize=14, color=color, fontweight='bold', pad=12
    )

    sorted_idx  = np.argsort(probs)[::-1]
    sorted_cls  = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_prob = probs[sorted_idx] * 100
    bar_colors  = ['#4CAF50' if i == pred_idx else '#90CAF9'
                   for i in sorted_idx]

    bars = ax2.barh(sorted_cls[::-1], sorted_prob[::-1],
                    color=bar_colors[::-1],
                    edgecolor='white', height=0.55)
    ax2.set_xlabel('Confidence (%)', fontsize=11)
    ax2.set_title('Confidence Per Class',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 115])
    ax2.axvline(x=threshold * 100, color='red', linestyle='--',
                alpha=0.6, linewidth=1.5,
                label=f'Threshold ({threshold*100:.0f}%)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, prob in zip(bars, sorted_prob[::-1]):
        ax2.text(bar.get_width() + 1,
                 bar.get_y() + bar.get_height() / 2,
                 f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    save = os.path.join(RESULTS_DIR, 'prediction_result.png')
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Prediction : {pred_class.upper()}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"\n  All probabilities (sorted):")
    for i in sorted_idx:
        bar = '█' * int(probs[i] * 40)
        print(f"    {CLASS_NAMES[i]:<20} "
              f"{probs[i]*100:>6.2f}%  {bar}")

    return pred_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — TFLITE EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_to_onnx(model, arch, model_label):
    """
    Exports trained model to ONNX format.
    ONNX can then be converted to TFLite using
    onnx-tf (separately) or ai_edge_torch for Flutter app.
    """
    model.eval().cpu()
    img_size    = INCEPTION_SIZE if arch == 'inception_v3' else IMG_SIZE
    dummy_input = torch.randn(1, 3, img_size, img_size)
    onnx_path   = os.path.join(MODELS_DIR, f"{arch}.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ ONNX exported: {onnx_path}  ({size_mb:.1f} MB)")
    model.to(device)
    return onnx_path


def save_class_labels(best_arch, best_acc, best_f1):
    """Saves class labels and model config for mobile app."""
    config = {
        'classes':      CLASS_NAMES,
        'num_classes':  NUM_CLASSES,
        'img_size':     IMG_SIZE,
        'mean':         MEAN,
        'std':          STD,
        'best_model':   best_arch,
        'test_accuracy': round(best_acc, 2),
        'macro_f1':     round(best_f1, 2),
        'class_to_idx': {c: i for i, c in enumerate(CLASS_NAMES)},
    }
    path = os.path.join(MODELS_DIR, 'labels.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Labels config saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# All 7 architectures with their image sizes
ARCHITECTURES = {
    'MobileNetV2'    : ('mobilenet_v2',    IMG_SIZE),
    'EfficientNet-B0': ('efficientnet_b0', IMG_SIZE),
    'EfficientNet-B3': ('efficientnet_b3', IMG_SIZE),
    'ResNet-50'      : ('resnet50',        IMG_SIZE),
    'DenseNet-201'   : ('densenet201',     IMG_SIZE),
    'InceptionV3'    : ('inception_v3',    INCEPTION_SIZE),
    'ViT-B/16'       : ('vit_b16',         IMG_SIZE),
}

all_results   = {}
all_histories = {}
best_model_obj = None
best_model_arch = None

for model_label, (arch, img_size) in ARCHITECTURES.items():

    print(f"\n\n{'#'*65}")
    print(f"  TRAINING: {model_label}")
    print(f"{'#'*65}")

    # Build dataloaders (InceptionV3 needs 299×299)
    train_loader, val_loader, test_loader = build_dataloaders(img_size)

    # Train
    model, history = train_model(arch, train_loader, val_loader)

    # Plot training curves
    plot_training_curves(history, model_label)

    # Evaluate on test set
    results = evaluate_model(model, test_loader, arch, model_label)

    # Plot confusion matrix
    plot_confusion_matrix(results, model_label)

    # GradCAM
    plot_gradcam(model, arch, test_loader, model_label)

    # Store results
    all_results[model_label]   = results
    all_histories[model_label] = history

    # Track best model
    if (best_model_obj is None or
            results['accuracy'] > all_results.get(
                best_model_arch, {}).get('accuracy', 0)):
        best_model_obj  = copy.deepcopy(model)
        best_model_arch = model_label

    print(f"\n  ✅ {model_label} done — "
          f"Acc: {results['accuracy']:.2f}%  "
          f"F1: {results['macro_f1']:.2f}%")

    # Free GPU memory before next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — FINAL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "#" * 65)
print("  FINAL OUTPUTS")
print("#" * 65)

# Comparison table + chart
plot_comparison_table(all_results, all_histories)

# Best model
best_arch_key = ARCHITECTURES[best_model_arch][0]
best_acc      = all_results[best_model_arch]['accuracy']
best_f1       = all_results[best_model_arch]['macro_f1']

print(f"\n  🏆 Best Model : {best_model_arch}")
print(f"     Accuracy   : {best_acc:.2f}%")
print(f"     Macro F1   : {best_f1:.2f}%")

# Export best model to ONNX
export_to_onnx(best_model_obj, best_arch_key, best_model_arch)

# Save labels config for app
save_class_labels(best_model_arch, best_acc, best_f1)

print(f"\n{'='*65}")
print("  PIPELINE COMPLETE")
print(f"{'='*65}")
print(f"  Models   → {MODELS_DIR}")
print(f"  Plots    → {PLOTS_DIR}")
print(f"  Results  → {RESULTS_DIR}")
print(f"\n  Files generated:")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"    📊 {f}")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"    📄 {f}")
for f in sorted(os.listdir(MODELS_DIR)):
    print(f"    💾 {f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — PREDICT ON A NEW IMAGE
# ─────────────────────────────────────────────────────────────────────────────
# After training is done, use this to test any new saree image:
#
# predict_image(
#     r"C:\Users\YourName\Desktop\test_saree.jpg",
#     best_model_obj,
#     arch=best_arch_key
# )
