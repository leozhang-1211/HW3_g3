"""
Brain-Computer Interface Multi-Classifier (BrainLink Version)
For EEG signal Relax / Focus / Stroop state classification (Three-Class)
Compares 4 models: MLP, SVM, Random Forest, and a Self-Selected Model

Evaluation: Leave-One-Subject-Out (LOSO) cross-validation.
Outputs:
  - bci_multi_model_comparison.png : model comparison, per-subject accuracy,
                                     confusion matrix (counts + row-normalized)
  - results_summary.txt            : text summary, ready to paste into the report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# ==========================================
# === STUDENT MODEL IMPORT HERE ===
# ==========================================
from sklearn.neighbors import KNeighborsClassifier
import scipy.signal as signal
from scipy.stats import kurtosis, skew

from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"[{desc}] Processing...")
        return iterable

# Parameter Settings
class Config:
    DATASET_PATH = "bci_dataset_115-1"
    CLASS_NAMES = ['Relax', 'Focus', 'Stroop']

    # MLP parameters
    HIDDEN_LAYERS = (128, 64, 32)
    MAX_ITER = 150
    LEARNING_RATE = 0.002
    ALPHA = 0.005

    # SVM parameters
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0

    # Random Forest parameters
    RF_ESTIMATORS = 100
    RF_MAX_DEPTH = None

    # Signal processing parameters
    SAMPLING_RATE = 512    
    SEGMENT_LENGTH = 5     
    OVERLAP_RATIO = 0.6    

    # Feature selection parameters
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 20

    # Reporting settings
    TARGET_ACCURACY = 0.75   
    FIGURE_NAME = 'bci_multi_model_comparison.png'
    SUMMARY_NAME = 'results_summary.txt'

    MODEL_COLORS = {
        'MLP': '#2a78d6',           
        'SVM': '#eb6834',           
        'RF': '#1baf7a',            
        'SelfSelected': '#4a3aa7',  
    }
    FALLBACK_COLOR = '#8a8985'
    REFERENCE_COLOR = '#52514e'     
    HEATMAP_CMAP = 'Blues'          

    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples):
    """Split a single round of continuous EEG signal into multiple segments"""
    if len(data) < segment_length_samples:
        return []

    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    nyquist = 0.5 * Config.SAMPLING_RATE
    low = 1.0 / nyquist
    high = 50.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]

        # ==========================================
        # === STUDENT PREPROCESSING HERE (Part 1)===
        # ==========================================
        segment = signal.filtfilt(b, a, segment)

        segments.append(segment)
        start += step

    return segments


def extract_features(segments):
    """
    State-of-the-art Feature Extraction for Single-Channel (Fp1) EEG
    Based on recent literature combining Time, Frequency, and Non-linear Dynamics.
    """
    features = []
    
    # 預先定義常數以加速運算
    eps = 1e-8
    fs = Config.SAMPLING_RATE
    
    for seg in segments:
        # ==========================================
        # 1. 基礎時域特徵 (Time Domain)
        # ==========================================
        # 捨棄單純的 Mean (易受基線影響)，改用更高階的統計量
        var_val = np.var(seg)
        skew_val = skew(seg)        # 偏態：捕捉訊號的不對稱性
        kurt_val = kurtosis(seg)    # 峰度：捕捉訊號中極端雜訊(如眨眼)的尖銳度
        
        # Hjorth Parameters (Activity, Mobility, Complexity)
        # 文獻證實能有效捕捉腦波在時域上的高頻震盪與複雜度
        diff1 = np.diff(seg)
        diff2 = np.diff(diff1)
        
        activity = var_val if var_val > 0 else eps
        var_d1 = np.var(diff1) if np.var(diff1) > 0 else eps
        var_d2 = np.var(diff2) if np.var(diff2) > 0 else eps
        
        mobility = np.sqrt(var_d1 / activity)
        complexity = np.sqrt(var_d2 / var_d1) / mobility

        # ==========================================
        # 2. 頻域特徵 (Frequency Domain via Welch's Method)
        # ==========================================
        # 使用 512 點的 Hamming Window 進行頻譜估計
        freqs, psd = signal.welch(seg, fs=fs, nperseg=512, window='hamming')
        
        # 根據文獻，前額葉在專注與放鬆任務中，主要反應在 Theta 到 Gamma 帶
        # 捨棄 4Hz 以下的 Delta 帶 (極易被眼動 EOG 汙染)
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'low_beta': (13, 20),
            'high_beta': (20, 30),
            'gamma': (30, 45)
        }
        
        band_powers = {}
        for band_name, (f_min, f_max) in bands.items():
            idx = np.where((freqs >= f_min) & (freqs < f_max))[0]
            # 使用梯形積分 (trapezoid) 計算頻帶絕對功率
            band_powers[band_name] = np.trapezoid(psd[idx], freqs[idx])
            
        total_power = sum(band_powers.values()) + eps
        
        # 萃取相對功率 (Relative Power) 並進行 Log 轉換
        # 這是機器學習模型最容易收斂的特徵型態
        log_rel_powers = {
            f"log_rel_{name}": np.log((power / total_power) + eps) 
            for name, power in band_powers.items()
        }

        # ==========================================
        # 3. 進階認知狀態指標 (Cognitive Ratios)
        # ==========================================
        # Theta / Beta Ratio (TBR): 評估專注度與認知負荷的黃金指標
        beta_power = band_powers['low_beta'] + band_powers['high_beta'] + eps
        tbr = band_powers['theta'] / beta_power
        
        # Alpha / Beta Ratio (ABR): 評估放鬆與清醒程度
        abr = band_powers['alpha'] / beta_power
        
        # Theta / Alpha Ratio (TAR): 認知疲勞與認知衝突(Stroop)指標
        tar = band_powers['theta'] / (band_powers['alpha'] + eps)

        # ==========================================
        # 4. 非線性動力學特徵 (Non-linear Dynamics)
        # ==========================================
        # 頻譜熵 (Spectral Entropy)：測量大腦狀態的混亂度
        # 放鬆時 (Relax) 熵值較低，專注或衝突時 (Focus/Stroop) 熵值會升高
        psd_norm = psd / (np.sum(psd) + eps)
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm))

        # Teager-Kaiser Energy Operator (TKEO) 的平均值
        # 對高頻瞬態能量變化極度敏感，有助於區分專注時的高頻微震盪
        tkeo = seg[1:-1]**2 - seg[:-2] * seg[2:]
        tkeo_mean = np.mean(tkeo)

        # ==========================================
        # 組合所有特徵 (共 16 維度)
        # ==========================================
        current_feature = np.array([
            skew_val, kurt_val,               # 統計形狀 (2)
            activity, mobility, complexity,   # Hjorth 參數 (3)
            log_rel_powers['log_rel_theta'],  # 對數相對功率 (5)
            log_rel_powers['log_rel_alpha'],
            log_rel_powers['log_rel_low_beta'],
            log_rel_powers['log_rel_high_beta'],
            log_rel_powers['log_rel_gamma'],
            tbr, abr, tar,                    # 認知比例指標 (3)
            spectral_entropy,                 # 頻譜熵 (1)
            tkeo_mean                         # TKEO 能量 (1)
        ])

        features.append(current_feature)

    return np.array(features)

def load_all_subjects():
    all_features = []
    all_labels = []
    all_subjects = []

    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Directory '{Config.DATASET_PATH}' not found")
        return None, None, None

    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])

    if len(subject_folders) < 2:
        print("Error: Not enough subjects. At least 2 subject folders are required for cross-validation.")
        return None, None, None

    print(f"Found {len(subject_folders)} subjects. Loading data...")

    segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        segments_by_class = [[] for _ in Config.CLASS_NAMES]

        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            pattern = os.path.join(subject_folder, f"*_{class_idx + 1}_*.txt")
            for file in sorted(glob.glob(pattern)):
                try:
                    data = np.loadtxt(file)
                    segs = create_segments(data, segment_length_samples, overlap_samples)
                    segments_by_class[class_idx].extend(segs)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

        if any(len(segs) == 0 for segs in segments_by_class):
            print(f"Warning: Insufficient data for {subject_id}. Skipping.")
            continue

        subject_features = []
        subject_labels = []
        for class_idx, segs in enumerate(segments_by_class):
            feats = extract_features(segs)
            subject_features.append(feats)
            subject_labels.append(np.full(len(feats), class_idx))

        subject_features = np.vstack(subject_features)
        subject_labels = np.hstack(subject_labels)

        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend([subject_id] * len(subject_labels))

        counts = ", ".join(f"{len(segs)} {name}"
                           for name, segs in zip(Config.CLASS_NAMES, segments_by_class))
        print(f" - {subject_id}: Successfully loaded {counts} segments")

    if not all_features:
        return None, None, None

    return np.vstack(all_features), np.hstack(all_labels), all_subjects


class MultiModelBCIClassifier:
    """Wrapper class supporting multiple classifier evaluations"""
    def __init__(self, model_type='MLP'):
        self.model_type = model_type

        # Model Selection
        if model_type == 'MLP':
            self.model = MLPClassifier(
                hidden_layer_sizes=Config.HIDDEN_LAYERS,
                max_iter=Config.MAX_ITER,
                learning_rate_init=Config.LEARNING_RATE,
                alpha=Config.ALPHA,
                activation='relu',
                solver='adam',
                batch_size=32,
                early_stopping=True,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'SVM':
            self.model = SVC(
                kernel=Config.SVM_KERNEL,
                C=Config.SVM_C,
                probability=True,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'RF':
            self.model = RandomForestClassifier(
                n_estimators=Config.RF_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'SelfSelected':
            # ==========================================
            # === STUDENT MODEL HERE ===================
            # ==========================================
            self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None

    def fit(self, X, y):
        if self.model is None:
            raise NotImplementedError("Model is currently blank (None).")

        X_scaled = self.scaler.fit_transform(X)
        if self.feature_selector is not None:
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X_scaled.shape[1])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        self.model.fit(X_selected, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise NotImplementedError("Model is currently blank (None).")

        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        raw_predictions = self.model.predict(X_selected)

        # ==========================================
        # === STUDENT POSTPROCESSING HERE ==========
        # ==========================================
        window_size = 3
        smoothed_predictions = np.copy(raw_predictions)
        for i in range(1, len(raw_predictions) - 1):
            window = raw_predictions[i-1:i+2]
            counts = np.bincount(window)
            smoothed_predictions[i] = np.argmax(counts)

        return smoothed_predictions


def leave_one_subject_out_validation(model_type, X, y, subjects):
    n_classes = len(Config.CLASS_NAMES)
    unique_subjects = sorted(set(subjects))
    results = {'accuracies': [], 'confusion_matrices': [], 'subject_names': []}

    for test_subject in tqdm(unique_subjects, desc=f"LOSO ({model_type:12})", leave=False):
        train_mask = [s != test_subject for s in subjects]
        test_mask = [s == test_subject for s in subjects]

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        classifier = MultiModelBCIClassifier(model_type=model_type)
        try:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        except NotImplementedError:
            return None

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

        results['accuracies'].append(accuracy)
        results['confusion_matrices'].append(cm)
        results['subject_names'].append(test_subject)

    return results

# ==============================================================
#  Reporting: accuracy summary + confusion matrix
# ==============================================================

def per_class_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)      
    predicted = cm.sum(axis=0)    

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return recall, precision, f1, support.astype(int), predicted.astype(int)


def format_summary(all_results, best_model_name):
    lines = []
    lines.append("=" * 50)
    lines.append("Summary of Classification Results (LOSO):")
    for model_type, res in all_results.items():
        mean_acc = np.mean(res['accuracies'])
        std_acc = np.std(res['accuracies'])
        lines.append(f" - {model_type:12}: {mean_acc:.3f} +/- {std_acc:.3f}")

    best_results = all_results[best_model_name]
    best_acc = np.mean(best_results['accuracies'])
    lines.append("")
    lines.append(f"Best Performing Model: {best_model_name} with Accuracy {best_acc:.3f}")
    if best_acc >= Config.TARGET_ACCURACY:
        lines.append(f"  -> reached the {Config.TARGET_ACCURACY:.0%} target")
    else:
        lines.append(f"  -> below the {Config.TARGET_ACCURACY:.0%} target")

    lines.append("")
    lines.append("Accuracy per held-out subject:")
    subject_names = best_results['subject_names']
    header = f"{'Subject':<16}" + "".join(f"{m:>14}" for m in all_results)
    lines.append(header)
    for i, subject in enumerate(subject_names):
        row = f"{subject:<16}" + "".join(f"{all_results[m]['accuracies'][i]:>14.3f}"
                                         for m in all_results)
        lines.append(row)

    total_cm = np.sum(best_results['confusion_matrices'], axis=0)
    recall, precision, f1, support, predicted = per_class_metrics(total_cm)

    lines.append("")
    lines.append(f"Detailed Class Report for Best Model [{best_model_name}]:")
    for k, class_name in enumerate(Config.CLASS_NAMES):
        tp = int(total_cm[k, k])
        lines.append(f"[{class_name} Class]")
        lines.append(f"  - Accuracy (Recall): {recall[k]:.3f} ({tp}/{support[k]})")
        lines.append(f"  - Precision: {precision[k]:.3f} ({tp}/{predicted[k]})")
        lines.append(f"  - F1-score: {f1[k]:.3f}")
        lines.append("")

    lines.append(f"Overall Confusion Matrix [{best_model_name}] (rows = true, cols = predicted):")
    header = f"{'':<10}" + "".join(f"{name:>10}" for name in Config.CLASS_NAMES)
    lines.append(header)
    for k, class_name in enumerate(Config.CLASS_NAMES):
        row = f"{class_name:<10}" + "".join(f"{int(v):>10}" for v in total_cm[k])
        lines.append(row)

    return "\n".join(lines)


def plot_comparison_results(all_results, best_model_name):
    model_types = list(all_results.keys())
    mean_accuracies = [np.mean(all_results[m]['accuracies']) for m in model_types]
    std_accuracies = [np.std(all_results[m]['accuracies']) for m in model_types]
    colors = [Config.MODEL_COLORS.get(m, Config.FALLBACK_COLOR) for m in model_types]

    best_results = all_results[best_model_name]
    subject_names = best_results['subject_names']
    class_names = Config.CLASS_NAMES

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BCI Model Classification Comparison - LOSO Results', fontsize=16)
    ax_models, ax_subjects, ax_cm, ax_cm_norm = axes.ravel()

    def style_axis(ax):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#d9d8d3', linewidth=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#d9d8d3')

    bars = ax_models.bar(model_types, mean_accuracies, yerr=std_accuracies,
                         width=0.6, color=colors,
                         error_kw=dict(ecolor=Config.REFERENCE_COLOR,
                                       capsize=4, lw=1.2))
    ax_models.set_title('Overall Mean Accuracy Comparison')
    ax_models.set_ylabel('Mean Accuracy')
    ax_models.set_ylim(0, 1)
    ax_models.axhline(y=Config.TARGET_ACCURACY, color=Config.REFERENCE_COLOR,
                      linestyle='--', linewidth=1.2,
                      label=f'Target: {Config.TARGET_ACCURACY:.2f}')
    ax_models.legend(frameon=False, fontsize=9)
    style_axis(ax_models)
    for bar, mean_acc, std_acc in zip(bars, mean_accuracies, std_accuracies):
        ax_models.text(bar.get_x() + bar.get_width() / 2.0, mean_acc + std_acc + 0.02,
                       f'{mean_acc:.3f}', ha='center', va='bottom',
                       fontweight='bold', color='#0b0b0b')

    n_models = len(model_types)
    x = np.arange(len(subject_names))
    group_width = 0.78
    width = group_width / n_models
    for i, model_type in enumerate(model_types):
        offset = (i - (n_models - 1) / 2) * width
        ax_subjects.bar(x + offset, all_results[model_type]['accuracies'],
                        width * 0.92,  
                        label=model_type, color=colors[i])
    ax_subjects.set_title('Accuracy by Held-out Subject (LOSO)')
    ax_subjects.set_ylabel('Accuracy')
    ax_subjects.set_xticks(x)
    ax_subjects.set_xticklabels(subject_names)
    ax_subjects.set_ylim(0, 1)
    ax_subjects.axhline(y=Config.TARGET_ACCURACY, color=Config.REFERENCE_COLOR,
                        linestyle='--', linewidth=1.2,
                        label=f'Target: {Config.TARGET_ACCURACY:.2f}')
    ax_subjects.legend(frameon=False, fontsize=9, ncol=n_models + 1,
                       loc='upper center', bbox_to_anchor=(0.5, -0.06))
    style_axis(ax_subjects)

    total_cm = np.sum(best_results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap=Config.HEATMAP_CMAP, cbar=False,
                linewidths=2, linecolor='white',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix - counts ({best_model_name})')
    ax_cm.set_xlabel('Predicted', labelpad=20)
    ax_cm.set_ylabel('Actual', labelpad=50)
    ax_cm.tick_params(length=0)

    row_sums = total_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.divide(total_cm, row_sums,
                            out=np.zeros(total_cm.shape, dtype=float),
                            where=row_sums > 0)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=Config.HEATMAP_CMAP,
                vmin=0, vmax=1, cbar=False, linewidths=2, linecolor='white',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm_norm)
    ax_cm_norm.set_title(f'Confusion Matrix - row-normalized ({best_model_name})')
    ax_cm_norm.set_xlabel('Predicted', labelpad=20)
    ax_cm_norm.set_ylabel('Actual', labelpad=50)
    ax_cm_norm.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(Config.FIGURE_NAME, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("BCI EEG Classification - Evaluation")
    print("=" * 60)

    X, y, subjects = load_all_subjects()
    if X is None:
        print("Validation aborted. Please check the dataset directory.")
        return
    print(f"Feature matrix: {X.shape[0]} segments x {X.shape[1]} features")

    models_to_evaluate = ['MLP', 'SVM', 'RF', 'SelfSelected']
    all_results = {}

    for model_type in models_to_evaluate:
        print(f"\nRunning Cross-Validation for model: [{model_type}]...")
        results = leave_one_subject_out_validation(model_type, X, y, subjects)

        if results is None:
            if model_type == 'SelfSelected':
                print("[Info] 'SelfSelected' is blank/None. Skipping evaluation.")
                continue
            else:
                print(f"Validation failed for model {model_type}.")
                return

        all_results[model_type] = results
        mean_acc = np.mean(results['accuracies'])
        print(f"-> [{model_type}] Overall Mean Accuracy: {mean_acc:.3f}")

    if not all_results:
        print("No model was evaluated successfully.")
        return

    best_model_name = max(all_results,
                          key=lambda m: np.mean(all_results[m]['accuracies']))

    summary = format_summary(all_results, best_model_name)
    print("\n" + summary)
    with open(Config.SUMMARY_NAME, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")

    plot_comparison_results(all_results, best_model_name)
    print(f"\nResults saved in '{Config.FIGURE_NAME}' and '{Config.SUMMARY_NAME}'")


if __name__ == "__main__":
    main()