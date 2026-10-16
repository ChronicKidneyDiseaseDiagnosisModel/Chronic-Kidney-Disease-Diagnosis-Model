import cv2 as cv
import numpy as np
import warnings
from matplotlib import pylab, patches
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def Plot_Conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'NBOA-IFD-STN', 'RBMO-IFD-STN', 'DOA-IFD-STN', 'POA-IFD-STN', 'RFPOA-IFD-STN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[1], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('--------------------------------------------------  Statistical Report',
          ' --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergenc.png")
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def ROC_curve():
    lw = 2
    cls = ['CNN-SVM', ' FuDNN', 'XceptionNet', 'IFD-STN', 'RFPOA-IFD-STN']
    Actual = np.load('Targets.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#00ffff", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('ROC curve')
    plt.savefig(path)
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    Classes = ['Normal', 'Abnormal']
    rot = 0
    fontsizes = 12
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
    ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
    ax.set_yticklabels(Classes, fontsize=fontsizes)
    plt.tight_layout()
    path = "./Results/Confusion_matrix.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def PLot_ACTIVATION():
    eval = np.load('Eval_ALL_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Graph_Terms = [0, 3, 4, 5, 6, 8, 9, 12, 13]
    Algorithm = ['NBOA-IFD-STN', 'RBMO-IFD-STN', 'DOA-IFD-STN', 'POA-IFD-STN', 'RFPOA-IFD-STN']
    Classifier = ['CNN-SVM', ' FuDNN', 'XceptionNet', 'IFD-STN', 'RFPOA-IFD-STN']
    ACT = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid']
    Alg_Colors = ["#8E44AD", "#E67E22", "#27AE60", "#2980B9", "#E91E63"]
    Met_Colors = ["#2C3E50", "#D35400", "#16A085", "#34495E", "#C0392B"]

    for j in range(len(Graph_Terms)):
        Graphs = eval[:5, :, Graph_Terms[j] + 4]

        Graph = Graphs[:, :5]
        bar_width = 0.1
        bar_gap = 0.1
        group_gap = 0.6
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        x = np.arange(Graph.shape[0]) * (1 + group_gap)
        n = len(Alg_Colors)
        for i in range(n):
            xpos = x + i * (bar_width + bar_gap)
            # glow
            for glow_w, alpha in [(0.35, 0.05), (0.25, 0.1)]:
                plt.bar(xpos, Graph[:, i], width=glow_w, color=Alg_Colors[i], alpha=alpha, zorder=1)
            # main bar
            plt.bar(xpos, Graph[:, i], width=bar_width, color=Alg_Colors[i], zorder=2)
        total_width = (n - 1) * (bar_width + bar_gap)
        for k in range(len(x) - 1):
            sep_x = (x[k] + x[k + 1]) / 2 + total_width / 2
            plt.axvline(x=sep_x, color='black', linewidth=1.5, alpha=0.6, linestyle='-', zorder=0)
        plt.xticks(x + total_width / 2, ACT, fontsize=10)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        legend_elements = [Line2D([0], [0], color=Alg_Colors[i], lw=6, label=Algorithm[i]) for i in
                           range(len(Algorithm))]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # plt.tight_layout()
        path = f"./Results/ACT {Terms[Graph_Terms[j]]} ALG.png"
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(f'ACT vs {Terms[Graph_Terms[j]]}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        # Method
        Graph = Graphs[:, 5:]
        bar_width = 0.1
        bar_gap = 0.1
        group_gap = 0.6
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        x = np.arange(Graph.shape[0]) * (1 + group_gap)
        n = len(Met_Colors)
        for i in range(n):
            xpos = x + i * (bar_width + bar_gap)
            # glow
            for glow_w, alpha in [(0.35, 0.05), (0.25, 0.1)]:
                plt.bar(xpos, Graph[:, i], width=glow_w, color=Met_Colors[i], alpha=alpha, zorder=1)
            # main bar
            plt.bar(xpos, Graph[:, i], width=bar_width, color=Met_Colors[i], zorder=2)
        total_width = (n - 1) * (bar_width + bar_gap)
        for k in range(len(x) - 1):
            sep_x = (x[k] + x[k + 1]) / 2 + total_width / 2
            plt.axvline(x=sep_x, color='black', linewidth=1.5, alpha=0.6, linestyle='-', zorder=0)
        plt.xticks(x + total_width / 2, ACT, fontsize=10)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        legend_elements = [Line2D([0], [0], color=Met_Colors[i], lw=6, label=Classifier[i]) for i in
                           range(len(Classifier))]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        path = f"./Results/ACT {Terms[Graph_Terms[j]]} MTD.png"
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(f'ACT vs {Terms[Graph_Terms[j]]}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def darken_color(hex_color, factor=0.5):
    rgb = mcolors.hex2color(hex_color)
    dark_rgb = tuple([max(0, c * factor) for c in rgb])
    return mcolors.to_hex(dark_rgb)


def draw_custom_bar(ax, x, height, color, max_height):
    bar_width = 0.8

    # Normalize triangle height (adaptive scaling)
    triangle_height = max_height * 0.04  # always 4% of max scale
    rect_height = height - triangle_height

    # Prevent negative height issue
    if rect_height < 0:
        rect_height = height * 0.8
        triangle_height = height * 0.2

    # Draw rectangle (main bar)
    rect = patches.Rectangle(
        (x - bar_width / 2, 0),
        bar_width,
        rect_height,
        facecolor=color,
        edgecolor='none',
        zorder=2
    )
    ax.add_patch(rect)

    # Draw triangle (top)
    triangle = patches.Polygon(
        [
            (x - bar_width / 2, rect_height),
            (x + bar_width / 2, rect_height),
            (x, rect_height + triangle_height)
        ],
        closed=True,
        facecolor=color,
        edgecolor='none',
        zorder=3
    )
    ax.add_patch(triangle)


def Plot_Feature():
    eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Graph_Terms = [0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    Classifier = ['PCA', ' DBN', 'AE', 'SAE', 'SSAE-LReLU']
    learn_per = ['35 %', '45 %', '55 %', '65 %', '75 %']
    colors_Alg = ['#D81B60', '#00ACC1', '#7CB342', '#FFB300', '#6D4C41']
    colors_Mtd = ['#a541c8', '#e67e22', '#c0392b', '#7f8c8d', '#6D4C41']
    for j in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[j] + 4]
        Algorithm_data = Graph[:, :5]
        groups = Algorithm_data.shape[0]
        bars_per_group = Algorithm_data.shape[1]
        max_data = np.max(Algorithm_data)
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#F8F9FA')
        for g in range(groups):
            for b in range(bars_per_group):
                x_pos = (g * (bars_per_group + 3)) + (b * 1.3)
                draw_custom_bar(ax, x_pos, Algorithm_data[g, b], colors_Alg[b], max_data)
                value = Algorithm_data[g, b]
                text_y = max_data * 0.02
        group_centers = []
        for g in range(groups):
            start = (g * (bars_per_group + 3))
            end = start + (bars_per_group - 1) * 1.3
            center = (start + end) / 2
            group_centers.append(center)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(learn_per, fontsize=11, fontweight='bold')
        ax.set_xlim(-2, group_centers[-1] + 5)
        legend_patches = [patches.Patch(color=colors_Alg[i], label=Classifier[i]) for i in range(len(Classifier))]
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False,
                  fontsize=12)
        ax.set_xlim(-2, (groups * (bars_per_group + 3.5)))
        ax.set_ylim(0, max_data * 1.1)
        ax.set_xlabel('Learning Percentage', fontsize=14, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=14, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        path = f"./Results/Feat {Terms[Graph_Terms[j]]} MTD.png"
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(f'Feat vs {Terms[Graph_Terms[j]]}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_KFold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Table_Term = [0, 3, 4, 6, 7, 8, 9, 10, 12]
    Kfold = ['1', '2', '3', '4', '5']
    Algorithm = ['K Fold', 'NBOA-IFD-STN', 'RBMO-IFD-STN', 'DOA-IFD-STN', 'POA-IFD-STN', 'RFPOA-IFD-STN']
    Classifier = ['K Fold', 'CNN-SVM', ' FuDNN', 'XceptionNet', 'IFD-STN', 'RFPOA-IFD-STN']
    for k in range(len(Table_Term)):
        value = eval[:5, :, Table_Term[k] + 4]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Kfold)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j])
        print('--------------------------------------------------', str(Terms[Table_Term[k]]),
              ' K Fold vs Algorithm Comparison Dataset  --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Kfold)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1])
        print('-------------------------------------------------- ', str(Terms[Table_Term[k]]),
              'K Fold vs Classifier Comparison Dataset  --------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    Plot_Conv()
    ROC_curve()
    Plot_Confusion()
    PLot_ACTIVATION()
    Plot_KFold()
    Plot_Feature()
