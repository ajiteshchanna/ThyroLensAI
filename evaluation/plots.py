"""Optional plots for evaluation outputs."""

import matplotlib.pyplot as plt


def save_confusion_matrix(matrix, path):
    values = [[matrix["true_negative"], matrix["false_positive"]], [matrix["false_negative"], matrix["true_positive"]]]
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(values, cmap="Blues")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], ["0", "1"])
    axis.set_yticks([0, 1], ["0", "1"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, values[row][column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_reliability_diagram(calibration, path):
    figure, axis = plt.subplots(figsize=(5, 4))
    rows = calibration.get("bins", [])
    axis.plot([0, 1], [0, 1], "--", color="gray")
    if rows:
        axis.plot([row["confidence"] for row in rows], [row["accuracy"] for row in rows], "o-")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Reliability diagram")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
