import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load CSV files
# -----------------------------
gcn_csv = 'results/training_results.csv'
laplacian_csv = 'results/hypergcn_laplacian_training_results.csv'

gcn_data = pd.read_csv(gcn_csv)
laplacian_data = pd.read_csv(laplacian_csv)
plt.rcParams.update({
    'figure.figsize': (4, 2.5),
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 17,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 3,
    'axes.linewidth': 1.3,
})
# -----------------------------
# Plot Loss
# -----------------------------
plt.figure()
plt.plot(laplacian_data['Epoch'], laplacian_data['Loss'], '-', label='HyperGCN')
plt.plot(gcn_data['Epoch'], gcn_data['Loss'], '-', label='GCN')
plt.xlabel('Epochs')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss_plot.pdf',
    format='pdf',
    bbox_inches='tight',
    metadata={'Creator': 'Matplotlib'}
)
# plt.show()

# -----------------------------
# Plot Train Accuracy
# -----------------------------
plt.figure()
plt.plot(laplacian_data['Epoch'], laplacian_data['Train Acc'], '-', label='HyperGCN')
plt.plot(gcn_data['Epoch'], gcn_data['Train Acc'], '-', label='GCN')
plt.xlabel('Epochs')
plt.title('Train Accuracy')
plt.grid(True)
plt.savefig('train_accuracy_plot.pdf',
    format='pdf',
    bbox_inches='tight',
    metadata={'Creator': 'Matplotlib'}
)
# plt.show()

# -----------------------------
# Plot Validation Accuracy
# -----------------------------
plt.figure()
plt.plot(laplacian_data['Epoch'], laplacian_data['Val Acc'], '-', label='HyperGCN')
plt.plot(gcn_data['Epoch'], gcn_data['Val Acc'], '-', label='GCN')
plt.xlabel('Epochs')
plt.title('Validation Accuracy')
plt.grid(True)
plt.savefig('val_accuracy_plot.pdf',
    format='pdf',
    bbox_inches='tight',
    metadata={'Creator': 'Matplotlib'}
)
# plt.show()

# -----------------------------
# Plot Test Accuracy
# -----------------------------
plt.figure()
plt.plot(laplacian_data['Epoch'], laplacian_data['Test Acc'], '-', label='HyperGCN')
plt.plot(gcn_data['Epoch'], gcn_data['Test Acc'], '-', label='GCN')
plt.xlabel('Epochs')
plt.title('Test Accuracy')
plt.legend(handlelength=3)
plt.grid(True)
plt.savefig('test_accuracy_plot.pdf',
    format='pdf',
    bbox_inches='tight',
    metadata={'Creator': 'Matplotlib'}
)
# plt.show()
