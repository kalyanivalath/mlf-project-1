# mlf-project-1

Artificial neural networks trained using supervised learning via error back propagation.

## Data organization

`Results/results.csv` contains one row per run (train, validate, and test) of the network.
Each dataset was run 50 times with 8 hidden units and 50 times with no hidden
layer, using seeds 0-49. All runs used a 70/20/10 train/validation/test split.

`Results/summary_results.csv` summarizes the results from `results.csv` by dataset
and number of hidden units. It contains the mean and standard deviation of the
training, validation, and test accuracy across the 50 runs.

### results.csv columns

- `dataset` - Name of the input data file
- `hidden` - Number of hidden units (0 = no hidden layer)
- `lr` - Learning rate
- `momentum` - Momentum
- `epochs` - Number of training epochs
- `batch` - Mini-batch size
- `seed` - Random seed for this run (controls the split, initial weights,
  and batch order)
- `train_acc` - Training accuracy (fraction correct, 0-1)
- `val_acc` - Validation accuracy
- `test_acc` - Test accuracy
- `best_epoch` - Epoch with the lowest validation loss; its weights were kept
- `test_TN` - Class 0 test points classified as Class 0
- `test_FP` - Class 0 test points classified as Class 1
- `test_FN` - Class 1 test points classified as Class 0
- `test_TP` - Class 1 test points classified as Class 1

### summary_results.csv columns

- `dataset` - Name of the dataset
- `hidden` - Number of hidden units
- `runs` - Number of runs summarized
- `train_mean` - Mean training accuracy
- `train_std` - Standard deviation of training accuracy
- `val_mean` - Mean validation accuracy
- `val_std` - Standard deviation of validation accuracy
- `test_mean` - Mean test accuracy
- `test_std` - Standard deviation of test accuracy
- `best_epoch_mean` - Mean best epoch

## Experiment setup

The original data was split into approximately 70% training, 20% validation, and 10% testing. The test data was held out before training and was only used to measure the final performance of the network.

For each dataset, the ANN was run 50 times with 8 hidden units and 50 times with no hidden layer. The random seed changed from 0-49 between runs.

Generated with:

for name in "Gaussian 2D Wide" "Gaussian 2D Narrow" "Gaussian 2D Overlap" \
            "Gaussian 3D Wide" "Gaussian 3D Narrow" "Gaussian 3D Overlap" \
            "Moons 2D Wide" "Moons 2D Narrow" "Moons 2D Overlap"
do
  python3 backprop_net.py "data/$name.csv" --runs 50 --plot --out Results/results.csv
  python3 backprop_net.py "data/$name.csv" --runs 50 --hidden 0 --plot --out Results/results.csv
done | tee Results/run_log.txt

The summarized results were generated with:
python3 summarize_res.py