# mlf-project-1
Artificial neural networks trained using supervised learning via error back propagation.














Data organization

results.csv contains one row per run (train, validate, and test) of the network.
Each dataset was run 50 times with 8 hidden units and 50 times with no hidden
layer, using seeds 0-49. All runs used a 70/20/10 train/validation/test split.

Columns:
  dataset      Name of the input data file
  hidden       Number of hidden units (0 = no hidden layer)
  lr           Learning rate
  momentum     Momentum
  epochs       Number of training epochs
  batch        Mini-batch size
  val, test    Validation and test fractions used for the split
  seed         Random seed for this run (controls the split, initial weights,
               and batch order)
  train_acc    Training accuracy (fraction correct, 0-1)
  val_acc      Validation accuracy
  test_acc     Test accuracy
  best_epoch   Epoch with the lowest validation loss; its weights were kept
  test_TN      Class 0 test points classified as Class 0
  test_FP      Class 0 test points classified as Class 1
  test_FN      Class 1 test points classified as Class 0
  test_TP      Class 1 test points classified as Class 1

Generated with:

for name in "Gaussian 2D Wide" "Gaussian 2D Narrow" "Gaussian 2D Overlap" \
                     "Gaussian 3D Wide" "Gaussian 3D Narrow" "Gaussian 3D Overlap" \
                     "Moons 2D Wide" "Moons 2D Narrow" "Moons 2D Overlap" 
do
  python3 backprop_net.py "data/$name.csv" --runs 50 --plot --out results.csv
  python3 backprop_net.py "data/$name.csv" --runs 50 --hidden 0 --plot --out results.csv
done | tee run_log.txt