# Feedforward backprop network for two-class classification
#
# Each run splits data into training, validation, and test sets
# Trains a fresh network for a fixed number of epochs
# Keeps the weights from the epoch with the lowest validation loss
# measures accuracy on the held out test set
#

# usage: 
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv"                             # one run full progress printed
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv" --runs 50                   # 50 runs + summary, printed
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv" --hidden 0                  # no hidden layer
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv" --hidden 8                  # 8 hidden units
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv" --plot                      # also saves plots (first run) 
#   python3 backprop_net.py "data/Gaussian 2D Wide.csv" --runs 50 --out results.csv # also saves results

# Output: results are printed to terminal only nothing is saved unless requested
#   --plot saves a learning curve + decision boundary image from the first run, 
#          next to the data file (ex data/Gaussian 2D Wide_results.png)
#   
#   --out appends one row per run to the named CSV file one is (created if missing, existing rows are kept delete the file to start fresh)
#   
# Options can be combined such as --runs 50 --plot --out results.csv

# CSV format: no header and no label column. Each row holds one Class 0 point(first half of the columns) and the Class 1 point (second half)
#
#
# Shell script for running the experiments:
#       for name in "Gaussian 2D Wide" "Gaussian 2D Narrow" "Gaussian 2D Overlap" \
#                     "Gaussian 3D Wide" "Gaussian 3D Narrow" "Gaussian 3D Overlap" \
#                     "Moons 2D Wide" "Moons 2D Narrow" "Moons 2D Overlap" 
#       do
#       python3 backprop_net.py "data/$name.csv" --runs 50 --plot --out results.csv
#       python3 backprop_net.py "data/$name.csv" --runs 50 --hidden 0 --plot --out results.csv
#       done | tee run_log.txt
#
#
#   -saves the terminal output to run_log.txt, saves the plots to *_results.png files, and appends results to results.csv
#   -must delete the results.csv, the run_log.txt, and the *_results.png files before re-running 
#   -if not the old results may be overwritten in the cases of the plots and terminal output run_log.txt 
#    and appended to the results.csv file



import argparse
import numpy as np
import csv
import os
import matplotlib.pyplot as plt



#--------------------------------------------------------------------------------------------------
# data
#--------------------------------------------------------------------------------------------------
    
    
   

def load_csv(path):
    # first half of columns class 0 and second half class 1
    A = np.loadtxt(path, delimiter=",")
    d = A.shape[1] // 2                       # 2 for the 2D files, 3 for the 3D
    X = np.vstack([A[:, :d], A[:,d:]])        # class 0 rows on top, class 1 rows below
    y = np.r_[np.zeros(len(A)), np.ones(len(A))].reshape(-1,1)
    return X, y


def train_val_split(X, y, val_frac, rng):
    idx = rng.permutation(len(X))
    n_val = int(round(val_frac * len(X)))
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]

#--------------------------------------------------------------------------------------------------
# activations
#--------------------------------------------------------------------------------------------------


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))



#--------------------------------------------------------------------------------------------------
# network
#--------------------------------------------------------------------------------------------------

class BackpropNet:
    #Input --> [tanh hidden layer] --> 1 sigmoid output, trained on binary cross entropy

    def __init__(self, n_in, n_hidden, rng):
        self.n_hidden = n_hidden
        if n_hidden > 0:
            # init keeps tanh units out of saturation at the start
            self.W1 = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_hidden))
            self.b1 = np.zeros((1, n_hidden))
            n_last = n_hidden
        else:
            n_last = n_in

        self.W2 = rng.normal(0, np.sqrt(1.0 / n_last), (n_last, 1))
        self.b2 = np.zeros((1,1)) 
        self.vel = {k:np.zeros_like(v) for k, v in self.params().items()} 

    def params(self):
        p = {"W2": self.W2, "b2": self.b2}
        if self.n_hidden > 0:
            p.update(W1=self.W1, b1=self.b1)

        return p

    def forward(self, X):
        if self.n_hidden > 0:
            # H = np.sigmoid(X @ self.W1 + self.b1)  #sigmoid for hidden layers
            H = np.tanh(X @ self.W1 + self.b1)       #tanh for hidden layers
        else:
            H = X

        out = sigmoid(H @ self.W2 + self.b2)
        return H, out
    
    def backward(self, X, y, H, out):
        n = len(X)

        #with sigmoid output + cross entropy, dL/dz_out simplifies to (out - y)
        d_out = (out - y) / n
        grads = {"W2": H.T @ d_out, "b2": d_out.sum(axis=0, keepdims=True), }  

        if self.n_hidden > 0:
            # d_hidden = (d_out @ self.W2.T) * (H * (1.0 - H))  #sigmoid'(z) = s(z)(1 - s(z))   #sigmoid for hidden layers
            d_hidden = (d_out @ self.W2.T) * (1.0 - H ** 2)  #tanh'(z) = 1- tanh(z)^2           #tanh for hidden layers
            grads["W1"] = X.T @ d_hidden
            grads["b1"] = d_hidden.sum(axis=0, keepdims=True)
        return grads

    def step(self, grads, lr, momentum):
        for k, p in self.params().items():
            self.vel[k] = momentum * self.vel[k] - lr * grads[k]
            p += self.vel[k]                 #in place update of the weight array

    def predict_prob(self, X):
        return self.forward(X)[1]

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(float)

def bce(out, y, eps=1e-12):
    return -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))
    
def accuracy(net, X, y):
    return float(np.mean(net.predict(X) == y))
    
#--------------------------------------------------------------------------------------------------
# training
#--------------------------------------------------------------------------------------------------

# Train for a fixed number of epochs, then restore the weights 
# from the epoch with the lowest validation loss.
# Returns the loss history and the best epoch
    

def train(net, Xtr, ytr, Xval, yval, lr, momentum, epochs, batch, rng, verbose=True):
    
    best = (np.inf, None, 0)
    history = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            b = idx[s:s + batch]
            H, out = net.forward(Xtr[b])
            net.step(net.backward(Xtr[b], ytr[b], H, out), lr, momentum)

        tr_loss = bce(net.predict_prob(Xtr), ytr)
        va_loss = bce(net.predict_prob(Xval), yval)
        history.append((tr_loss, va_loss))

        if va_loss < best[0] - 1e-6:
            best = (va_loss, {k: v.copy() for k, v in net.params().items()}, ep)
        
        if verbose and (ep % max(1, epochs // 10) == 0 or ep ==1):
            print(f"epoch {ep:5d} train loss {tr_loss:.4f} val loss {va_loss:.4f} "
                  f"train acc {accuracy(net,Xtr, ytr):.3f}")

    #restore best weights
    for k, v in best[1].items():        
        net.params()[k][...] = v
    return history, best[2]


def run_once(X, y, a, seed, verbose):
    # one complete run
    # split, standardize train, and measure
    #returns a results dictionary

    rng = np.random.default_rng(seed)

    #hold out test data first
    Xrest, yrest, Xte, yte = train_val_split(X, y, a.test, rng) 

    #spit the rest into train/val
    Xtr, ytr, Xval, yval = train_val_split(Xrest, yrest, a.val, rng)

    # standardize using training stats only
    mu, sd = Xtr.mean(0), Xtr.std(0) 
    Xtr, Xval, Xte = (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd

    net = BackpropNet(X.shape[1], a.hidden, rng)
    hist, best_ep = train(net, Xtr, ytr, Xval, yval, a.lr, a.momentum, a.epochs, a.batch, rng, verbose=verbose)

    pred = net.predict(Xte)
    results = {
        "seed": seed,
        "train_acc": accuracy(net, Xtr, ytr), 
        "val_acc": accuracy(net, Xval, yval),
        "test_acc": accuracy(net, Xte, yte), 
        "best_epoch": best_ep, 
        "test_TN": int(np.sum((pred == 0) & (yte == 0))),   # Class 0 correctly called 0
        "test_FP": int(np.sum((pred == 1) & (yte == 0))),   # Class 0 wrongly called 1  
        "test_FN": int(np.sum((pred == 0) & (yte == 1))),   # Class 1 wrongly called 0
        "test_TP": int(np.sum((pred == 1) & (yte == 1))),   # Class 1 correctly called 1 

    }
    return results, net, hist, mu, sd



#--------------------------------------------------------------------------------------------------
# plotting
#--------------------------------------------------------------------------------------------------
            
def plot_results(net, X, y, history, mu, sd, title, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tr, va = zip(*history)
    axes[0].plot(tr, label="train")
    axes[0].plot(va, label="validation")
    axes[0].set(xlabel="epochs", ylabel="cross-entropy loss", title="Learning curve")
    axes[0].legend()

    if X.shape[1] == 2:
        Xo = X * sd + mu
        pad = 0.1 * (Xo.max(0) - Xo.min(0))
        g0, g1 = np.meshgrid(np.linspace(Xo[:,0].min() - pad[0], Xo[:,0].max() + pad[0], 300), 
                             np.linspace(Xo[:,1].min() - pad[1], Xo[:,1].max() + pad[1], 300))
        grid = (np.c_[g0.ravel(), g1.ravel()] - mu) / sd

        P = net.predict_prob(grid).reshape(g0.shape)
        axes[1].contourf(g0, g1, P, levels=20, cmap="RdBu_r", alpha=0.6)
        axes[1].contour(g0, g1, P, levels=[0.5], colors="k")
        axes[1].scatter(Xo[:,0], Xo[:,1], c=y.ravel(), cmap="RdBu_r", edgecolors="k", s=18)
        axes[1].set(title="Decision boundary (p = 0.5)", xlabel="x1", ylabel="x2")
    else:
        axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    print(f"Saved plot to {outfile}")







#--------------------------------------------------------------------------------------------------
# main
#--------------------------------------------------------------------------------------------------
# for setting the split of training/validation/test
# first select --test (fraction) 
# then of what is left determine how much fraction your validation is by 
# --val default = (validation_fraction_desired / fraction that is left after removing test fraction)
# for 70training/20validation/10test you use the value of --test default=0.1  --val default=0.2222
# for 70training/15validation/15test you use the value of --test default=0.15 --val defaault=0.1765

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hidden", type=int, default=8, help="hidden units (0 = no hidden layer)")
    ap.add_argument("--lr", type=float, default=0.1, help="learning rate")
    ap.add_argument("--momentum", type=float, default=0.9, help="momentum")
    ap.add_argument("--epochs", type=int, default=2000, help="How many epochs each run trains")
    ap.add_argument("--batch", type=int, default=32, help="mini-batch size")
    ap.add_argument("--test", type=float, default=0.1, help="fraction held out for testing from original data set")
    ap.add_argument("--val", type=float, default=0.2222, help="fraction held out for validation from what is left of the data set after removing test")
    ap.add_argument("--seed", type=int, default=0, help="seed for the first run")
    ap.add_argument("--runs", type=int, default=1, help="number of runs (seeds seed ... seed + runs - 1)")
    ap.add_argument("--out", help="CSV file to append per-run results to")
    ap.add_argument("--plot", action="store_true", help="save plots from the first run")
    
    a = ap.parse_args()

    X, y = load_csv(a.csv)
    name = os.path.splitext(os.path.basename(a.csv))[0]
    print(f"{name}: {len(X)} examples, {X.shape[1]} features, hidden={a.hidden}, runs={a.runs}")

    results = []
    for i in range(a.runs):
        seed = a.seed + i
        r, net, hist, mu, sd = run_once(X,y,a,seed,verbose=(a.runs == 1))
        results.append(r)

        if a.runs == 1:
            print(f"\nBest epoch (lowest validation loss): {r['best_epoch']}")
            print(f"Final train accuracy: {r['train_acc']:.3f}")
            print(f"Final validation accuracy: {r['val_acc']:.3f}")
            print(f"Final test accuracy: {r['test_acc']:.3f}")
            print(f"Test errors: {r['test_FP']} Class 0 called Class 1, "
                  f"{r['test_FN']} Class 1 called Class 0 ")
        else:
            print(f"run {i + 1:3d} seed ({seed:3d}) train {r['train_acc']:.3f} " 
                  f"val {r['val_acc']:.3f} test {r['test_acc']:.3f} best epoch {r['best_epoch']}")
        if a.plot and i == 0:
            plot_results(net, (X - mu) / sd, y, hist, mu ,sd, 
                         f"{name}  (hidden={a.hidden}, lr={a.lr}, seed={seed})", f"{name}_h{a.hidden}_results.png")  

    if a.runs> 1:
        def summary(key):
            v = np.array([r[key] for r in results], dtype=float)
            return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}"
        print(f"\nSummary over {a.runs} runs (mean +/- standard deviation):")
        print(f"  train accuracy:       {summary('train_acc')}")
        print(f"  validation accuracy:  {summary('val_acc')}")
        print(f"  test accuracy:        {summary('test_acc')}")
        print(f"  best epoch:           {summary('best_epoch')}")
        print(f"  total test errors:    {sum(r['test_FP'] for r in results)} Class 0 called Class 1, "
              f"{sum(r['test_FN'] for r in results)} Class 1 called Class 0")

    if a.out:
        fields = ["dataset", "hidden", "lr", "momentum", "epochs", "batch"] + list(results[0])
        new_file = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for r in results:
                w.writerow({"dataset": name, "hidden": a.hidden, "lr": a.lr, "momentum": a.momentum, "epochs": a.epochs,
                            "batch": a.batch, **r})
        print(f"Appended {len(results)} runs to {a.out}")


if __name__ == "__main__":
    main()

    

