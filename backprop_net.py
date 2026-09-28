import argparse
import numpy as np



#--------------------------------------------------------------------------------------------------
# data
#--------------------------------------------------------------------------------------------------
    
    
   

def load_csv(path):
    # first half of columns class 0 and second half class 1
    A = np.loadtxt(path, delimiter=",")
    d = A.shape[1] // 2                       #2 for the 2D files, 3 for the 3D
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
            # LeCun initialization keeps tanh units out of saturation at the start
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
            H = np.tanh(X @ self.W1 + self.b1)
        else:
            H = X

        out = sigmoid(H @ self.W2 + self.b2)
        return H, out
    
    def backward(self, X, y, H, out):
        n = len(X)

        #with sigmoid output + cross entropy, dL/dz_out  simplifies to (out - y)
        d_out = (out - y) / n
        grads = {"W2": H.T @ d_out, "b2": d_out.sum(axis=0, keepdims=True), }  

        if self.n_hidden > 0:
            d_hidden = (d_out @ self.W2.T) * (1.0 - H ** 2)  #tanh'(z) = 1- tanh(z)^2
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

def train(net, Xtr, ytr, Xval, yval, lr, momentum, epochs, batch, patience, rng, verbose=True):
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
        elif ep - best [2] >= patience:
            if verbose:
                print(f"Early stop at epoch {ep} (best epoch {best[2]})")
            break
        if verbose and (ep % max(1, epochs // 10) == 0 or ep ==1):
            print(f"epoch {ep:5d} train loss {tr_loss:.4f} val loss {va_loss:.4f} "
                  f"train acc {accuracy(net,Xtr, ytr):.3f}")

    for k, v in best[1].items():        #restore weights
        net.params()[k][...] = v
    return history

#--------------------------------------------------------------------------------------------------
# plotting
#--------------------------------------------------------------------------------------------------
            







#--------------------------------------------------------------------------------------------------
# main
#--------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hidden", type=int, default=8, help= "hidden units (0 = no hidden layer)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val", type=float, default=0.2, help="fraction held out for validation")
    ap.add_argument("--patience", type=int, default=200, help="epochs without validation improvement before stopping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    X, y = load_csv(a.csv)
    print(f"{len(X)} examples, {X.shape[1]} features")

    Xtr, ytr, Xval, yval = train_val_split(X, y, a.val, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12  # standardize using training stats only
    Xtr, Xval = (Xtr - mu) / sd, (Xval - mu) / sd

    net = BackpropNet(X.shape[1], a.hidden, rng)
    hist = train(net, Xtr, ytr, Xval, yval, a.lr, a.momentum, a.epochs, a.batch, a.patience, rng)

    print(f"\nFinal train accuracy:     {accuracy(net, Xtr, ytr):.3f}")
    print(f"Final validation accuracy:     {accuracy(net, Xval, yval):.3f}")

    # if a.plot:
    #     Xall = (X - mu) / sd 
    #     plot_results(net, Xall, y, hist, mu,  sd, f"{a.csv} (hidden={a.hidden}, lr={a.lr})",
    #                  a.csv.rsplit(".", 1)[0] + "_results.png")

if __name__ == "__main__":
    main()

    

