import torch as th

"""
Author: Josue N Rivera (github.com/JosueCom)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using PyTorch
Full project repository: https://github.com/JosueCom/Lign (A graph deep learning framework that works alongside PyTorch)
"""


def randomize_tensor(tensor):
    return tensor[th.randperm(len(tensor))]


def distance_matrix(x, y=None, p=2, dist="L2"):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    if dist=="L2":
        return th.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
    elif dist=="cosine":
        return th.cosine_similarity(x.unsqueeze(2), y.permute(1, 0).unsqueeze(0))
    else:
        raise NotImplementedError()


class NN:

    def __init__(self, X=None, Y=None, p=2):
        self.train_label = None
        self.train_pts = None
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x, distance_metric="L2"):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p, distance_metric)
        labels = th.argmin(dist, dim=1) if distance_metric == "L2" else th.argmax(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.unique_labels = None
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


class KMeans(NN):

    def __init__(self, X, k, n_iters=10, p=2, distance_metric="cosine"):
        self.k = k
        self.n_iters = n_iters
        self.p = p
        self.train(X, distance_metric)

    def train(self, X, distance_metric):

        self.train_pts = randomize_tensor(X)[:self.k]
        self.train_label = th.tensor(range(self.k), dtype=th.int64, device=X.device)

        for _ in range(self.n_iters):
            labels = self.predict(X, distance_metric)

            for lab in range(self.k):
                self.train_pts[lab] = th.mean(X[labels == lab], dim=0)


class KMedians(NN):

    def __init__(self, X, k, n_iters=10, p=2, distance_metric="cosine"):
        self.k = k
        self.n_iters = n_iters
        self.p = p
        self.train(X, distance_metric)

    def train(self, X, distance_metric):

        self.train_pts = randomize_tensor(X)[:self.k]
        self.train_label = th.tensor(range(self.k), dtype=th.int64, device=X.device)

        for _ in range(self.n_iters):
            labels = self.predict(X, distance_metric)

            for lab in range(self.k):
                self.train_pts[lab] = th.median(X[labels == lab], dim=0)[0]
