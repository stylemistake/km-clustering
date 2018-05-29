import math
import random

from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import norm

class Point:
  def __init__(self, pos, meta = None):
    self.pos = pos
    self.meta = meta
  def __iter__(self):
    for i in self.pos:
      yield i

## Euclidean distance between two points
def eucl_distance(a, b):
  a = tuple(a)
  b = tuple(b)
  total = 0;
  for i in range(0, len(a)):
    total += (a[i] - b[i]) ** 2
  return math.sqrt(total)

def first(items):
  return items[0] if items else None

def last(items):
  return items[-1] if items else None

## Find an item nearest to the predicate
def nearest(items, predicate, as_index = False, distance_fn = eucl_distance):
  result = None
  min_dist = math.inf
  for i, item in enumerate(items):
    cur_dist = distance_fn(item, predicate)
    if (cur_dist < min_dist):
      min_dist = cur_dist
      result = i if as_index else item
  return result

def tuple_apply(fn, a, b):
  size = len(a)
  result = [None] * size
  for i in range(size):
    result[i] = fn(a[i], b[i])
  return tuple(result)

def mean(items):
  return tuple(sum(x) / len(items) for x in zip(*items))

## TODO: support for Point class
def std(items):
  m = mean(items)
  size = len(m)
  s = [0] * size
  n = len(items)
  for x in items:
    x = tuple(x)
    for i in range(0, size):
      s[i] += (x[i] - m[i]) ** 2
  for i in range(0, size):
    s[i] = math.sqrt(s[i] / n)
  return tuple(s)

def mean_normalize(items):
  x_mean = mean(items)
  return [tuple_apply(lambda a, b: a - b, x, x_mean) for x in items]

def find_two_farthest(items):
  length = len(items)
  max_dist = 0
  result = None
  for i in range(0, length - 1):
    for j in range(i, length):
      a = items[i]
      b = items[j]
      cur_dist = distance(a, b)
      if (cur_dist > max_dist):
        max_dist = cur_dist
        result = (a, b)
  return result

## KMeans
def kmeans(points, k = 2, epsilon = 0.001, iterations = 1000):
  ## Initialize cluster centers with a random sample
  cluster_centers = random.sample(points, k)

  ## Iterate
  for it in range(iterations):
    ## Assign points to clusters
    clusters = [[] for _ in range(k)]
    for p, point in enumerate(points):
      c = nearest(cluster_centers, point, as_index = True)
      clusters[c].append(point)

    ## Calculate new cluster centers
    cluster_centers_new = [None] * k
    for c, cluster_points in enumerate(clusters):
      cluster_centers_new[c] = mean(cluster_points)

    ## Evaluate difference between new and old cluster centers
    cluster_center_distances = [eucl_distance(a, b)
      for a, b in zip(cluster_centers, cluster_centers_new)]

    print('iteration: {}, diff: {}'.format(it, cluster_center_distances))

    ## Break loop when change is smaller than epsilon
    if max(cluster_center_distances) < epsilon:
      break

    ## Replace cluster centers with newer and continue iteration
    cluster_centers = cluster_centers_new

  return cluster_centers, clusters

def gaussian_pdf(point, mu, sigma, lam):
  point = tuple(point)
  n = len(point)
  probability = lam
  for i in range(n):
    probability *= norm.pdf(point[i], mu[i], sigma[i][i])
  return probability

## Expectation Minimization
def em_gaussian(points, k = 2, epsilon = 0.001, iterations = 1000):
  params = {
    'mu': [tuple(x) for x in random.sample(points, k)],
    'sig': [[[1, 0], [0, 1]] for _ in range(k)],
    'lambda': [1/k for _ in range(k)],
  }
  count = len(points)

  ## Iterate
  for it in range(iterations):
    clusters = [[] for _ in range(k)]

    ## Expectation
    for point in points:
      max_prob = 0
      max_prob_cluster = None
      for i in range(k):
        prob = gaussian_pdf(point, params['mu'][i], params['sig'][i], params['lambda'][i])
        if prob > max_prob:
          max_prob = prob
          max_prob_cluster = i
      clusters[max_prob_cluster].append(point)

    ## Maximization
    params_new = {}
    params_new['lambda'] = [len(clusters[i]) / float(count) for i in range(k)]
    params_new['mu'] = [mean(clusters[i]) for i in range(k)]
    params_new['sig'] = []
    for i in range(k):
      cluster_std = std(clusters[i])
      cluster_std = [[cluster_std[0], 0], [0, cluster_std[1]]]
      params_new['sig'].append(cluster_std)

    print(params_new)

    ## Evaluate difference between new and old params
    diff = 0
    for i in range(k):
      diff += (params['mu'][i][0] - params_new['mu'][i][0]) ** 2
      diff += (params['mu'][i][1] - params_new['mu'][i][1]) ** 2
    diff = diff ** 0.5

    ## Break loop when change is smaller than epsilon
    if diff < epsilon:
      break

    params = params_new

    print("iteration {}, diff {}".format(it, diff))

  return [], clusters
