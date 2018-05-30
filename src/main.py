import sys
import math
import lib
import random
import argparse
import csv

DISTANCE_FNS = {
  'eucl': {
    'fn': lib.eucl_distance,
    'desc': 'euclidean distance',
  },
  'sad': {
    'fn': lib.sad_distance,
    'desc': 'sum of absolute difference',
  },
  'ssd': {
    'fn': lib.ssd_distance,
    'desc': 'sum of squared difference, aka euclidean norm',
  },
  'chebyshev': {
    'fn': lib.chebyshev_distance,
    'desc': 'chebyshev distance',
  },
}

DISTANCE_FN_HELP = '(' + ', '.join(DISTANCE_FNS.keys()) + ')'

## Setup CLI
parser = argparse.ArgumentParser(prog = 'bin/clustering',
  formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i',
  metavar = '<file>',
  dest = 'input',
  action = 'store',
  default = None,
  help = 'path to the input CSV file')
parser.add_argument('-o',
  metavar = '<file>',
  dest = 'output',
  action = 'store',
  default = None,
  help = 'path to the output CSV file')
parser.add_argument('-c', '--columns',
  metavar = '<list>',
  dest = 'columns',
  action = 'store',
  default = None,
  help = 'specify CSV columns to use (example: 0,1,2)')
parser.add_argument('-k', '--clusters',
  metavar = '<number>',
  dest = 'k',
  action = 'store',
  default = 2,
  help = 'number of clusters')
parser.add_argument('-d', '--distance',
  metavar = '<type>',
  dest = 'distance',
  action = 'store',
  default = 'eucl',
  choices = DISTANCE_FNS.keys(),
  help = 'distance function ' + DISTANCE_FN_HELP)
parser.add_argument('--kmeans',
  dest = 'kmeans',
  action = 'store_true',
  default = False,
  help = 'use KMeans algorithm')
parser.add_argument('--em',
  dest = 'em',
  action = 'store_true',
  default = False,
  help = 'use Expectation Minimization algorithm')
parser.add_argument('--plot',
  dest = 'plot',
  action = 'store_true',
  default = False,
  help = 'show a scatter plot with resulting clusters')
parser.add_argument('--print',
  dest = 'print',
  action = 'store_true',
  default = False,
  help = 'print results to console')

## Parse arguments
args = parser.parse_args()

## Get column mapping (comma separated)
if args.columns is None:
  column_mapping = None
else:
  column_mapping = [int(x) for x in args.columns.split(',')]

## Get data
if args.input:
  ## CSV data from file
  data = []
  with open(args.input) as file:
    reader = csv.reader(file)
    for row in reader:
      if len(row) <= 0:
        continue
      if column_mapping:
        point = tuple([float(row[i]) for i in column_mapping])
      else:
        point = tuple([float(x) for x in row])
      data.append(lib.Point(point, row))
else:
  ## Random data
  data = []
  for _ in range(80):
    data.append(tuple([random.random() *  5 +  0, random.random() *  5 +  0]))
    data.append(tuple([random.random() *  5 + 15, random.random() *  5 + 10]))
    data.append(tuple([random.random() *  6 +  2, random.random() *  6 + 10]))
    data.append(tuple([random.random() * 16 +  2, random.random() * 16 +  2]))
    data.append(tuple([random.random() * 16 +  2, random.random() * 16 +  2]))

## Do calculations
if args.kmeans:
  centers, clusters = lib.kmeans(data, k = int(args.k),
    distance_fn = DISTANCE_FNS[args.distance]['fn'])
elif args.em:
  centers, clusters = lib.em_gaussian(data, k = int(args.k))
else:
  parser.print_help()
  sys.exit()

## Plot results
if args.plot:
  from matplotlib import pyplot
  COLORS = [
    '#cc4444', '#44cc44', '#4444cc',
    '#cc8844', '#44cc88', '#8844cc',
    '#cccc44', '#44cccc', '#cc44cc',
    '#88cc44', '#4488cc', '#cc4488',
  ]
  def plot_scatter(items):
    for i, item in enumerate(items):
      if isinstance(item, list):
        for x in item:
          x = tuple(x)
          pyplot.scatter(x[0], x[1], s = 10, c = COLORS[i], alpha = 0.5)
      else:
        item = tuple(item)
        pyplot.scatter(item[0], item[1], s = 50, c = COLORS[i])
  plot_scatter(centers)
  plot_scatter(clusters)
  pyplot.show()

## Print to console
if args.print:
  for cluster_i, cluster in enumerate(clusters):
    for point in cluster:
      cluster_id = 'cluster-' + str(cluster_i)
      print(cluster_id, point.meta)

## Output to CSV
if args.output:
  with open(args.output, 'w') as file:
    writer = csv.writer(file)
    for cluster_i, cluster in enumerate(clusters):
      for point in cluster:
        cluster_id = 'cluster-' + str(cluster_i)
        row = []
        row.extend(point.meta)
        row.append(cluster_id)
        writer.writerow(row)
